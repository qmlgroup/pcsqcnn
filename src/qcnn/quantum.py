"""Quantum-state building blocks for the Torch PCS-QCNN implementation.

The module provides the full state-space machinery used by the hybrid model:

- FRQI-like image encoding into the canonical state contract
  ``[B, X, CX, Y, CY, F]``;
- Pauli-parameterized feature unitaries
  ``U(theta) = exp(i * sum_alpha theta_alpha P_alpha)``;
- Fourier-domain helpers on the active index registers;
- the reduced Fourier junction that replaces
  ``IQFT -> move_active_qubit_to_condition -> QFT`` exactly;
- condition-aware feature-block multiplexing;
- marginal measurement back to real-valued readout tensors.

Throughout the module, leading axes are batched elementwise unless a helper
explicitly states otherwise. The active index registers live on axes ``1`` and
``3`` of the canonical tensor, while the explicit condition registers live on
axes ``2`` and ``4``.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import torch
from torch import nn

from qcnn.article import warn_for_article_alignment
from qcnn.layout import RegisterLayout2D


def _complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Map a supported real dtype to the matching complex dtype.

    Args:
        real_dtype: Real tensor dtype expected to be ``torch.float32`` or
            ``torch.float64``.

    Returns:
        ``torch.complex64`` for ``torch.float32`` and ``torch.complex128`` for
        ``torch.float64``. Any other dtype falls back to ``torch.complex64`` and
        is rejected earlier by public constructors.
    """

    if real_dtype == torch.float64:
        return torch.complex128
    return torch.complex64


_PAULI_SINGLE_QUBIT = torch.tensor(
    [
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
        [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
        [[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]],
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]],
    ],
    dtype=torch.complex128,
)

_PAULI_BASIS_DEVICE_DTYPE_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_FOURIER_JUNCTION_PHASE_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _validate_brightness_range(brightness_range: tuple[float, float]) -> tuple[float, float]:
    """Normalize and validate an encoder-side brightness interval ``(a, b)``.

    Args:
        brightness_range: Pair ``(a, b)`` used to map normalized grayscale
            inputs into encoder angles.

    Returns:
        The same interval converted to Python ``float`` values.

    Raises:
        ValueError: If the interval cannot be unpacked as two values or does
            not satisfy ``a < b``.
    """

    try:
        start, end = brightness_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "brightness_range must be a pair (a, b) with a < b, "
            f"got {brightness_range!r}."
        ) from exc

    start = float(start)
    end = float(end)
    if start >= end:
        raise ValueError(
            "brightness_range must satisfy a < b, "
            f"got {brightness_range}."
        )
    return start, end


def _feature_qubits_from_pauli_parameter_count(parameter_count: int) -> int:
    """Infer the feature-register width from ``P = 4**f - 1``.

    Args:
        parameter_count: Length of the trailing Pauli-coefficient axis.

    Returns:
        The unique integer ``f >= 1`` such that ``parameter_count = 4**f - 1``.

    Raises:
        ValueError: If ``parameter_count`` does not match the reduced Pauli
            basis size for any positive integer number of qubits.
    """

    if parameter_count < 3:
        raise ValueError(
            "pauli_coefficients_to_unitary expects shape [..., P] with "
            "P = 4**f - 1 for some integer f >= 1, "
            f"got P={parameter_count}."
        )

    reduced = parameter_count + 1
    num_qubits = 0
    while reduced % 4 == 0:
        reduced //= 4
        num_qubits += 1
    if reduced == 1:
        return num_qubits

    raise ValueError(
        "pauli_coefficients_to_unitary expects shape [..., P] with "
        "P = 4**f - 1 for some integer f >= 1, "
        f"got P={parameter_count}."
    )


@lru_cache(maxsize=None)
def _pauli_basis_matrices(num_qubits: int) -> torch.Tensor:
    """Build the reduced multi-qubit Pauli basis in lexicographic order.

    Args:
        num_qubits: Number of feature-register qubits.

    Returns:
        A complex tensor of shape ``[4**num_qubits - 1, 2**num_qubits, 2**num_qubits]``.
        The basis is ordered lexicographically over the single-qubit labels
        ``(I, X, Y, Z)`` with MSB-left convention and the all-identity string
        omitted.

    Raises:
        ValueError: If ``num_qubits < 1``.

    Notes:
        The basis is cached with ``functools.lru_cache`` because it is reused by
        every call to ``pauli_coefficients_to_unitary`` with the same number of
        feature qubits.
    """

    if num_qubits < 1:
        raise ValueError(f"num_qubits must be at least 1, got {num_qubits}.")

    basis: list[torch.Tensor] = []
    for labels in product(range(4), repeat=num_qubits):
        if all(label == 0 for label in labels):
            continue

        matrix = _PAULI_SINGLE_QUBIT[labels[0]]
        for label in labels[1:]:
            matrix = torch.kron(matrix, _PAULI_SINGLE_QUBIT[label])
        basis.append(matrix)

    return torch.stack(basis, dim=0)


def _materialized_pauli_basis_matrices(
    num_qubits: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the reduced Pauli basis cached on a specific device and dtype.

    Args:
        num_qubits: Number of feature-register qubits.
        device: Target device for the returned basis tensor.
        dtype: Target complex dtype for the returned basis tensor.

    Returns:
        The reduced Pauli basis with shape
        ``[4**num_qubits - 1, 2**num_qubits, 2**num_qubits]`` materialized on
        ``device`` with dtype ``dtype``.

    Notes:
        The CPU master copy is cached by ``_pauli_basis_matrices``. This helper
        adds a second cache keyed by ``(num_qubits, device, dtype)`` so repeated
        calls do not re-upload the basis tensor from host memory to the GPU on
        every forward pass.
    """

    cache_key = (num_qubits, str(device), dtype)
    cached_basis = _PAULI_BASIS_DEVICE_DTYPE_CACHE.get(cache_key)
    if cached_basis is None:
        cached_basis = _pauli_basis_matrices(num_qubits).to(device=device, dtype=dtype)
        _PAULI_BASIS_DEVICE_DTYPE_CACHE[cache_key] = cached_basis
    return cached_basis


def pauli_coefficients_to_unitary(params: torch.Tensor) -> torch.Tensor:
    """Exponentiate Pauli-basis coefficients into batched unitaries.

    Contract:
        ``params`` must have shape ``[..., P]`` with ``P = 4**f - 1`` for some
        integer ``f >= 1``. The last axis is ordered lexicographically over the
        single-qubit labels ``(I, X, Y, Z)`` with MSB-left convention and the
        all-identity string omitted; for ``f = 1`` this reduces to ``[X, Y, Z]``.

    Formula:
        Let ``H(theta) = sum_alpha theta_alpha P_alpha`` over the reduced Pauli
        basis. The returned block is
        ``U(theta) = exp(i * H(theta))``.

    Returns:
        A complex tensor of shape ``[..., 2**f, 2**f]``. All leading axes are
        treated elementwise, so different parameter blocks do not interact.

    Raises:
        ValueError: If ``params`` is scalar, not floating-point, or the last
            axis length is not of the form ``4**f - 1``.

    Notes:
        ``float32`` inputs produce ``complex64`` outputs and ``float64`` inputs
        produce ``complex128`` outputs. The dominant cost is the batched matrix
        exponential on the final two axes.
    """

    if params.ndim < 1:
        raise ValueError(
            "pauli_coefficients_to_unitary expects shape [..., P], "
            f"got scalar tensor with shape {tuple(params.shape)}."
        )
    if not torch.is_floating_point(params):
        raise ValueError(
            "pauli_coefficients_to_unitary expects a real floating tensor, "
            f"got dtype {params.dtype}."
        )

    num_qubits = _feature_qubits_from_pauli_parameter_count(int(params.shape[-1]))
    complex_dtype = _complex_dtype_for(params.dtype)
    basis = _materialized_pauli_basis_matrices(
        num_qubits,
        device=params.device,
        dtype=complex_dtype,
    )
    generator = torch.einsum("...p,pij->...ij", params.to(dtype=complex_dtype), basis)
    return torch.linalg.matrix_exp(basis.new_tensor(1j) * generator)


def _apply_qft_2d(state: torch.Tensor) -> torch.Tensor:
    """Apply separable Fourier transforms to the active x/y axes.

    Contract:
        ``state`` must have canonical shape ``[B, X, CX, Y, CY, F]`` and a
        complex dtype.

    Formula:
        The helper applies ``torch.fft.fft`` along axis ``1`` and then axis ``3``.
        In other words, for fixed ``(b, cx, cy, f)`` it computes the separable
        2D transform over the active index amplitudes only.

    Returns:
        A tensor with the same shape and dtype as the input.

    Raises:
        ValueError: If the tensor is not rank-6 or is not complex-valued.

    Notes:
        The implementation uses PyTorch's default FFT normalization. No
        ``norm="ortho"`` scaling is applied because the model always uses QFT
        and IQFT in matched pairs.
    """

    if state.ndim != 6:
        raise ValueError(
            "_apply_qft_2d expects state tensors with shape [B, X, CX, Y, CY, F], "
            f"got {tuple(state.shape)}."
        )
    if not torch.is_complex(state):
        raise ValueError(f"_apply_qft_2d expects a complex state tensor, got dtype {state.dtype}.")

    return torch.fft.fft(torch.fft.fft(state, dim=1), dim=3)


def _apply_iqft_2d(state: torch.Tensor) -> torch.Tensor:
    """Apply separable inverse Fourier transforms to the active x/y axes.

    Contract:
        ``state`` must have canonical shape ``[B, X, CX, Y, CY, F]`` and a
        complex dtype.

    Formula:
        The helper applies ``torch.fft.ifft`` along axis ``1`` and then axis ``3``.
        It is the exact inverse of ``_apply_qft_2d`` under PyTorch's default FFT
        convention.

    Returns:
        A tensor with the same shape and dtype as the input.

    Raises:
        ValueError: If the tensor is not rank-6 or is not complex-valued.
    """

    if state.ndim != 6:
        raise ValueError(
            "_apply_iqft_2d expects state tensors with shape [B, X, CX, Y, CY, F], "
            f"got {tuple(state.shape)}."
        )
    if not torch.is_complex(state):
        raise ValueError(f"_apply_iqft_2d expects a complex state tensor, got dtype {state.dtype}.")

    return torch.fft.ifft(torch.fft.ifft(state, dim=1), dim=3)


def _apply_unitary_to_active_axis_1d(
    state: torch.Tensor,
    unitary: torch.Tensor,
    *,
    active_axis: int,
) -> torch.Tensor:
    """Apply a dense unitary matrix to one active spatial register axis.

    Contract:
        ``state`` must have canonical shape ``[B, X, CX, Y, CY, F]`` and a
        complex dtype. ``unitary`` must be a complex square matrix whose size
        matches the selected active axis. ``active_axis`` must be ``1`` for the
        active x-register or ``3`` for the active y-register.

    Formula:
        For the selected active axis, the helper computes
        ``out[i, ...] = sum_j unitary[i, j] * state[j, ...]`` while treating
        all remaining axes elementwise.

    Returns:
        A complex tensor with the same shape and dtype as ``state``.

    Raises:
        ValueError: If ``state`` is not canonical rank-6 complex data, if
            ``unitary`` is not a complex square matrix, if ``active_axis`` is
            unsupported, or if the matrix size does not match the chosen axis.

    Notes:
        The implementation moves the selected active axis to the front, applies
        one batched ``einsum``, and then restores the canonical axis order.
    """

    if state.ndim != 6:
        raise ValueError(
            "_apply_unitary_to_active_axis_1d expects state tensors with shape "
            "[B, X, CX, Y, CY, F], "
            f"got {tuple(state.shape)}."
        )
    if active_axis not in {1, 3}:
        raise ValueError(f"active_axis must be 1 or 3, got {active_axis}.")
    if not torch.is_complex(state):
        raise ValueError(
            "_apply_unitary_to_active_axis_1d expects a complex state tensor, "
            f"got dtype {state.dtype}."
        )
    if unitary.ndim != 2:
        raise ValueError(
            "_apply_unitary_to_active_axis_1d expects unitary matrices with shape [N, N], "
            f"got {tuple(unitary.shape)}."
        )
    if unitary.shape[0] != unitary.shape[1]:
        raise ValueError(
            "_apply_unitary_to_active_axis_1d expects a square unitary matrix, "
            f"got shape {tuple(unitary.shape)}."
        )
    if not torch.is_complex(unitary):
        raise ValueError(
            "_apply_unitary_to_active_axis_1d expects a complex unitary matrix, "
            f"got dtype {unitary.dtype}."
        )

    active_dim = int(state.shape[active_axis])
    if unitary.shape[0] != active_dim:
        raise ValueError(
            "_apply_unitary_to_active_axis_1d matrix size must match the selected active axis: "
            f"got {tuple(unitary.shape)} for axis size {active_dim}."
        )

    moved_state = torch.movedim(state, active_axis, 0)
    transformed = torch.einsum("ij,j...->i...", unitary.to(device=state.device, dtype=state.dtype), moved_state)
    return torch.movedim(transformed, 0, active_axis)


def _materialized_fourier_junction_phase(
    active_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the cached Fourier-junction phase vector for one active dimension.

    Args:
        active_dim: Active register size before pooling. Must be even and at
            least ``2``.
        device: Target device for the returned phase tensor.
        dtype: Target complex dtype for the returned phase tensor.

    Returns:
        A 1D complex tensor with shape ``[active_dim / 2]`` containing the
        phase values ``exp(2 pi i q / active_dim)`` for ``q in [0, active_dim / 2)``.

    Raises:
        ValueError: If ``active_dim < 2`` or ``active_dim`` is odd.

    Notes:
        The tensor is cached per ``(active_dim, device, dtype)`` so repeated
        Fourier-junction calls do not rebuild the same ``arange + exp`` phase
        vector on every forward pass.
    """

    if active_dim < 2:
        raise ValueError(
            "_materialized_fourier_junction_phase requires at least two active basis states, "
            f"got {active_dim}."
        )
    if active_dim % 2 != 0:
        raise ValueError(
            "_materialized_fourier_junction_phase requires an even active dimension, "
            f"got {active_dim}."
        )

    cache_key = (active_dim, str(device), dtype)
    cached_phase = _FOURIER_JUNCTION_PHASE_CACHE.get(cache_key)
    if cached_phase is None:
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        q = torch.arange(active_dim // 2, device=device, dtype=real_dtype)
        cached_phase = torch.exp((2j * torch.pi / active_dim) * q).to(dtype=dtype)
        _FOURIER_JUNCTION_PHASE_CACHE[cache_key] = cached_phase
    return cached_phase


def _apply_fourier_junction_1d(
    state: torch.Tensor,
    *,
    active_axis: int,
    condition_axis: int,
) -> torch.Tensor:
    """Apply the closed-form Fourier junction along one active index axis.

    Args:
        state: Canonical state tensor with shape ``[B, X, CX, Y, CY, F]``.
        active_axis: Axis of the active register to shrink.
        condition_axis: Matching condition-register axis to grow.

    Contract:
        The helper is exactly equivalent to applying ``IQFT`` on the chosen
        active axis, moving one qubit from that active register into the
        matching condition register, and then applying ``QFT`` on the reduced
        active register.

    Formula:
        With ``N`` the active dimension before pooling, ``q in [0, N/2)``, and
        ``b in {0, 1}``, the reduced junction is
        ``out[q, b] = 0.5 * exp(2 pi i q b / N) * (in[q] + (-1)**b * in[q + N/2])``.

    Returns:
        A tensor with unchanged outer axis order and updated register sizes:
        the chosen active dimension is halved and the chosen condition
        dimension is doubled.

    Raises:
        ValueError: If the tensor is not rank-6, is not complex-valued, or the
            selected active dimension is odd or smaller than ``2``.

    Notes:
        The static phase vector is cached per active dimension, device, and
        dtype so repeated calls with the same layer geometry avoid rebuilding
        the same ``arange + exp`` tensor.
    """

    if state.ndim != 6:
        raise ValueError(
            "_apply_fourier_junction_1d expects state tensors with shape [B, X, CX, Y, CY, F], "
            f"got {tuple(state.shape)}."
        )
    if not torch.is_complex(state):
        raise ValueError(
            f"_apply_fourier_junction_1d expects a complex state tensor, got dtype {state.dtype}."
        )

    active_dim = int(state.shape[active_axis])
    if active_dim < 2:
        raise ValueError(
            "_apply_fourier_junction_1d requires at least two active basis states "
            f"along axis {active_axis}, got {active_dim}."
        )
    if active_dim % 2 != 0:
        raise ValueError(
            "_apply_fourier_junction_1d requires an even active dimension, "
            f"got {active_dim} on axis {active_axis}."
        )

    state = torch.movedim(state, (active_axis, condition_axis), (0, 1))
    half_dim = active_dim // 2
    low = state[:half_dim]
    high = state[half_dim:]
    phase = _materialized_fourier_junction_phase(
        active_dim,
        device=state.device,
        dtype=state.dtype,
    )
    reshape_shape = (half_dim, 1) + (1,) * (state.ndim - 2)
    phase = phase.reshape(reshape_shape)

    branch0 = 0.5 * (low + high)
    branch1 = 0.5 * phase * (low - high)
    out = torch.cat((branch0, branch1), dim=1)
    return torch.movedim(out, (0, 1), (active_axis, condition_axis))


def _apply_fourier_junction_2d(
    state: torch.Tensor,
    layout: RegisterLayout2D,
) -> tuple[torch.Tensor, RegisterLayout2D]:
    """Collapse adjacent IQFT/QFT layers across one pooling step on each axis.

    Args:
        state: Canonical complex state tensor ``[B, X, CX, Y, CY, F]``.
        layout: Layout describing ``state`` before pooling.

    Returns:
        A pair ``(next_state, next_layout)`` equivalent to
        ``_apply_iqft_2d(state)``, followed by
        ``move_active_qubit_to_condition(..., x=1, y=1)``, followed by
        ``_apply_qft_2d(...)`` on the reduced active registers.

    Raises:
        ValueError: If ``state`` does not match ``layout``, is not complex, or
            if either active index register has fewer than two basis states.

    Notes:
        The helper applies the x-junction and y-junction sequentially. It is
        used by default inside ``PCSQCNN`` to avoid materializing an explicit
        IQFT/pooling/QFT triple between neighboring layers.
    """

    layout.validate_state_shape(state)
    if not torch.is_complex(state):
        raise ValueError(
            f"_apply_fourier_junction_2d expects a complex state tensor, got dtype {state.dtype}."
        )
    if layout.x_active_dim < 2:
        raise ValueError(
            "Fourier junction requires at least two active x-register basis states, "
            f"got {layout.x_active_dim}."
        )
    if layout.y_active_dim < 2:
        raise ValueError(
            "Fourier junction requires at least two active y-register basis states, "
            f"got {layout.y_active_dim}."
        )

    state = _apply_fourier_junction_1d(state, active_axis=1, condition_axis=2)
    x_layout = RegisterLayout2D(
        image_size=layout.image_size,
        feature_qubits=layout.feature_qubits,
        x_condition_qubits=layout.x_condition_qubits + 1,
        y_condition_qubits=layout.y_condition_qubits,
    )
    state = _apply_fourier_junction_1d(state, active_axis=3, condition_axis=4)
    output_layout = RegisterLayout2D(
        image_size=layout.image_size,
        feature_qubits=layout.feature_qubits,
        x_condition_qubits=x_layout.x_condition_qubits,
        y_condition_qubits=x_layout.y_condition_qubits + 1,
    )
    return state, output_layout


class ModeMultiplexer2D(nn.Module):
    """Apply condition- and mode-dependent feature unitaries to a canonical QCNN state.

    Args:
        layout: Register layout describing the canonical state shape.
        x_condition_qubits_to_use: Number of most-significant x-condition bits
            used to select a block.
        y_condition_qubits_to_use: Number of most-significant y-condition bits
            used to select a block.
        multiplexer_init_scale: Standard deviation of the Gaussian Pauli-coefficient
            initialization used for each trainable feature block.
        dtype: Real dtype used for the trainable Pauli coefficients.

    Contract:
        The input state must have canonical shape ``[B, X, CX, Y, CY, F]``. For
        every selected branch ``(u, v)`` and active mode ``(x, y)``, the layer
        stores one feature-unitary block
        ``U[u, v, x, y] in C^{F x F}``.

    Formula:
        The trainable parameters are real Pauli coefficients with shape
        ``[2**k_x, 2**k_y, X, Y, 4**f - 1]``. They are converted into blocks via
        ``U = exp(i * sum_alpha theta_alpha P_alpha)`` and applied only on the
        feature axis.

    Notes:
        Selected condition bits are taken from the most-significant positions of
        the explicit condition registers. Because
        ``move_active_qubit_to_condition`` inserts newly pooled qubits as new
        MSBs, choosing the first ``k`` bits means conditioning on the most
        recently measured outcomes. Unselected condition bits behave like extra
        batch axes and are not mixed by the layer.
    """

    def __init__(
        self,
        layout: RegisterLayout2D,
        *,
        x_condition_qubits_to_use: int = 0,
        y_condition_qubits_to_use: int = 0,
        multiplexer_init_scale: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        if x_condition_qubits_to_use < 0:
            raise ValueError(
                "x_condition_qubits_to_use must be non-negative, "
                f"got {x_condition_qubits_to_use}."
            )
        if y_condition_qubits_to_use < 0:
            raise ValueError(
                "y_condition_qubits_to_use must be non-negative, "
                f"got {y_condition_qubits_to_use}."
            )
        if x_condition_qubits_to_use > layout.x_condition_qubits:
            raise ValueError(
                "x_condition_qubits_to_use cannot exceed layout.x_condition_qubits: "
                f"got {x_condition_qubits_to_use} > {layout.x_condition_qubits}."
            )
        if y_condition_qubits_to_use > layout.y_condition_qubits:
            raise ValueError(
                "y_condition_qubits_to_use cannot exceed layout.y_condition_qubits: "
                f"got {y_condition_qubits_to_use} > {layout.y_condition_qubits}."
            )
        if multiplexer_init_scale <= 0:
            raise ValueError(
                "multiplexer_init_scale must be positive, "
                f"got {multiplexer_init_scale}."
            )

        self.layout = layout
        self.real_dtype = dtype
        self.x_condition_qubits_to_use = x_condition_qubits_to_use
        self.y_condition_qubits_to_use = y_condition_qubits_to_use
        self.multiplexer_init_scale = float(multiplexer_init_scale)
        self.x_selected_condition_dim = 2 ** x_condition_qubits_to_use
        self.y_selected_condition_dim = 2 ** y_condition_qubits_to_use
        pauli_parameter_count = 4 ** layout.feature_qubits - 1
        self.block_parameters = nn.Parameter(
            torch.randn(
                (
                    self.x_selected_condition_dim,
                    self.y_selected_condition_dim,
                    layout.x_active_dim,
                    layout.y_active_dim,
                    pauli_parameter_count,
                ),
                dtype=dtype,
            )
            * self.multiplexer_init_scale
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the selected feature block to every canonical state slice.

        Args:
            state: Complex tensor with shape ``[B, X, CX, Y, CY, F]`` matching
                ``self.layout``.

        Returns:
            A complex tensor with the same shape as the input.

        Raises:
            ValueError: If ``state`` does not match the expected layout or is
                not complex-valued.

        Notes:
            The implementation reshapes the condition axes into
            ``selected x remaining`` factors and then performs a batched
            ``einsum`` over the feature axis. Computational cost scales with the
            number of active modes and selected condition branches.
        """

        self.layout.validate_state_shape(state)
        if not torch.is_complex(state):
            raise ValueError(
                "ModeMultiplexer2D expects a complex state tensor, "
                f"got dtype {state.dtype}."
            )

        batch_size, x_dim, cx_dim, y_dim, cy_dim, feature_dim = state.shape
        x_remaining_condition_dim = cx_dim // self.x_selected_condition_dim
        y_remaining_condition_dim = cy_dim // self.y_selected_condition_dim

        blocks = pauli_coefficients_to_unitary(self.block_parameters).to(dtype=state.dtype)
        blocks = blocks.to(device=state.device)

        split_state = state.reshape(
            batch_size,
            x_dim,
            self.x_selected_condition_dim,
            x_remaining_condition_dim,
            y_dim,
            self.y_selected_condition_dim,
            y_remaining_condition_dim,
            feature_dim,
        )
        transformed = torch.einsum("uvxyij,bxuryvwj->bxuryvwi", blocks, split_state)
        return transformed.reshape(batch_size, x_dim, cx_dim, y_dim, cy_dim, feature_dim)


class FrqiEncoder2D(nn.Module):
    """Encode grayscale images into the canonical QCNN wavefunction tensor layout.

    Args:
        feature_qubits: Number of qubits in the feature register.
        brightness_range: Angle interval ``(a, b)`` used to map normalized
            grayscale values ``x in [0, 1]`` into encoder arguments
            ``p = a + (b - a) * x`` before applying ``sin`` / ``cos``.
        dtype: Real dtype used for the brightness amplitudes before conversion
            to the matching complex dtype.

    Formula:
        For every normalized pixel value ``x_uv``, the encoder first computes
        ``p_uv = a + (b - a) * x_uv`` and then writes
        ``|phi_uv> = sin(p_uv)|0> + cos(p_uv)|1>`` into the least-significant
        feature qubit. Additional feature qubits, when present, are initialized
        to ``|0...0>``.

    Contract:
        ``forward(images)`` expects shape ``[B, X, Y]`` containing normalized
        grayscale images and returns a complex tensor of shape
        ``[B, X, 1, Y, 1, F]`` where ``F = 2**feature_qubits``.

    Notes:
        The encoder intentionally omits the article's global
        ``1 / sqrt(XY)`` normalization factor. Encoded sample norms are
        ``sqrt(XY)`` rather than ``1``, and the constructor emits the
        documented runtime warning for that mismatch.
    """

    def __init__(
        self,
        feature_qubits: int = 1,
        *,
        brightness_range: tuple[float, float] = (-1.0, 1.0),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        if feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {feature_qubits}.")

        self.feature_qubits = feature_qubits
        self.brightness_range = _validate_brightness_range(brightness_range)
        self.real_dtype = dtype
        self.complex_dtype = _complex_dtype_for(dtype)

        warn_for_article_alignment(
            brightness_range=self.brightness_range,
            include_encoder_mismatch=True,
            stacklevel=2,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of square images into canonical complex states.

        Args:
            images: Real tensor with shape ``[B, X, Y]`` containing normalized
                grayscale values.

        Returns:
            A complex tensor with shape ``[B, X, 1, Y, 1, F]``.

        Raises:
            ValueError: If ``images`` is not rank-3 or is not square.
        """

        if images.ndim != 3:
            raise ValueError(
                "FrqiEncoder2D expects image tensors with shape [B, X, Y], "
                f"got {tuple(images.shape)}."
            )

        batch_size, x_dim, y_dim = images.shape
        if x_dim != y_dim:
            raise ValueError(f"Input images must be square, got X={x_dim}, Y={y_dim}.")

        layout = RegisterLayout2D(image_size=x_dim, feature_qubits=self.feature_qubits)
        images = images.to(dtype=self.real_dtype)
        start, end = self.brightness_range
        angles = start + (end - start) * images

        color_subspace = torch.stack(
            (
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        )
        if layout.feature_dim == 2:
            feature_state = color_subspace
        else:
            auxiliary_zeros = torch.zeros(
                (*color_subspace.shape[:-1], layout.feature_dim - 2),
                dtype=self.real_dtype,
                device=images.device,
            )
            feature_state = torch.cat((color_subspace, auxiliary_zeros), dim=-1)

        return feature_state[:, :, None, :, None, :].to(dtype=self.complex_dtype)


class MarginalMeasurement2D(nn.Module):
    """Measure a canonical QCNN state into normalized real marginals.

    Contract:
        Input states must have canonical shape ``[B, X, CX, Y, CY, F]`` and a
        complex dtype. The output has shape ``[B, X, Y, F]``.

    Formula:
        Let ``psi`` denote the input amplitude tensor. The output marginal is
        ``m[b, x, y, f] = (1 / (X * CX * Y * CY)) * sum_{cx, cy} |psi[b, x, cx, y, cy, f]|^2``.

    Notes:
        The explicit normalization by ``X * CX * Y * CY`` compensates for the
        encoder's deliberate omission of the article's global
        ``1 / sqrt(XY)`` factor.
    """

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Measure a canonical state tensor into normalized real marginals.

        Args:
            state: Complex tensor with shape ``[B, X, CX, Y, CY, F]``.

        Returns:
            A real tensor with shape ``[B, X, Y, F]``.

        Raises:
            ValueError: If ``state`` is not rank-6 or is not complex-valued.
        """

        if state.ndim != 6:
            raise ValueError(
                "MarginalMeasurement2D expects state tensors with shape "
                "[B, X, CX, Y, CY, F], "
                f"got {tuple(state.shape)}."
            )
        if not torch.is_complex(state):
            raise ValueError(
                "MarginalMeasurement2D expects a complex state tensor, "
                f"got dtype {state.dtype}."
            )

        probabilities = state.real.square() + state.imag.square()
        marginal = probabilities.sum(dim=(2, 4))
        normalization = state.shape[1] * state.shape[2] * state.shape[3] * state.shape[4]
        return marginal / normalization


class FiniteShotHistogram2D(nn.Module):
    """Convert exact readout probabilities into finite-shot frequency histograms.

    Contract:
        Input tensors must have shape ``[B, X, Y, F]`` and a real floating
        dtype. The output has the same shape and dtype.

    Notes:
        ``shot_budget=None`` and ``shot_budget=0`` both mean infinite-shot
        mode, in which the layer is an identity. The same pass-through behavior
        is used while the module is in training mode so the surrounding model
        remains differentiable under the existing training loop.
    """

    def __init__(self, shot_budget: int | None = None) -> None:
        super().__init__()
        self.shot_budget = self._normalize_shot_budget(shot_budget)

    @staticmethod
    def _normalize_shot_budget(shot_budget: int | None) -> int | None:
        if shot_budget is None:
            return None
        if isinstance(shot_budget, bool) or not isinstance(shot_budget, int):
            raise ValueError(
                "shot_budget must be an integer, 0, or None, "
                f"got {shot_budget!r}."
            )
        if shot_budget < 0:
            raise ValueError(f"shot_budget must be non-negative, got {shot_budget}.")
        if shot_budget == 0:
            return None
        return shot_budget

    @staticmethod
    def _is_functorch_transform_active(*tensors: torch.Tensor) -> bool:
        functorch_state = getattr(torch._C, "_functorch", None)
        if functorch_state is None:
            return False
        if functorch_state.maybe_current_level() is not None:
            return True
        return any(
            functorch_state.is_batchedtensor(tensor)
            or functorch_state.is_gradtrackingtensor(tensor)
            for tensor in tensors
        )

    @staticmethod
    def _validate_probabilities(probabilities: torch.Tensor) -> None:
        if probabilities.ndim != 4:
            raise ValueError(
                "FiniteShotHistogram2D expects tensors with shape [B, X, Y, F], "
                f"got {tuple(probabilities.shape)}."
            )
        if not torch.is_floating_point(probabilities):
            raise ValueError(
                "FiniteShotHistogram2D expects a real floating tensor, "
                f"got dtype {probabilities.dtype}."
            )
        if FiniteShotHistogram2D._is_functorch_transform_active(probabilities):
            return
        if not torch.isfinite(probabilities).all():
            raise ValueError("FiniteShotHistogram2D expects finite probabilities.")
        if torch.any(probabilities < 0):
            raise ValueError("FiniteShotHistogram2D expects non-negative probabilities.")

    def _sample_histogram_from_valid_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        flattened = probabilities.reshape(probabilities.shape[0], -1)
        totals = flattened.sum(dim=1, keepdim=True)
        if (
            not self._is_functorch_transform_active(probabilities, flattened, totals)
            and torch.any(totals <= 0)
        ):
            raise ValueError("FiniteShotHistogram2D expects each sample to have positive total mass.")
        normalized = flattened / totals.clamp_min(torch.finfo(flattened.dtype).tiny)
        sample_indices = torch.multinomial(normalized, num_samples=self.shot_budget, replacement=True)
        counts = torch.zeros_like(flattened)
        counts.scatter_add_(
            dim=1,
            index=sample_indices,
            src=torch.ones_like(sample_indices, dtype=flattened.dtype),
        )
        histogram = counts / float(self.shot_budget)
        return histogram.reshape_as(probabilities)

    def sample_repeated_histograms(
        self,
        probabilities: torch.Tensor,
        *,
        repetitions: int,
        block_size: int = 100,
    ) -> torch.Tensor:
        """Sample repeated finite-shot histograms in repetition blocks."""

        if repetitions <= 0:
            raise ValueError(f"repetitions must be positive, got {repetitions}.")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")

        self._validate_probabilities(probabilities)
        if self.training or self.shot_budget is None:
            return probabilities.unsqueeze(0).expand(repetitions, *probabilities.shape).clone()

        flattened = probabilities.reshape(probabilities.shape[0], -1)
        totals = flattened.sum(dim=1, keepdim=True)
        if (
            not self._is_functorch_transform_active(probabilities, flattened, totals)
            and torch.any(totals <= 0)
        ):
            raise ValueError("FiniteShotHistogram2D expects each sample to have positive total mass.")
        normalized = flattened / totals.clamp_min(torch.finfo(flattened.dtype).tiny)

        chunks: list[torch.Tensor] = []
        batch_size = probabilities.shape[0]
        flat_dim = flattened.shape[1]
        for start in range(0, repetitions, block_size):
            current_block_size = min(block_size, repetitions - start)
            block_probabilities = (
                normalized.unsqueeze(0)
                .expand(current_block_size, -1, -1)
                .reshape(current_block_size * batch_size, flat_dim)
            )
            sample_indices = torch.multinomial(
                block_probabilities,
                num_samples=self.shot_budget,
                replacement=True,
            )
            counts = torch.zeros_like(block_probabilities)
            counts.scatter_add_(
                dim=1,
                index=sample_indices,
                src=torch.ones_like(sample_indices, dtype=block_probabilities.dtype),
            )
            histogram = counts / float(self.shot_budget)
            chunks.append(
                histogram.reshape(current_block_size, *probabilities.shape)
            )
        return torch.cat(chunks, dim=0)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Sample per-example multinomial histograms from exact readout probabilities."""
        self._validate_probabilities(probabilities)

        if self.training or self.shot_budget is None:
            return probabilities
        return self._sample_histogram_from_valid_probabilities(probabilities)
