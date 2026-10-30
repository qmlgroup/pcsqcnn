"""QCS-QCNN baseline: qubit-cyclic-shift convolution on the FRQI index registers.

The module implements the address-encoding QCS-QCNN of the article's
Fig. 1(a,c) as a matched control for ``PCSQCNN``. It keeps every part of the
PCS-QCNN pipeline except the convolution itself:

- the same ``FrqiEncoder2D`` state ``[B, X, CX, Y, CY, F]``;
- the same deferred-measurement pooling of the least-significant active index
  qubit per axis after every non-final layer, with later layers conditioned on
  the most recently pooled bit of each axis;
- the same ``MarginalMeasurement2D`` full readout, optional finite-shot
  sampling, and linear head, so the head is identical to the PCS-QCNN head.

Where the PCS-QCNN applies ``QFT -> Fourier-mode multiplexer -> IQFT``, each
QCS-QCNN layer applies ``K`` sublayers. Every sublayer evolves the active x
register together with the feature register, and then the active y register
together with the feature register, under a translation-invariant local
generator

``U_x = exp(i * sum_{j=0}^{n-1} tau_j(h_x))``,
``h_x = sum_alpha theta_alpha P_alpha``,

where ``tau_j`` places the shared term ``h_x`` on the neighbouring index qubits
``(j, j + 1 mod n)`` (periodic wrap, as in Fig. 1(a)) and the feature register,
and ``P_alpha`` runs over the full Pauli basis of those ``2 + n_f`` qubits.
Because the generator is a sum over all cyclic placements of one shared term,
``U_x`` commutes exactly with the qubit cyclic shift ``S`` of the x register for
every register size; first-order Trotterization of the same generator gives the
brickwork of shared local gates drawn in Fig. 1(a). Under address encoding, ``S``
does not act as a pixel translation (Lemma 1), so this layer family is
QCS-equivariant but not PCS-equivariant.

Simulation:
    Every generator commutes with ``S``, so it is block-diagonal in the
    eigenbasis of ``S``, the cyclic analogue of the Fourier basis that
    diagonalizes the PCS translation. The module expands the shared Pauli
    terms directly in that basis, exponentiates each symmetry block with an
    eigendecomposition (unitary to working precision), and applies the
    block-diagonal unitaries to the state after rotating the active index
    registers into the same basis.

The model always returns logits. Softmax and the training loss remain outside
the module.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch
from torch import nn

from qcnn.hybrid import _build_layer_layouts, _flatten_readout_histogram, _validate_image_spatial_shape
from qcnn.layout import RegisterLayout2D, move_active_qubit_to_condition
from qcnn.quantum import (
    FiniteShotHistogram2D,
    FrqiEncoder2D,
    MarginalMeasurement2D,
    _materialized_pauli_basis_matrices,
    hermitian_exponential,
)

LOCAL_INDEX_QUBITS = 2
"""Number of neighbouring index qubits touched by one local generator term."""

_DEVICE_TENSOR_CACHE: dict[tuple[str, int, str, torch.dtype], torch.Tensor] = {}


def local_term_parameter_count(feature_qubits: int) -> int:
    """Return the number of Pauli coefficients of one local term, ``4**(2 + n_f)``."""

    return 4 ** (LOCAL_INDEX_QUBITS + feature_qubits)


def _bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - position)) & 1 for position in range(width)]


@lru_cache(maxsize=None)
@torch.inference_mode(False)
def _local_term_placements(index_qubits: int, feature_qubits: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return the gather data that places one local term on every cyclic qubit pair.

    Contract:
        The register basis index is ``t = x * F + f`` with the ``n`` index bits
        of ``x`` MSB-first and ``F = 2**feature_qubits``. For pair ``j`` the
        local term acts on index qubits ``(j, j + 1 mod n)`` and the feature
        register, in that MSB-left order. The result holds one pair
        ``(local_index, rest_index)`` of long tensors with shape ``[2**n * F]``
        per placement: ``local_index[t]`` is the basis index of ``t`` on the
        local qubits and ``rest_index[t]`` labels the untouched index qubits.

    Formula:
        ``tau_j(h)[t, t'] = h[local_index[t], local_index[t']] * [rest_index[t] == rest_index[t']]``.
    """

    if index_qubits < LOCAL_INDEX_QUBITS:
        raise ValueError(f"Local pair terms need at least two index qubits, got {index_qubits}.")
    feature_dim = 2**feature_qubits
    placements = []
    for first in range(index_qubits):
        second = (first + 1) % index_qubits
        rest = [qubit for qubit in range(index_qubits) if qubit not in (first, second)]
        local_index = []
        rest_index = []
        for x in range(2**index_qubits):
            bits = _bits(x, index_qubits)
            for f in range(feature_dim):
                local_index.append((bits[first] * 2 + bits[second]) * feature_dim + f)
                rest_value = 0
                for qubit in rest:
                    rest_value = rest_value * 2 + bits[qubit]
                rest_index.append(rest_value)
        placements.append((torch.tensor(local_index), torch.tensor(rest_index)))
    return tuple(placements)


def translation_invariant_generator(local_terms: torch.Tensor, *, index_qubits: int, feature_qubits: int) -> torch.Tensor:
    """Sum one local Hermitian term over all cyclic placements on an index register.

    Contract:
        ``local_terms`` has shape ``[..., 4 F, 4 F]`` and acts on two index
        qubits followed by the feature register. The result has shape
        ``[..., 2**n F, 2**n F]`` on the register ``index (x) feature``.

    Formula:
        ``H = sum_{j=0}^{n-1} tau_j(h)``, where ``tau_j`` places ``h`` on the
        index qubits ``(j, j + 1 mod n)``.
    """

    generator = None
    for local_index, rest_index in _local_term_placements(index_qubits, feature_qubits):
        local_index = local_index.to(local_terms.device)
        same_rest = (rest_index[:, None] == rest_index[None, :]).to(device=local_terms.device)
        placed = local_terms[..., local_index[:, None], local_index[None, :]] * same_rest
        generator = placed if generator is None else generator + placed
    return generator


def cyclic_shift_matrix(index_qubits: int, feature_qubits: int = 0) -> torch.Tensor:
    """Return the qubit cyclic shift ``S`` on ``index (x) feature`` as a real permutation matrix.

    Formula:
        ``S |q_0 q_1 ... q_{n-1}> = |q_{n-1} q_0 ... q_{n-2}>`` on the index
        qubits and the identity on the feature register.
    """

    dimension = 2**index_qubits
    shift = torch.zeros(dimension, dimension, dtype=torch.float64)
    for x in range(dimension):
        bits = _bits(x, index_qubits)
        rotated = [bits[-1], *bits[:-1]]
        shift[int("".join(map(str, rotated)), 2), x] = 1.0
    return torch.kron(shift, torch.eye(2**feature_qubits, dtype=torch.float64))


@lru_cache(maxsize=None)
@torch.inference_mode(False)
def cyclic_shift_eigenbasis(index_qubits: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return a unitary that block-diagonalizes every operator commuting with ``S``.

    Contract:
        Returns ``(basis, sizes)`` where ``basis`` is a ``complex128`` matrix of
        shape ``[2**n, 2**n]`` whose columns are eigenvectors of the qubit
        cyclic shift ``S``, grouped by eigenvalue ``exp(2 pi i k / n)`` for
        ``k = 0, ..., n - 1``, and ``sizes`` lists the size of each group.

    Formula:
        For an orbit ``{S^m r}`` of size ``p`` and a ``k`` with ``k p = 0 mod n``,
        the column ``p**-0.5 * sum_m exp(-2 pi i k m / n) |S^m r>`` has
        eigenvalue ``exp(2 pi i k / n)``. Operators commuting with ``S`` do not
        couple different eigenvalues, so ``basis^dagger A basis`` is block
        diagonal for every such ``A``.
    """

    dimension = 2**index_qubits
    shift = cyclic_shift_matrix(index_qubits)
    seen: set[int] = set()
    orbits: list[list[int]] = []
    for start in range(dimension):
        if start in seen:
            continue
        orbit = [start]
        while True:
            following = int(torch.argmax(shift[:, orbit[-1]]).item())
            if following == start:
                break
            orbit.append(following)
        seen.update(orbit)
        orbits.append(orbit)

    columns: list[torch.Tensor] = []
    sizes: list[int] = []
    for k in range(index_qubits):
        count = 0
        for orbit in orbits:
            period = len(orbit)
            if (k * period) % index_qubits != 0:
                continue
            column = torch.zeros(dimension, dtype=torch.complex128)
            for m, state in enumerate(orbit):
                column[state] = complex(math.cos(-2 * math.pi * k * m / index_qubits), math.sin(-2 * math.pi * k * m / index_qubits))
            columns.append(column / math.sqrt(period))
            count += 1
        sizes.append(count)
    return torch.stack(columns, dim=1), tuple(sizes)


@lru_cache(maxsize=None)
@torch.inference_mode(False)
def _symmetry_block_pauli_terms(index_qubits: int, feature_qubits: int) -> tuple[torch.Tensor, ...]:
    """Expand every translation-invariant Pauli generator in the symmetry blocks.

    Returns one ``complex128`` tensor per non-empty eigenvalue block with shape
    ``[4**(2 + n_f), d_k F, d_k F]``: block ``k`` of
    ``(W (x) I)^dagger sum_j tau_j(P_alpha) (W (x) I)`` for every Pauli string
    ``P_alpha`` of the local term, where ``W`` is ``cyclic_shift_eigenbasis``.
    The expansion is computed once per register geometry in chunks of Pauli
    strings to bound memory.
    """

    feature_dim = 2**feature_qubits
    paulis = _materialized_pauli_basis_matrices(
        LOCAL_INDEX_QUBITS + feature_qubits,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )
    basis, sizes = cyclic_shift_eigenbasis(index_qubits)
    register_basis = torch.kron(basis, torch.eye(feature_dim, dtype=torch.complex128))
    bounds = []
    start = 0
    for size in sizes:
        if size:
            bounds.append((start, start + size * feature_dim))
        start += size * feature_dim
    chunks: list[list[torch.Tensor]] = [[] for _ in bounds]
    for chunk in paulis.split(64):
        generators = translation_invariant_generator(chunk, index_qubits=index_qubits, feature_qubits=feature_qubits)
        rotated = register_basis.conj().T @ generators @ register_basis
        for block_chunks, (lower, upper) in zip(chunks, bounds, strict=True):
            block_chunks.append(rotated[:, lower:upper, lower:upper].clone())
    return tuple(torch.cat(block_chunks, dim=0) for block_chunks in chunks)


@torch.inference_mode(False)
def _materialized_block_terms(
    index_qubits: int,
    feature_qubits: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    cache_key = ("block_terms", index_qubits * 100 + feature_qubits, str(device), dtype)
    cached = _DEVICE_TENSOR_CACHE.get(cache_key)
    if cached is None:
        cached = tuple(
            block.reshape(block.shape[0], -1).to(device=device, dtype=dtype)
            for block in _symmetry_block_pauli_terms(index_qubits, feature_qubits)
        )
        _DEVICE_TENSOR_CACHE[cache_key] = cached
    return cached


def symmetry_block_generators(
    parameters: torch.Tensor,
    *,
    index_qubits: int,
    feature_qubits: int,
) -> list[torch.Tensor]:
    """Return the blocks of ``(W (x) I)^dagger sum_j tau_j(h) (W (x) I)``.

    Contract:
        ``parameters`` has shape ``[..., 4**(2 + n_f)]``. The result holds one
        Hermitian block of shape ``[..., d_k F, d_k F]`` per non-empty
        eigenvalue group of ``S``, in the complex dtype matching
        ``parameters`` and on the same device.
    """

    complex_dtype = torch.complex128 if parameters.dtype == torch.float64 else torch.complex64
    block_terms = _materialized_block_terms(
        index_qubits,
        feature_qubits,
        device=parameters.device,
        dtype=complex_dtype,
    )
    flat_parameters = parameters.reshape(-1, parameters.shape[-1]).to(dtype=complex_dtype)
    generators = []
    for terms in block_terms:
        size = math.isqrt(terms.shape[-1])
        generators.append((flat_parameters @ terms).reshape(*parameters.shape[:-1], size, size))
    return generators


def block_exponentials(generators: list[torch.Tensor]) -> list[torch.Tensor]:
    """Exponentiate Hermitian generator blocks spectrally on their own device.

    Contract:
        Every entry has shape ``[..., d, d]``; the results have the same shapes,
        devices, and dtypes and stay differentiable.

    Notes:
        Each block is symmetrized and passed to ``hermitian_exponential``, so the
        result is unitary to the working precision: about ``1e-6`` for the
        ``complex64`` blocks of a ``float32`` model and ``1e-15`` for a
        ``float64`` model. The symmetry blocks are at most ``2**n F / n``-ish in
        size, which keeps batched device eigensolvers cheap.
    """

    return [hermitian_exponential(0.5 * (block + block.conj().transpose(-1, -2))) for block in generators]


@torch.inference_mode(False)
def _materialized_eigenbasis(index_qubits: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cache_key = ("eigenbasis", index_qubits, str(device), dtype)
    cached = _DEVICE_TENSOR_CACHE.get(cache_key)
    if cached is None:
        cached = cyclic_shift_eigenbasis(index_qubits)[0].to(device=device, dtype=dtype)
        _DEVICE_TENSOR_CACHE[cache_key] = cached
    return cached


def _block_diagonal(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Assemble ``[..., D, D]`` block-diagonal matrices from diagonal blocks ``[..., d_k, d_k]``."""

    dimension = sum(block.shape[-1] for block in blocks)
    rows = []
    offset = 0
    for block in blocks:
        size = block.shape[-1]
        rows.append(
            torch.cat(
                (
                    block.new_zeros(*block.shape[:-1], offset),
                    block,
                    block.new_zeros(*block.shape[:-1], dimension - offset - size),
                ),
                dim=-1,
            )
        )
        offset += size
    return torch.cat(rows, dim=-2)


def _apply_alternating_register_unitaries(
    state: torch.Tensor,
    unitaries: torch.Tensor,
    *,
    x_selected_dim: int,
    y_selected_dim: int,
) -> torch.Tensor:
    """Apply ``K`` pairs of branch-selected unitaries on ``(x, f)`` and then ``(y, f)``.

    Contract:
        ``state`` has canonical shape ``[B, X, CX, Y, CY, F]`` and ``unitaries``
        has shape ``[x_selected_dim, y_selected_dim, K, 2, D, D]`` with
        ``D = X F = Y F``; index ``0`` on the fourth axis acts on
        ``(x, f)`` and index ``1`` on ``(y, f)``. Branch ``(u, v)`` is read from
        the most-significant condition bits, as in ``ModeMultiplexer2D``.

    Formula:
        For every branch and sublayer ``s``,
        ``psi <- U[u, v, s, 1] (on y, f) U[u, v, s, 0] (on x, f) psi``.

    Notes:
        The state is kept in a branch-major layout whose last two axes are the
        register being transformed and the feature register, so every factor
        is one batched matrix product; switching between the x and y factors
        costs one transpose.
    """

    batch_size, x_dim, cx_dim, y_dim, cy_dim, feature_dim = state.shape
    branches = x_selected_dim * y_selected_dim
    sublayers = unitaries.shape[2]
    transposed = unitaries.reshape(branches, sublayers, 2, *unitaries.shape[-2:]).transpose(-1, -2)
    # [b, x, u, r, y, v, w, f] -> [u, v, b, r, w, y, x, f]: x-factor acts on the trailing (x, f) pair.
    working = state.reshape(
        batch_size,
        x_dim,
        x_selected_dim,
        cx_dim // x_selected_dim,
        y_dim,
        y_selected_dim,
        cy_dim // y_selected_dim,
        feature_dim,
    ).permute(2, 5, 0, 3, 6, 4, 1, 7)
    leading = working.shape[:6]
    for sublayer in range(sublayers):
        working = (working.reshape(branches, -1, x_dim * feature_dim) @ transposed[:, sublayer, 0]).reshape(
            *leading,
            x_dim,
            feature_dim,
        )
        working = working.transpose(5, 6)
        working = (working.reshape(branches, -1, y_dim * feature_dim) @ transposed[:, sublayer, 1]).reshape(
            *leading[:5],
            x_dim,
            y_dim,
            feature_dim,
        )
        working = working.transpose(5, 6)
    # [u, v, b, r, w, y, x, f] -> [b, x, u, r, y, v, w, f]
    return working.permute(2, 6, 0, 3, 5, 1, 4, 7).reshape(batch_size, x_dim, cx_dim, y_dim, cy_dim, feature_dim)


def _rotate_active_registers(state: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Apply ``basis`` to the active x and y index registers of a canonical state."""

    state = torch.einsum("xg,bgcydf->bxcydf", basis, state)
    return torch.einsum("yh,bxchdf->bxcydf", basis, state)


class QCSIndexLayer2D(nn.Module):
    """One QCS-QCNN layer of translation-invariant local evolutions on the index registers.

    Args:
        layout: Register layout of the state entering the layer.
        x_condition_qubits_to_use: Number of most-significant x-condition bits
            that select a branch, ``0`` for the first layer and ``1`` later.
        y_condition_qubits_to_use: Same for the y-condition register.
        sublayers: Number ``K`` of ``(x, y)`` sublayer pairs.
        generator_init_scale: Upper endpoint of the uniform initialization
            interval; every Pauli coefficient is sampled from
            ``Uniform(0, generator_init_scale)``.
        dtype: Real dtype of the trainable coefficients.

    Contract:
        ``local_parameters`` has shape
        ``[2**k_x, 2**k_y, K, 2, 4**(2 + n_f)]``: one local term per branch,
        sublayer, and axis (``0`` for x, ``1`` for y). ``forward`` maps a
        canonical state ``[B, X, CX, Y, CY, F]`` matching ``layout`` to the same
        shape.

    Formula:
        For each branch, sublayer ``s`` applies
        ``exp(i sum_j tau_j(h_{s,x}))`` on ``index_x (x) feature`` and then
        ``exp(i sum_j tau_j(h_{s,y}))`` on ``index_y (x) feature``. Each factor
        commutes with the qubit cyclic shift of its index register.
    """

    def __init__(
        self,
        layout: RegisterLayout2D,
        *,
        x_condition_qubits_to_use: int = 0,
        y_condition_qubits_to_use: int = 0,
        sublayers: int = 1,
        generator_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if sublayers < 1:
            raise ValueError(f"sublayers must be positive, got {sublayers}.")
        if generator_init_scale <= 0:
            raise ValueError(f"generator_init_scale must be positive, got {generator_init_scale}.")
        if layout.x_active_dim != layout.y_active_dim:
            raise ValueError("QCSIndexLayer2D expects equal active x and y register sizes.")
        index_qubits = layout.x_active_dim.bit_length() - 1
        if index_qubits < LOCAL_INDEX_QUBITS:
            raise ValueError(
                "QCS layers need at least two active index qubits per axis, "
                f"got {index_qubits}."
            )

        self.layout = layout
        self.index_qubits = index_qubits
        self.feature_qubits = layout.feature_qubits
        self.sublayers = sublayers
        self.generator_init_scale = float(generator_init_scale)
        self.x_selected_condition_dim = 2**x_condition_qubits_to_use
        self.y_selected_condition_dim = 2**y_condition_qubits_to_use
        self.local_parameters = nn.Parameter(
            torch.empty(
                (
                    self.x_selected_condition_dim,
                    self.y_selected_condition_dim,
                    sublayers,
                    2,
                    local_term_parameter_count(layout.feature_qubits),
                ),
                dtype=dtype,
            ).uniform_(0.0, self.generator_init_scale)
        )

    def block_generators(self) -> list[torch.Tensor]:
        """Return the Hermitian ``S``-eigenbasis generator blocks ``[2**k_x, 2**k_y, K, 2, d, d]``."""

        return symmetry_block_generators(
            self.local_parameters,
            index_qubits=self.index_qubits,
            feature_qubits=self.feature_qubits,
        )

    def forward(self, state: torch.Tensor, unitary_blocks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Apply the layer; ``unitary_blocks`` optionally passes precomputed exponentiated blocks."""

        self.layout.validate_state_shape(state)
        if not torch.is_complex(state):
            raise ValueError(f"QCSIndexLayer2D expects a complex state tensor, got dtype {state.dtype}.")
        if unitary_blocks is None:
            unitary_blocks = block_exponentials(self.block_generators())
        unitary_blocks = [block.to(device=state.device, dtype=state.dtype) for block in unitary_blocks]
        basis = _materialized_eigenbasis(self.index_qubits, device=state.device, dtype=state.dtype)

        state = _rotate_active_registers(state, basis.conj().T)
        state = _apply_alternating_register_unitaries(
            state,
            _block_diagonal(unitary_blocks),
            x_selected_dim=self.x_selected_condition_dim,
            y_selected_dim=self.y_selected_condition_dim,
        )
        return _rotate_active_registers(state, basis)


class QCSQCNN(nn.Module):
    """A multilayer QCS-QCNN on the FRQI address encoding with a linear head.

    Args:
        image_size: Square image side length. Must be a power of two.
        num_classes: Number of logits produced by the final linear head.
        feature_qubits: Number of feature-register qubits ``n_f``.
        quantum_layers: Number of QCS layers ``Q``. Every layer needs at least
            two active index qubits per axis, so ``Q <= log2(image_size) - 1``.
        sublayers: Number ``K`` of ``(x, y)`` sublayer pairs per layer. ``K``
            scales the quantum parameter count without changing the readout.
        brightness_range: Encoder-side angle interval ``(a, b)`` of the FRQI
            map, as in ``PCSQCNN``.
        shot_budget: Optional finite-shot readout budget used in evaluation
            mode, as in ``PCSQCNN``.
        generator_init_scale: Upper endpoint of the uniform Pauli-coefficient
            initialization interval of every local term.
        dtype: Real dtype used by the encoder and the trainable coefficients.

    Formula:
        ``encoder -> [QCS layer -> pooling]^(Q-1) -> QCS layer -> marginal
        measurement -> finite-shot histogram -> linear classifier``, with the
        same encoder, pooling, conditioning, readout, and head as ``PCSQCNN``.
        The first layer is unconditional; later layers select one local term
        per most recently pooled ``(x, y)`` bit pair.

    Returns:
        ``forward(images)`` returns logits of shape ``[B, num_classes]``.

    Notes:
        Layer ``l`` holds ``b_l * K * 2 * 4**(2 + n_f)`` quantum parameters with
        ``b_1 = 1`` and ``b_l = 4`` for ``l >= 2``. For ``Q = 3`` and
        ``n_f = 2`` this is ``4608 K``, so ``K = 8`` matches the 36,864 quantum
        parameters of the PCS-QCNN exactly.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        feature_qubits: int = 1,
        quantum_layers: int = 1,
        *,
        sublayers: int = 1,
        brightness_range: tuple[float, float] = (0.0, math.pi),
        shot_budget: int | None = None,
        generator_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {feature_qubits}.")

        self.layer_layouts = _build_layer_layouts(
            image_size=image_size,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
        )
        final_layout = self.layer_layouts[-1]
        if final_layout.x_active_dim < 2**LOCAL_INDEX_QUBITS:
            raise ValueError(
                "QCSQCNN needs at least two active index qubits per axis in every layer: "
                f"quantum_layers={quantum_layers} is too deep for image_size={image_size}."
            )
        self.image_size = image_size
        self.num_classes = num_classes
        self.feature_qubits = feature_qubits
        self.quantum_layers = quantum_layers
        self.sublayers = sublayers
        self.generator_init_scale = float(generator_init_scale)
        self.encoder = FrqiEncoder2D(feature_qubits=feature_qubits, brightness_range=brightness_range, dtype=dtype)
        self.brightness_range = self.encoder.brightness_range
        self.measurement = MarginalMeasurement2D()
        self.readout_histogram = FiniteShotHistogram2D(shot_budget=shot_budget)
        self.shot_budget = self.readout_histogram.shot_budget
        self.layers = nn.ModuleList(
            [
                QCSIndexLayer2D(
                    layout,
                    x_condition_qubits_to_use=0 if layer_idx == 0 else 1,
                    y_condition_qubits_to_use=0 if layer_idx == 0 else 1,
                    sublayers=sublayers,
                    generator_init_scale=self.generator_init_scale,
                    dtype=dtype,
                )
                for layer_idx, layout in enumerate(self.layer_layouts)
            ]
        )
        self.readout_spatial_shape = (final_layout.x_active_dim, final_layout.y_active_dim, final_layout.feature_dim)
        self.classifier = nn.Linear(math.prod(self.readout_spatial_shape), num_classes, dtype=dtype)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self.encoder.real_dtype = self.classifier.weight.dtype
        self.encoder.complex_dtype = (
            torch.complex128 if self.classifier.weight.dtype == torch.float64 else torch.complex64
        )
        return result

    def layer_unitary_blocks(self) -> list[list[torch.Tensor]]:
        """Return every layer's exponentiated ``S``-eigenbasis blocks."""

        return [block_exponentials(layer.block_generators()) for layer in self.layers]

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Return exact measured readout probabilities ``[B, X, Y, F]`` before finite-shot sampling."""

        _validate_image_spatial_shape(images, image_size=self.image_size, model_name="QCSQCNN")
        state = self.encoder(images)
        current_layout = self.layer_layouts[0]
        for layer_idx, (layer, unitary_blocks) in enumerate(zip(self.layers, self.layer_unitary_blocks(), strict=True)):
            state = layer(state, unitary_blocks=unitary_blocks)
            if layer_idx < self.quantum_layers - 1:
                state, current_layout = move_active_qubit_to_condition(
                    state,
                    current_layout,
                    x_qubits_to_condition=1,
                    y_qubits_to_condition=1,
                )
        return self.measurement(state)

    def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
        """Map measured readout histograms into logits through the linear head."""

        flattened = _flatten_readout_histogram(
            readout_histogram,
            expected_spatial_shape=self.readout_spatial_shape,
            model_name="QCSQCNN",
        )
        return self.classifier(flattened)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the full QCS-QCNN pipeline on images ``[B, image_size, image_size]``."""

        measured = self.exact_quantum_readout_probabilities(images)
        measured = self.readout_histogram(measured)
        return self.classify_readout_histogram(measured)
