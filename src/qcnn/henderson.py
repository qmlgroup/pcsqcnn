"""Henderson-QCNN baseline: a trainable multilayer quanvolution network.

The module implements a Henderson-like reference model for ``PCSQCNN``. It
follows the pixel-to-qubit "quanvolution" construction of Henderson et al.
(2019) in the form popularized by the PennyLane demo ``tutorial_quanvolution``,
extended to a trainable multilayer QCNN with measurement-conditioned pooling.
Its weight sharing is a qubit-cyclic-shift symmetry of the pixel qubits; the
address-encoding (FRQI) QCS-QCNN of the article's Fig. 1(a,c) lives in
``qcnn.qcs``. The layer stack is:

- every pixel ``x_uv in [0, 1]`` is angle-encoded into its own qubit as
  ``RY(theta_uv)|0>`` with ``theta_uv = a + (b - a) * x_uv``; the default
  ``(a, b) = (0, pi)`` reproduces the demo encoding ``RY(pi * x)``;
- every layer applies one shared trainable 4-qubit unitary
  ``U = exp(i * sum_alpha theta_alpha P_alpha)`` to each non-overlapping
  ``2 x 2`` block of the current qubit lattice (kernel 2, stride 2), using the
  same full Pauli-basis parameterization as the PCS-QCNN multiplexer blocks;
- after every non-final layer, three qubits of each block are measured in the
  computational basis and a measurement-conditioned single-qubit unitary
  ``V_m`` acts on the remaining top-left qubit, which becomes one site of the
  next, twice coarser lattice (MERA/Cong-style pooling);
- after the final layer, all four qubits of every block are measured and the
  per-block outcome distributions are passed to one linear-softmax head.

Several independent filter trees (``channels``) can be stacked side by side;
their readouts are concatenated before the shared head.

Symmetry:
    Weight sharing over the block tiling makes each layer commute with cyclic
    shifts of the qubit lattice by the stride, i.e. with the even subgroup of
    qubit cyclic shifts (QCS). Under pixel-to-qubit encoding, qubit
    permutations coincide with pixel translations (case (i) of the article's
    Lemma 1), so the stack is equivariant to cyclic image shifts by
    ``2**quantum_layers`` pixels at the readout.

Simulation:
    The encoded state is a product state and every layer acts on disjoint
    blocks, so the global state stays a product over blocks at all depths. The
    module stores every surviving single-qubit state by its Pauli vector
    ``r = (Tr rho, <X>, <Y>, <Z>)``. A block input with product state has
    Pauli coefficients ``c_beta = prod_w r_w[beta_w]`` for the 256 Pauli
    strings ``beta`` on four qubits, and both layer maps are linear in ``c``.
    Their real transfer tensors are built from the trainable unitaries in the
    Heisenberg picture, which gives an exact infinite-shot evaluation of the
    measured circuit without materializing density matrices per block.

The model always returns logits. Softmax and the training loss remain outside
the module.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qcnn.layout import is_power_of_two
from qcnn.quantum import (
    _feature_qubits_from_pauli_parameter_count,
    _materialized_pauli_basis_matrices,
    _validate_brightness_range,
    hermitian_exponential,
)

BLOCK_QUBITS = 4
"""Number of qubits in one ``2 x 2`` block."""

BLOCK_DIM = 2**BLOCK_QUBITS
"""Hilbert-space dimension of one ``2 x 2`` block."""

BLOCK_PAULI_COUNT = 4**BLOCK_QUBITS
"""Number of Pauli strings (and real block parameters) on one block."""

POOLING_OUTCOME_COUNT = 2 ** (BLOCK_QUBITS - 1)
"""Number of measurement outcomes of the three pooled qubits of one block."""

INFERENCE_CHUNK_SIZE = 256
"""Largest image batch contracted at once when autograd is disabled."""

_PAULI_TRANSFORM_DEVICE_DTYPE_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_COMPILED_CONTRACTIONS = None


def _angle_encode_pauli_vectors(
    images: torch.Tensor,
    *,
    brightness_range: tuple[float, float],
) -> torch.Tensor:
    """Map grayscale pixels to Pauli vectors of ``RY(theta)|0>`` product states.

    Contract:
        ``images`` must have shape ``[B, X, Y]`` with normalized grayscale
        values. The result has shape ``[B, X, Y, 4]`` and the same real dtype.

    Formula:
        With ``theta = a + (b - a) * x`` the encoded qubit is
        ``RY(theta)|0> = cos(theta / 2)|0> + sin(theta / 2)|1>`` and its Pauli
        vector is ``(1, sin(theta), 0, cos(theta))`` in the order
        ``(I, X, Y, Z)``.
    """

    start, end = brightness_range
    angles = start + (end - start) * images
    ones = torch.ones_like(angles)
    zeros = torch.zeros_like(angles)
    return torch.stack((ones, torch.sin(angles), zeros, torch.cos(angles)), dim=-1)


def _split_block_pair_products(pauli_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group a site lattice into ``2 x 2`` blocks and pair their Pauli vectors.

    Contract:
        ``pauli_vectors`` has channel-major shape ``[C, B, X, Y, 4]`` with even
        ``X`` and ``Y``. Block ``(i, j)`` holds the sites ``(2i, 2j)``,
        ``(2i, 2j + 1)``, ``(2i + 1, 2j)``, and ``(2i + 1, 2j + 1)`` as wires
        ``0, 1, 2, 3``, the patch order of the PennyLane quanvolution demo.
        Wire ``0`` is the most-significant qubit of the block register.

    Formula:
        Returns ``r01 = r_0 (x) r_1`` and ``r23 = r_2 (x) r_3``, each with shape
        ``[C, B * (X / 2) * (Y / 2), 16]``, so the Pauli coefficients of the
        block input are ``c[16 * k + l] = r01[k] * r23[l]`` in the MSB-left
        Pauli-string order of ``_pauli_basis_matrices(4)``.
    """

    channels, batch_size, x_dim, y_dim, _ = pauli_vectors.shape
    blocks = pauli_vectors.reshape(channels, batch_size, x_dim // 2, 2, y_dim // 2, 2, 4)
    wire0 = blocks[:, :, :, 0, :, 0]
    wire1 = blocks[:, :, :, 0, :, 1]
    wire2 = blocks[:, :, :, 1, :, 0]
    wire3 = blocks[:, :, :, 1, :, 1]
    rows = batch_size * (x_dim // 2) * (y_dim // 2)
    r01 = (wire0[..., :, None] * wire1[..., None, :]).reshape(channels, rows, 16)
    r23 = (wire2[..., :, None] * wire3[..., None, :]).reshape(channels, rows, 16)
    return r01, r23


def _contract_shared_blocks(pauli_vectors: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    """Contract a channel-shared lattice ``[1, B, X, Y, 4]`` against ``[C, O, 256]``.

    The 256 block coefficients are materialized once and contracted against
    all channels in one matrix product. Returns ``[C, B, X / 2, Y / 2, O]``.
    """

    _, batch_size, x_dim, y_dim, _ = pauli_vectors.shape
    channels, output_dim, _ = transfer.shape
    r01, r23 = _split_block_pair_products(pauli_vectors)
    rows = r01.shape[1]
    coefficients = (r01[0, :, :, None] * r23[0, :, None, :]).reshape(rows, BLOCK_PAULI_COUNT)
    output = coefficients @ transfer.reshape(channels * output_dim, BLOCK_PAULI_COUNT).T
    output = output.reshape(rows, channels, output_dim).permute(1, 0, 2)
    return output.reshape(channels, batch_size, x_dim // 2, y_dim // 2, output_dim)


def _contract_channel_blocks(pauli_vectors: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    """Contract a channel-resolved lattice ``[C, B, X, Y, 4]`` against ``[C, O, 256]``.

    The pair ``r23`` is contracted by a batched matrix product and ``r01`` by
    a broadcast reduction, which keeps the per-block intermediate at
    ``O * 16`` entries. Returns ``[C, B, X / 2, Y / 2, O]``.
    """

    channels, batch_size, x_dim, y_dim, _ = pauli_vectors.shape
    output_dim = transfer.shape[1]
    r01, r23 = _split_block_pair_products(pauli_vectors)
    rows = r01.shape[1]
    # transfer[c, o, 16 * k + l] -> [c, l, (o, k)] so that r23 contracts first.
    right = transfer.reshape(channels, output_dim, 16, 16).permute(0, 3, 1, 2).reshape(
        channels,
        16,
        output_dim * 16,
    )
    partial = torch.bmm(r23, right).reshape(channels, rows, output_dim, 16)
    output = (partial * r01[:, :, None, :]).sum(dim=-1)
    return output.reshape(channels, batch_size, x_dim // 2, y_dim // 2, output_dim)


def _apply_block_transfer(pauli_vectors: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    """Apply one linear block map to every ``2 x 2`` block of a site lattice.

    Contract:
        ``pauli_vectors`` has channel-major shape ``[C_in, B, X, Y, 4]`` with
        ``C_in`` either ``1`` (input shared by all channels) or ``C``.
        ``transfer`` has shape ``[C, O, 256]``. The result has shape
        ``[C, B, X / 2, Y / 2, O]``.

    Formula:
        ``out[c, b, i, j, o] = sum_beta transfer[c, o, beta] * c_beta`` where
        ``c_beta = prod_w r_w[beta_w]`` are the Pauli coefficients of block
        ``(i, j)``.

    Notes:
        Shared and channel-resolved inputs use separate contraction helpers,
        which ``use_compiled_block_transfer`` compiles independently. Inputs
        are made contiguous so compiled graphs do not specialize on strides.
    """

    input_channels = pauli_vectors.shape[0]
    channels = transfer.shape[0]
    if input_channels not in {1, channels}:
        raise ValueError(
            "Block transfer expects 1 or C input channels, "
            f"got {input_channels} for C={channels}."
        )
    shared_contraction, channel_contraction = _COMPILED_CONTRACTIONS or (
        _contract_shared_blocks,
        _contract_channel_blocks,
    )
    contraction = shared_contraction if input_channels == 1 else channel_contraction
    return contraction(pauli_vectors.contiguous(), transfer.contiguous())


def _single_qubit_pauli_matrices(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _materialized_pauli_basis_matrices(1, device=device, dtype=dtype)


def _complex_dtype_for_real(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _exact_unitaries(parameters: torch.Tensor) -> torch.Tensor:
    """Exponentiate Pauli coefficients in double precision on the host.

    Contract:
        ``parameters`` has shape ``[..., 4**f]`` on any device. The result is a
        ``torch.complex128`` CPU tensor of shape ``[..., 2**f, 2**f]`` that
        stays differentiable with respect to ``parameters``.

    Formula:
        ``U = exp(i * sum_alpha theta_alpha P_alpha)``, the same map as
        ``pauli_coefficients_to_unitary``, evaluated spectrally because every
        generator is Hermitian.

    Notes:
        Scaling-and-squaring in single precision loses about ``1e-4`` of
        unitarity on 16 x 16 blocks with the article initialization, and
        double-precision kernels are slow on consumer GPUs. The handful of
        batch-independent matrices per step is therefore exponentiated on the
        host, which also keeps the step free of device-side syncs inside
        ``torch.linalg.matrix_exp``.
    """

    host_parameters = parameters.to(device="cpu", dtype=torch.float64)
    num_qubits = _feature_qubits_from_pauli_parameter_count(int(host_parameters.shape[-1]))
    basis = _materialized_pauli_basis_matrices(num_qubits, device=host_parameters.device, dtype=torch.complex128)
    generator = torch.einsum("...p,pij->...ij", host_parameters.to(dtype=torch.complex128), basis)
    return hermitian_exponential(generator)


def _materialized_pauli_transform_matrix(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the cached ``[256, 256]`` matrix ``M[(r, s), beta] = P_beta[s, r]``."""

    cache_key = (str(device), dtype)
    cached = _PAULI_TRANSFORM_DEVICE_DTYPE_CACHE.get(cache_key)
    if cached is None:
        basis = _materialized_pauli_basis_matrices(BLOCK_QUBITS, device=device, dtype=dtype)
        cached = basis.permute(2, 1, 0).reshape(BLOCK_DIM * BLOCK_DIM, BLOCK_PAULI_COUNT).contiguous()
        _PAULI_TRANSFORM_DEVICE_DTYPE_CACHE[cache_key] = cached
    return cached


def _pauli_string_coefficients(matrices: torch.Tensor) -> torch.Tensor:
    """Return ``Tr(P_beta A)`` for all 256 four-qubit Pauli strings.

    Contract:
        ``matrices`` has shape ``[..., 16, 16]``. The result has shape
        ``[..., 256]`` and the same complex dtype, ordered like
        ``_pauli_basis_matrices(4)``.

    Formula:
        ``Tr(P_beta A) = sum_{r, s} P_beta[s, r] A[r, s]``, evaluated as one
        matrix product of the flattened matrices with the cached transform
        matrix ``M[(r, s), beta] = P_beta[s, r]``.
    """

    leading_shape = matrices.shape[:-2]
    transform = _materialized_pauli_transform_matrix(device=matrices.device, dtype=matrices.dtype)
    coefficients = matrices.reshape(-1, BLOCK_DIM * BLOCK_DIM) @ transform
    return coefficients.reshape(*leading_shape, BLOCK_PAULI_COUNT)


def use_compiled_block_transfer(enabled: bool = True) -> None:
    """Route every block contraction through ``torch.compile``.

    The switch is a process-wide performance option for long CUDA training
    runs: compilation fuses the broadcast products and reductions of the
    block contractions and roughly halves the data-path time. It is not
    part of any saved model specification, results agree with the eager path
    up to floating-point rounding, and passing ``enabled=False`` restores the
    eager path.
    """

    global _COMPILED_CONTRACTIONS
    if enabled:
        # Train/eval batch sizes, ragged final batches, and layer geometries each
        # add guarded graph variants; keep them all compiled instead of falling
        # back to eager after dynamo's default limit of 8.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    _COMPILED_CONTRACTIONS = (
        (
            torch.compile(_contract_shared_blocks, dynamic=True),
            torch.compile(_contract_channel_blocks, dynamic=True),
        )
        if enabled
        else None
    )


def pooling_transfer_tensor(block_unitaries: torch.Tensor, pooling_unitaries: torch.Tensor) -> torch.Tensor:
    """Build the Heisenberg-picture transfer tensor of one conv-and-pool block.

    Contract:
        ``block_unitaries`` has shape ``[C, 16, 16]`` and ``pooling_unitaries``
        has shape ``[C, 8, 2, 2]``. Both are complex. The result is a real
        tensor of shape ``[C, 4, 256]``.

    Formula:
        The block applies ``U``, measures wires ``1, 2, 3`` with outcome
        ``m = 4 m_1 + 2 m_2 + m_3``, applies ``V_m`` on wire ``0``, and discards
        ``m``. For an input ``rho = (1 / 16) sum_beta c_beta P_beta`` the
        surviving qubit has Pauli vector ``r_out[o] = sum_beta T[o, beta] c_beta``
        with
        ``T[o, beta] = Tr(U^dagger O_o U P_beta) / 16`` and
        ``O_o = sum_m (V_m^dagger sigma_o V_m) (x) |m><m|``.

    Notes:
        ``T[0, beta] = delta_{beta, 0}``, so the surviving state keeps the
        trace of the block input.
    """

    channels = block_unitaries.shape[0]
    device = block_unitaries.device
    complex_dtype = block_unitaries.dtype
    sigma = _single_qubit_pauli_matrices(device=device, dtype=complex_dtype)
    identity = torch.eye(POOLING_OUTCOME_COUNT, device=device, dtype=complex_dtype)

    corrected_paulis = torch.einsum("cmji,ojk,cmkl->cmoil", pooling_unitaries.conj(), sigma, pooling_unitaries)
    observables = torch.einsum("cmoab,mn->coambn", corrected_paulis, identity).reshape(
        channels,
        4,
        BLOCK_DIM,
        BLOCK_DIM,
    )
    evolved = block_unitaries.conj().transpose(-1, -2)[:, None] @ observables @ block_unitaries[:, None]
    return _pauli_string_coefficients(evolved).real / BLOCK_DIM


def readout_transfer_tensor(block_unitaries: torch.Tensor) -> torch.Tensor:
    """Build the transfer tensor of one conv block followed by full readout.

    Contract:
        ``block_unitaries`` has shape ``[C, 16, 16]`` and is complex. The
        result is a real tensor of shape ``[C, 16, 256]``.

    Formula:
        For an input ``rho = (1 / 16) sum_beta c_beta P_beta`` the outcome
        probabilities of measuring all four block qubits after ``U`` are
        ``p(z) = sum_beta F[z, beta] c_beta`` with
        ``F[z, beta] = (U P_beta U^dagger)[z, z] / 16``. Outcome ``z`` is
        indexed MSB-left over wires ``0, 1, 2, 3``.

    Notes:
        ``F[z, beta] = Tr(P_beta A_z) / 16`` with the rank-one operator
        ``A_z[j, i] = conj(U[z, j]) * U[z, i]``.
    """

    row_operators = block_unitaries.conj()[..., :, None] * block_unitaries[..., None, :]
    return _pauli_string_coefficients(row_operators).real / BLOCK_DIM


class HendersonBlockLayer2D(nn.Module):
    """One Henderson-QCNN layer: shared 2x2-block unitaries with optional pooling.

    Args:
        channels: Number of independent filter trees. Each channel owns its
            own block unitary (and pooling unitaries) shared over all blocks.
        pool: Whether the layer ends with measurement-conditioned pooling to
            one qubit per block. Non-pooling layers end with full readout of
            all four block qubits.
        block_init_scale: Upper endpoint of the uniform initialization
            interval. Every Pauli coefficient is sampled from
            ``Uniform(0, block_init_scale)``.
        dtype: Real dtype of the trainable Pauli coefficients.

    Contract:
        ``forward(pauli_vectors)`` expects the channel-major lattice
        ``[C_in, B, X, Y, 4]`` with ``C_in in {1, channels}`` and even ``X``
        and ``Y``. Pooling layers return the Pauli vectors of the surviving
        qubits with shape ``[channels, B, X / 2, Y / 2, 4]``; readout layers
        return per-block outcome distributions with shape
        ``[channels, B, X / 2, Y / 2, 16]``.

    Formula:
        ``block_parameters`` has shape ``[channels, 256]`` and parameterizes
        ``U = exp(i * sum_alpha theta_alpha P_alpha)`` on the four block
        qubits. Pooling layers additionally hold ``pooling_parameters`` with
        shape ``[channels, 8, 4]`` that parameterize one single-qubit
        ``V_m = exp(i * sum_alpha phi_alpha sigma_alpha)`` per outcome ``m`` of
        the three measured qubits.

    Notes:
        Unitaries are exponentiated once per forward pass in double precision,
        independent of the batch. The transfer tensor is then assembled on the
        parameter device in the complex dtype matching the parameters.
        ``HendersonQCNN`` batches this construction over all layers and passes the
        result to ``forward`` through ``transfer``.
    """

    def __init__(
        self,
        channels: int,
        *,
        pool: bool,
        block_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if block_init_scale <= 0:
            raise ValueError(f"block_init_scale must be positive, got {block_init_scale}.")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")

        self.channels = channels
        self.pool = bool(pool)
        self.block_init_scale = float(block_init_scale)
        self.block_parameters = nn.Parameter(
            torch.empty((channels, BLOCK_PAULI_COUNT), dtype=dtype).uniform_(0.0, self.block_init_scale)
        )
        if self.pool:
            self.pooling_parameters = nn.Parameter(
                torch.empty((channels, POOLING_OUTCOME_COUNT, 4), dtype=dtype).uniform_(
                    0.0,
                    self.block_init_scale,
                )
            )
        else:
            self.register_parameter("pooling_parameters", None)

    @property
    def output_dim(self) -> int:
        """Per-block output size: a Pauli vector (4) or an outcome distribution (16)."""

        return 4 if self.pool else BLOCK_DIM

    def block_unitaries(self) -> torch.Tensor:
        """Return the shared block unitaries as a CPU ``complex128`` tensor ``[channels, 16, 16]``."""

        return _exact_unitaries(self.block_parameters)

    def pooling_unitaries(self) -> torch.Tensor:
        """Return the conditioned unitaries as a CPU ``complex128`` tensor ``[channels, 8, 2, 2]``."""

        if self.pooling_parameters is None:
            raise RuntimeError("Readout layers do not have pooling unitaries.")
        return _exact_unitaries(self.pooling_parameters)

    def transfer_tensor_from_unitaries(
        self,
        block_unitaries: torch.Tensor,
        pooling_unitaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the real transfer tensor ``[channels, output_dim, 256]`` from unitaries.

        The unitaries are moved to the parameter device in the complex dtype
        that matches the parameter dtype before the transfer tensor is built.
        """

        device = self.block_parameters.device
        real_dtype = self.block_parameters.dtype
        complex_dtype = _complex_dtype_for_real(real_dtype)
        block_unitaries = block_unitaries.to(device=device, dtype=complex_dtype)
        if not self.pool:
            return readout_transfer_tensor(block_unitaries).to(dtype=real_dtype)
        if pooling_unitaries is None:
            raise ValueError("Pooling layers require pooling_unitaries.")
        pooling_unitaries = pooling_unitaries.to(device=device, dtype=complex_dtype)
        return pooling_transfer_tensor(block_unitaries, pooling_unitaries).to(dtype=real_dtype)

    def transfer_tensor(self) -> torch.Tensor:
        """Return the real block transfer tensor with shape ``[channels, output_dim, 256]``."""

        return self.transfer_tensor_from_unitaries(
            self.block_unitaries(),
            self.pooling_unitaries() if self.pool else None,
        )

    def forward(self, pauli_vectors: torch.Tensor, transfer: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the layer to every ``2 x 2`` block of the incoming site lattice.

        Args:
            pauli_vectors: Channel-major lattice ``[C_in, B, X, Y, 4]``.
            transfer: Optional precomputed ``transfer_tensor()`` result. When
                omitted, it is built from the current parameters.
        """

        if pauli_vectors.ndim != 5 or pauli_vectors.shape[-1] != 4:
            raise ValueError(
                "HendersonBlockLayer2D expects Pauli vectors with shape [C, B, X, Y, 4], "
                f"got {tuple(pauli_vectors.shape)}."
            )
        if pauli_vectors.shape[2] % 2 != 0 or pauli_vectors.shape[3] % 2 != 0:
            raise ValueError(
                "HendersonBlockLayer2D expects an even site lattice, "
                f"got {tuple(pauli_vectors.shape[2:4])}."
            )
        if transfer is None:
            transfer = self.transfer_tensor()
        return _apply_block_transfer(pauli_vectors, transfer.to(dtype=pauli_vectors.dtype))


class HendersonQCNN(nn.Module):
    """A multilayer Henderson-QCNN (trainable multilayer quanvolution) with logits output.

    Args:
        image_size: Square image side length. Must be a power of two that is
            at least ``2**quantum_layers`` because every layer halves both
            lattice axes.
        num_classes: Number of logits produced by the final linear head.
        channels: Number of independent filter trees whose readouts are
            concatenated before the head.
        quantum_layers: Number of conv layers ``Q``. Layers ``1 .. Q - 1`` end
            with pooling; layer ``Q`` ends with full readout.
        brightness_range: Encoder-side angle interval ``(a, b)``. Pixels are
            encoded as ``RY(a + (b - a) * x)|0>``; the default ``(0, pi)`` is
            the PennyLane quanvolution demo encoding ``RY(pi * x)``.
        block_init_scale: Upper endpoint of the uniform Pauli-coefficient
            initialization interval of every block and pooling unitary.
        dtype: Real dtype of the encoded Pauli vectors and the trainable
            coefficients.

    Formula:
        For ``Q = quantum_layers``, each channel computes
        ``encoder -> [U_l on 2x2 blocks -> measure 3 qubits -> V_m]^(Q-1)
        -> U_Q on 2x2 blocks -> measure all``. The readout entry for channel
        ``c``, final block ``(i, j)``, and outcome ``z`` is
        ``m[c, i, j, z] = p_{c, i, j}(z) / (channels * n_blocks)``, so the
        flattened readout is a probability vector of length
        ``channels * n_blocks * 16`` like the PCS-QCNN full readout.

    Returns:
        ``forward(images)`` returns logits of shape ``[B, num_classes]``.

    Notes:
        Each channel has ``256 * Q + 32 * (Q - 1)`` trainable quantum
        parameters. The head has
        ``channels * n_blocks * 16 * num_classes + num_classes`` parameters
        with ``n_blocks = (image_size / 2**Q)**2``. Inference is exact
        (infinite-shot); the PCS-QCNN finite-shot sampling layer has no QCS
        counterpart here.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        channels: int = 1,
        quantum_layers: int = 1,
        *,
        brightness_range: tuple[float, float] = (0.0, math.pi),
        block_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if quantum_layers < 1:
            raise ValueError(f"quantum_layers must be positive, got {quantum_layers}.")
        if not is_power_of_two(image_size):
            raise ValueError(f"image_size must be a positive power of two, got {image_size}.")
        if image_size < 2**quantum_layers:
            raise ValueError(
                "image_size must be at least 2**quantum_layers: "
                f"got image_size={image_size} for quantum_layers={quantum_layers}."
            )
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.channels = channels
        self.quantum_layers = quantum_layers
        self.brightness_range = _validate_brightness_range(brightness_range)
        self.block_init_scale = float(block_init_scale)
        self.real_dtype = dtype
        self.layers = nn.ModuleList(
            [
                HendersonBlockLayer2D(
                    channels,
                    pool=layer_idx < quantum_layers - 1,
                    block_init_scale=self.block_init_scale,
                    dtype=dtype,
                )
                for layer_idx in range(quantum_layers)
            ]
        )
        final_size = image_size // 2**quantum_layers
        self.readout_shape = (channels, final_size, final_size, BLOCK_DIM)
        self.classifier = nn.Linear(math.prod(self.readout_shape), num_classes, dtype=dtype)

    def transfer_tensors(self) -> list[torch.Tensor]:
        """Return the transfer tensors of all layers, built in one host round trip.

        All quantum parameters are copied to the host once, exponentiated in
        two batched double-precision calls (block and pooling unitaries), and
        sent back to assemble the per-layer transfer tensors on the parameter
        device.
        """

        block_parameters = [layer.block_parameters for layer in self.layers]
        pooling_parameters = [layer.pooling_parameters for layer in self.layers if layer.pool]
        flat_parameters = torch.cat([parameter.reshape(-1) for parameter in (*block_parameters, *pooling_parameters)])
        host_parameters = flat_parameters.to(device="cpu", dtype=torch.float64)
        block_size = self.channels * BLOCK_PAULI_COUNT
        block_count = len(block_parameters) * block_size
        block_unitaries = _exact_unitaries(
            host_parameters[:block_count].reshape(len(block_parameters), self.channels, BLOCK_PAULI_COUNT)
        )
        pooling_unitaries = None
        if pooling_parameters:
            pooling_unitaries = _exact_unitaries(
                host_parameters[block_count:].reshape(
                    len(pooling_parameters),
                    self.channels,
                    POOLING_OUTCOME_COUNT,
                    4,
                )
            )

        transfers: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            transfers.append(
                layer.transfer_tensor_from_unitaries(
                    block_unitaries[layer_idx],
                    pooling_unitaries[layer_idx] if layer.pool else None,
                )
            )
        return transfers

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into channel-shared Pauli vectors ``[1, B, X, Y, 4]``."""

        if images.ndim != 3:
            raise ValueError(
                f"HendersonQCNN expects image tensors with shape [B, X, Y], got {tuple(images.shape)}."
            )
        if tuple(images.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError(
                "HendersonQCNN expects image tensors with spatial shape "
                f"({self.image_size}, {self.image_size}), got {tuple(images.shape[1:])}."
            )
        pauli_vectors = _angle_encode_pauli_vectors(
            images.to(dtype=self.real_dtype),
            brightness_range=self.brightness_range,
        )
        return pauli_vectors[None]

    def block_readout_distributions(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-block outcome distributions ``[B, C, X_out, Y_out, 16]``.

        Every final block distribution sums to one. This is the unnormalized
        counterpart of ``exact_quantum_readout_probabilities``.

        Notes:
            The transfer tensors are built once per call. When autograd is
            disabled, images are contracted in chunks of
            ``INFERENCE_CHUNK_SIZE``; the model has no batch-coupled
            operations, so chunking only bounds memory.
        """

        transfers = self.transfer_tensors()
        if torch.is_grad_enabled() or images.shape[0] <= INFERENCE_CHUNK_SIZE:
            return self._contract_layers(images, transfers)
        return torch.cat(
            [self._contract_layers(chunk, transfers) for chunk in images.split(INFERENCE_CHUNK_SIZE)],
            dim=0,
        )

    def _contract_layers(self, images: torch.Tensor, transfers: list[torch.Tensor]) -> torch.Tensor:
        state = self.encode(images)
        for layer, transfer in zip(self.layers, transfers, strict=True):
            state = layer(state, transfer=transfer)
        return state.transpose(0, 1)

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Return the normalized exact readout ``[B, C, X_out, Y_out, 16]``."""

        distributions = self.block_readout_distributions(images)
        block_count = self.readout_shape[0] * self.readout_shape[1] * self.readout_shape[2]
        return distributions / block_count

    def classify_readout_histogram(self, readout_histogram: torch.Tensor) -> torch.Tensor:
        """Map readout probabilities ``[..., C, X_out, Y_out, 16]`` to logits."""

        if tuple(readout_histogram.shape[-4:]) != self.readout_shape:
            raise ValueError(
                f"HendersonQCNN expects readout tensors ending with {self.readout_shape}, "
                f"got trailing shape {tuple(readout_histogram.shape[-4:])}."
            )
        return self.classifier(readout_histogram.reshape(*readout_histogram.shape[:-4], -1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the full Henderson-QCNN pipeline on images ``[B, image_size, image_size]``."""

        return self.classify_readout_histogram(self.exact_quantum_readout_probabilities(images))
