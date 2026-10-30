"""Hur-QCNN baseline: the amplitude-encoded QCNN of Hur, Kim, and Park.

The module implements the quantum convolutional neural network of T. Hur,
L. Kim, and D. K. Park, "Quantum convolutional neural network for classical
data classification", Quantum Mach. Intell. 4, 3 (2022), arXiv:2108.00661,
with the circuit wiring of the authors' code (github.com/takh04/QCNN,
``QCNN_circuit.py`` and ``unitary.py``). This is a literature comparator for ``PCSQCNN``, not an exactly
QCS-equivariant control: every convolution shares a two-qubit gate over a ring,
but the ordered overlapping gates do not commute with a one-site shift.


- amplitude encoding: the ``image_size**2`` pixels, flattened row-major, are
  normalized and loaded as the amplitudes of ``n = 2 log2(image_size)``
  qubits, wire 0 being the most significant bit (PennyLane convention); for
  ``32 x 32`` images wires 0-4 hold the row bits and wires 5-9 the column
  bits. ``encoding="frqi"`` (our extension) loads the image with the FRQI
  encoding of ``PCSQCNN`` instead: pixel ``x_uv`` becomes
  ``(sin(p_uv)|0> + cos(p_uv)|1>) / sqrt(XY)`` with ``p_uv = a + (b - a) x_uv``
  on one extra colour qubit, the last and least significant wire, which joins
  the convolution ring like an address qubit; for ``32 x 32`` images the
  network is ``11 -> conv -> pool -> 6 -> conv -> measure 6``;
- convolution: the same two-qubit gate acts on all nearest-neighbour pairs of
  the active qubits ``a_0, ..., a_{m-1}`` with a periodic boundary, ordered as
  in the authors' code: the closing pair ``(a_0, a_{m-1})`` first, then the
  pairs ``(a_{2k}, a_{2k+1})``, then ``(a_{2k+1}, a_{2k+2})``. The gate is
  the paper's convolutional circuit 9 (``U_SU4``, an arbitrary SU(4) with 15
  parameters), and every layer applies ``filters`` independent convolutions
  (the paper's ``l_i``);
- pooling: the two-parameter circuit of the paper's Fig. 3 (``Pooling_ansatz1``:
  ``CRZ(theta_1)``, ``X`` on the control, ``CRX(theta_2)``) acts on every pair
  ``(a_{2k+1} -> a_{2k})``, after which the control ``a_{2k+1}`` is traced
  out; an unpaired last qubit stays active;
- multi-class output: convolution-pooling layers repeat while the pooled
  register keeps at least ``ceil(log2 num_classes)`` qubits (the multi-class
  construction suggested in the paper's Sec. V); a final convolution acts on
  the remaining register, all of whose qubits are measured, and the outcome
  probabilities feed one linear-softmax head (the paper's hybrid ending in
  Sec. II and the readout convention of the benchmark).

``channels`` independent copies of the circuit (our extension, as the
channels of ``HendersonQCNN``) read the same encoded state; their outcome
distributions, divided by ``channels``, are concatenated before the head. This
scales the parameter count without deepening the circuit.

For ``8`` qubits and two classes the construction reduces to the paper's
Fig. 1 circuit with 51 parameters (ansatz 9a), except that the authors' code
puts the control of the last pooling gate on wire 0 and measures wire 4; with
an arbitrary SU(4) convolution in front of it, both orientations span the same
model family. For ``32 x 32`` images and ten classes the network is
``10 -> conv -> pool -> 5 -> conv -> measure 5``.

Symmetry:
    The weight sharing is invariant under cyclic shifts of the qubit ring,
    i.e. the QCS prior of the article's Fig. 1(a). Under amplitude encoding
    the qubits are address bits, so qubit shifts are not pixel translations
    (case (ii) of the article's Lemma 1). The gates of one convolution do not
    commute, so the composed layer is not exactly shift-equivariant either.

Simulation:
    Traced-out qubits are never acted on again, so the network is simulated
    as one pure ``n``-qubit state vector per channel and the readout
    marginalizes over them (deferred measurement); this equals the
    reduced-density-matrix evaluation of the measured circuit exactly. The
    disjoint gates of a layer are applied together as two dense operators on
    the two sides of a wire split, which is several times faster on GPUs than
    one two-qubit contraction per gate.

The model always returns logits. Softmax and the training loss remain outside
the module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from qcnn.hybrid import _validate_image_spatial_shape
from qcnn.layout import is_power_of_two
from qcnn.quantum import _validate_brightness_range

CONVOLUTION_GATE_PARAMETERS = 15
POOLING_GATE_PARAMETERS = 2
# State amplitudes simulated at once; larger channel counts are split into chunks.
MAX_CHUNK_AMPLITUDES = 2**22


@dataclass(frozen=True)
class HurLayerSchedule:
    """Gate wiring of one Hur-QCNN circuit.

    ``convolutions[l]`` lists the ``(first, second)`` wires of every gate of
    convolution ``l``; ``poolings[l]`` lists the ``(control, target)`` wires of
    pooling ``l``. Convolution ``l`` precedes pooling ``l``; when there is one
    more convolution than poolings, the last one is the output convolution.
    """

    qubits: int
    convolutions: tuple[tuple[tuple[int, int], ...], ...]
    poolings: tuple[tuple[tuple[int, int], ...], ...]
    measured_wires: tuple[int, ...]


def _ring_convolution_pairs(active: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if len(active) < 2:
        return ()
    if len(active) == 2:
        return ((active[0], active[1]),)
    closing = ((active[0], active[-1]),)
    even = tuple((active[index], active[index + 1]) for index in range(0, len(active) - 1, 2))
    odd = tuple((active[index], active[index + 1]) for index in range(1, len(active) - 1, 2))
    return closing + even + odd


def build_hur_layer_schedule(qubits: int, *, output_qubits: int) -> HurLayerSchedule:
    """Return the convolution and pooling wiring for ``qubits`` input qubits.

    Convolution-pooling layers repeat while the pooled register keeps at least
    ``output_qubits`` qubits; a final convolution acts on the remaining
    register when it has at least two qubits.
    """

    if qubits < 2:
        raise ValueError(f"qubits must be at least 2, got {qubits}.")
    if not 1 <= output_qubits <= qubits:
        raise ValueError(f"output_qubits must lie in [1, {qubits}], got {output_qubits}.")
    active = tuple(range(qubits))
    convolutions: list[tuple[tuple[int, int], ...]] = []
    poolings: list[tuple[tuple[int, int], ...]] = []
    while len(active) >= 2 and math.ceil(len(active) / 2) >= output_qubits:
        convolutions.append(_ring_convolution_pairs(active))
        poolings.append(tuple((active[index + 1], active[index]) for index in range(0, len(active) - 1, 2)))
        active = active[::2]
    if len(active) >= 2:
        convolutions.append(_ring_convolution_pairs(active))
    return HurLayerSchedule(
        qubits=qubits,
        convolutions=tuple(convolutions),
        poolings=tuple(poolings),
        measured_wires=active,
    )


def encode_image_state(
    images: torch.Tensor,
    *,
    encoding: str,
    brightness_range: tuple[float, float],
    real_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode square images ``[B, H, W]`` into normalized qubit states ``[B, 2**n]``.

    ``"amplitude"`` loads the L2-normalized pixels, flattened row-major, as the
    amplitudes of ``n = 2 log2 H`` qubits (an all-zero image maps to the zero
    state); ``"frqi"`` writes ``(sin(p_uv)|0> + cos(p_uv)|1>) / sqrt(HW)`` with
    ``p_uv = a + (b - a) x_uv`` and the colour qubit as the last of
    ``n = 2 log2 H + 1`` wires. Wire 0 is the most significant bit.
    """

    state_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    images = images.to(real_dtype)
    if encoding == "frqi":
        start, end = brightness_range
        angles = start + (end - start) * images
        # Pixel bits are the most significant, the colour qubit is the last wire.
        amplitudes = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).reshape(images.shape[0], -1)
        return (amplitudes / math.sqrt(images.shape[1] * images.shape[2])).to(state_dtype)
    amplitudes = images.reshape(images.shape[0], -1)
    amplitudes = amplitudes / amplitudes.norm(dim=1, keepdim=True).clamp_min(torch.finfo(real_dtype).tiny)
    return amplitudes.to(state_dtype)


def _u3(theta: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(0.5 * theta).to(torch.complex64 if theta.dtype == torch.float32 else torch.complex128)
    sin = torch.sin(0.5 * theta).to(cos.dtype)
    phase_phi = torch.polar(torch.ones_like(phi), phi).to(cos.dtype)
    phase_lam = torch.polar(torch.ones_like(lam), lam).to(cos.dtype)
    return torch.stack(
        (
            torch.stack((cos, -phase_lam * sin), dim=-1),
            torch.stack((phase_phi * sin, phase_phi * phase_lam * cos), dim=-1),
        ),
        dim=-2,
    )


def _ry(theta: torch.Tensor) -> torch.Tensor:
    return _u3(theta, torch.zeros_like(theta), torch.zeros_like(theta))


def _rz(theta: torch.Tensor) -> torch.Tensor:
    phase = torch.polar(torch.ones_like(theta), 0.5 * theta)
    zero = torch.zeros_like(phase)
    return torch.stack((torch.stack((phase.conj(), zero), dim=-1), torch.stack((zero, phase), dim=-1)), dim=-2)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(0.5 * theta).to(torch.complex64 if theta.dtype == torch.float32 else torch.complex128)
    minus_i_sin = -1j * torch.sin(0.5 * theta).to(cos.dtype)
    return torch.stack((torch.stack((cos, minus_i_sin), dim=-1), torch.stack((minus_i_sin, cos), dim=-1)), dim=-2)


def _kron(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ab,...cd->...acbd", first, second).reshape(*first.shape[:-2], 4, 4)


def _cnot(*, control_first: bool, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    permutation = (0, 1, 3, 2) if control_first else (0, 3, 2, 1)
    return torch.eye(4, dtype=dtype, device=device)[list(permutation)]


def su4_gate(parameters: torch.Tensor) -> torch.Tensor:
    """Return the 15-parameter ``U_SU4`` gate ``[..., 4, 4]`` of the authors' code.

    The first gate factor is ``wires[0]``: ``U3 x U3``, ``CNOT(0 -> 1)``,
    ``RY(p6) x RZ(p7)``, ``CNOT(1 -> 0)``, ``RY(p8) x 1``, ``CNOT(0 -> 1)``,
    ``U3 x U3``.
    """

    p = parameters.unbind(-1)
    first = _kron(_u3(p[0], p[1], p[2]), _u3(p[3], p[4], p[5]))
    dtype, device = first.dtype, first.device
    cnot_01 = _cnot(control_first=True, dtype=dtype, device=device)
    cnot_10 = _cnot(control_first=False, dtype=dtype, device=device)
    identity = torch.eye(2, dtype=dtype, device=device).expand_as(_ry(p[8]))
    middle = _kron(_ry(p[6]), _rz(p[7]))
    last_rotation = _kron(_ry(p[8]), identity)
    final = _kron(_u3(p[9], p[10], p[11]), _u3(p[12], p[13], p[14]))
    return final @ cnot_01 @ last_rotation @ cnot_10 @ middle @ cnot_01 @ first


def _controlled(unitary: torch.Tensor) -> torch.Tensor:
    """Return ``|0><0| (x) 1 + |1><1| (x) U`` ``[..., 4, 4]`` for ``U [..., 2, 2]``."""

    identity = torch.eye(2, dtype=unitary.dtype, device=unitary.device).expand_as(unitary)
    zero = torch.zeros_like(unitary)
    return torch.cat((torch.cat((identity, zero), dim=-1), torch.cat((zero, unitary), dim=-1)), dim=-2)


def pooling_gate(parameters: torch.Tensor) -> torch.Tensor:
    """Return the two-parameter pooling gate ``[..., 4, 4]`` on ``(control, target)``.

    ``CRZ(p0)``, ``X`` on the control, ``CRX(p1)``: a control in ``|1>``
    rotates the target by ``RZ(p0)``, a control in ``|0>`` by ``RX(p1)``.
    """

    rz = _rz(parameters[..., 0])
    rx = _rx(parameters[..., 1])
    identity = torch.eye(2, dtype=rz.dtype, device=rz.device)
    flip_control = _kron(torch.tensor([[0, 1], [1, 0]], dtype=rz.dtype, device=rz.device), identity)
    return _controlled(rx) @ flip_control @ _controlled(rz)


def apply_two_qubit_gate(state: torch.Tensor, gate: torch.Tensor, wires: tuple[int, int]) -> torch.Tensor:
    """Apply per-channel gates ``[C, 4, 4]`` to ``wires`` of ``state [C, B, 2, ..., 2]``.

    ``wires[0]`` is the first gate factor.
    """

    axes = (2 + wires[0], 2 + wires[1])
    moved = torch.movedim(state, axes, (-2, -1))
    updated = torch.einsum("cb...ij,cklij->cb...kl", moved, gate.reshape(-1, 2, 2, 2, 2))
    return torch.movedim(updated, (-2, -1), axes)


@dataclass(frozen=True)
class GatePhase:
    """Mutually disjoint gates of one layer, applied around a wire split.

    The state ``[C, B, 2**n]`` is viewed as ``[C, B, 2**split, 2**(n - split)]``:
    the ``left`` gates (wires ``< split``) and the ``right`` gates (wires
    ``>= split``) act as one dense operator per side, and ``straddling`` gates
    are applied one by one.
    """

    split: int
    left: tuple[tuple[int, int], ...]
    right: tuple[tuple[int, int], ...]
    straddling: tuple[tuple[int, int], ...]


def build_gate_phases(pairs: tuple[tuple[int, int], ...], *, qubits: int) -> tuple[GatePhase, ...]:
    """Group ordered gates into consecutive phases of disjoint gates and split each phase.

    Disjoint gates commute, so a phase may apply its gates in any order. Each
    split minimizes the number of straddling gates, then the dense-operator
    cost ``2**split + 2**(n - split)`` of the sides that hold gates.
    """

    groups: list[list[tuple[int, int]]] = []
    for pair in pairs:
        if not groups or any(set(pair) & set(other) for other in groups[-1]):
            groups.append([])
        groups[-1].append(pair)

    phases: list[GatePhase] = []
    for group in groups:

        def cost(split: int, group: list[tuple[int, int]] = group) -> tuple[int, int]:
            straddling = sum(min(pair) < split <= max(pair) for pair in group)
            left = any(max(pair) < split for pair in group)
            right = any(min(pair) >= split for pair in group)
            return straddling, left * 2**split + right * 2 ** (qubits - split)

        split = min(range(1, qubits), key=cost)
        phases.append(
            GatePhase(
                split=split,
                left=tuple(pair for pair in group if max(pair) < split),
                right=tuple(pair for pair in group if min(pair) >= split),
                straddling=tuple(pair for pair in group if min(pair) < split <= max(pair)),
            )
        )
    return tuple(phases)


def _side_operator(gate: torch.Tensor, pairs: tuple[tuple[int, int], ...], *, offset: int, bits: int) -> torch.Tensor:
    """Return the dense operator ``[C, 2**bits, 2**bits]`` of ``pairs`` on wires ``offset, ..., offset + bits - 1``."""

    channels = gate.shape[0]
    identity = torch.eye(2**bits, dtype=gate.dtype, device=gate.device)
    # Basis columns ride in the batch slot, so the result holds U e_j in row j.
    columns = identity.expand(channels, -1, -1).reshape(channels, 2**bits, *([2] * bits))
    for first, second in pairs:
        columns = apply_two_qubit_gate(columns, gate, (first - offset, second - offset))
    return columns.reshape(channels, 2**bits, 2**bits).transpose(1, 2)


def phase_operators(
    gate: torch.Tensor, phase: GatePhase, *, qubits: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the dense ``(left, right)`` operators ``[..., 2**k, 2**k]`` of one phase, ``None`` for an empty side.

    ``gate`` may carry any leading dimensions, for example ``[C, F, 4, 4]``
    for all filters of a layer at once.
    """

    leading = gate.shape[:-2]
    flat = gate.reshape(-1, 4, 4)
    left = right = None
    if phase.left:
        left = _side_operator(flat, phase.left, offset=0, bits=phase.split)
        left = left.reshape(*leading, *left.shape[1:])
    if phase.right:
        right = _side_operator(flat, phase.right, offset=phase.split, bits=qubits - phase.split)
        right = right.reshape(*leading, *right.shape[1:])
    return left, right


def apply_gate_phase(
    state: torch.Tensor,
    gate: torch.Tensor,
    phase: GatePhase,
    *,
    qubits: int,
    operators: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Apply one phase of per-channel gates ``[C, 4, 4]`` to ``state [C, B, 2**qubits]``.

    ``operators`` optionally passes the phase's precomputed ``phase_operators``.
    """

    channels, batch = state.shape[:2]
    rows, columns = 2**phase.split, 2 ** (qubits - phase.split)
    left, right = phase_operators(gate, phase, qubits=qubits) if operators is None else operators
    if right is not None:
        state = torch.matmul(state.reshape(channels, batch * rows, columns), right.transpose(1, 2))
    if left is not None:
        state = torch.einsum("cbrk,csr->cbsk", state.reshape(channels, batch, rows, columns), left)
    if phase.straddling:
        state = state.reshape(channels, batch, *([2] * qubits))
        for wires in phase.straddling:
            state = apply_two_qubit_gate(state, gate, wires)
    return state.reshape(channels, batch, 2**qubits)


class HurConvolutionLayer(nn.Module):
    """``filters`` shared-gate ring convolutions per channel, one 15-parameter SU(4) gate each."""

    _qcnn_quantum_parameter_module = True

    def __init__(
        self,
        pairs: tuple[tuple[int, int], ...],
        *,
        qubits: int,
        channels: int,
        filters: int,
        init_scale: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.qubits = qubits
        self.phases = build_gate_phases(pairs, qubits=qubits)
        self.angles = nn.Parameter(
            init_scale * torch.rand(channels, filters, CONVOLUTION_GATE_PARAMETERS, dtype=dtype)
        )

    def forward(self, state: torch.Tensor, channels: slice = slice(None)) -> torch.Tensor:
        """Apply the layer of the ``channels`` slice to ``state [C_slice, B, 2**qubits]``."""

        gates = su4_gate(self.angles[channels]).to(state.dtype)  # [C, F, 4, 4]
        filters = gates.shape[1]

        def per_filter(operator: torch.Tensor | None) -> list[torch.Tensor | None]:
            return [None] * filters if operator is None else list(operator.unbind(1))

        # Build the dense operators of all filters at once; only their application is sequential.
        operators = [
            list(zip(*(per_filter(side) for side in phase_operators(gates, phase, qubits=self.qubits)), strict=True))
            for phase in self.phases
        ]
        for index, gate in enumerate(gates.unbind(1)):
            for phase, phase_filters in zip(self.phases, operators, strict=True):
                state = apply_gate_phase(state, gate, phase, qubits=self.qubits, operators=phase_filters[index])
        return state


class HurPoolingLayer(nn.Module):
    """One shared two-parameter pooling gate per channel on every ``(control, target)`` pair."""

    _qcnn_quantum_parameter_module = True

    def __init__(
        self,
        pairs: tuple[tuple[int, int], ...],
        *,
        qubits: int,
        channels: int,
        init_scale: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.qubits = qubits
        self.phases = build_gate_phases(pairs, qubits=qubits)
        self.angles = nn.Parameter(init_scale * torch.rand(channels, POOLING_GATE_PARAMETERS, dtype=dtype))

    def forward(self, state: torch.Tensor, channels: slice = slice(None)) -> torch.Tensor:
        """Apply the layer of the ``channels`` slice to ``state [C_slice, B, 2**qubits]``."""

        gate = pooling_gate(self.angles[channels]).to(state.dtype)
        for phase in self.phases:
            state = apply_gate_phase(state, gate, phase, qubits=self.qubits)
        return state


class HurQCNN(nn.Module):
    """The amplitude-encoded QCNN of Hur, Kim, and Park with a linear head.

    Args:
        image_size: Square image side length. Must be a power of two.
        num_classes: Number of logits produced by the final linear head.
        filters: Number of independent convolutions per layer (the paper's
            ``l_i``, equal in every layer).
        channels: Number of independent circuit copies whose readouts are
            concatenated before the head.
        output_qubits: Minimum number of measured qubits; defaults to
            ``ceil(log2 num_classes)``.
        encoding: ``"amplitude"`` for the paper's amplitude encoding or
            ``"frqi"`` for the FRQI encoding of ``PCSQCNN`` on one extra
            colour qubit (the last wire).
        brightness_range: FRQI angle interval ``(a, b)``; ignored by the
            amplitude encoding.
        angle_init_scale: Upper endpoint of the uniform initialization
            interval of every gate angle.
        dtype: Real dtype of the angles and the head; the state uses the
            matching complex dtype.

    Returns:
        ``forward(images)`` returns logits of shape ``[B, num_classes]``.

    Notes:
        A convolution holds ``15 * filters`` parameters per channel and a
        pooling 2; for ``32 x 32`` images and ten classes the quantum core
        holds ``channels * (30 * filters + 2)`` parameters and the head reads
        ``32 * channels`` probabilities, so it holds ``320 * channels + 10``.
        ``channels=8`` matches the 256-dimensional readout and 2,570-parameter
        head of the PCS-QCNN; ``channels=112`` its 39,434 parameters in total.
        With ``encoding="frqi"`` the 11-qubit circuit holds
        ``channels * (30 * filters + 2)`` parameters as well and the head
        reads ``64 * channels`` probabilities.

        Channels are simulated in chunks of at most ``MAX_CHUNK_AMPLITUDES``
        state amplitudes; with gradients enabled and more than one chunk,
        every chunk is recomputed in the backward pass (activation
        checkpointing), which changes memory, not results.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        filters: int = 1,
        *,
        channels: int = 1,
        output_qubits: int | None = None,
        encoding: str = "amplitude",
        brightness_range: tuple[float, float] = (0.0, math.pi),
        angle_init_scale: float = 2.0 * math.pi,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if not is_power_of_two(image_size):
            raise ValueError(f"image_size must be a power of two, got {image_size}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if filters < 1:
            raise ValueError(f"filters must be positive, got {filters}.")
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        if encoding not in ("amplitude", "frqi"):
            raise ValueError(f"encoding must be 'amplitude' or 'frqi', got {encoding!r}.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.filters = filters
        self.channels = channels
        self.encoding = encoding
        self.brightness_range = _validate_brightness_range(brightness_range)
        self.qubits = 2 * int(math.log2(image_size)) + (1 if encoding == "frqi" else 0)
        self.output_qubits = max(1, math.ceil(math.log2(num_classes))) if output_qubits is None else output_qubits
        self.angle_init_scale = float(angle_init_scale)
        self.state_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.schedule = build_hur_layer_schedule(self.qubits, output_qubits=self.output_qubits)

        layers: list[nn.Module] = []
        for index, pairs in enumerate(self.schedule.convolutions):
            layers.append(
                HurConvolutionLayer(
                    pairs,
                    qubits=self.qubits,
                    channels=channels,
                    filters=filters,
                    init_scale=self.angle_init_scale,
                    dtype=dtype,
                )
            )
            if index < len(self.schedule.poolings):
                layers.append(
                    HurPoolingLayer(
                        self.schedule.poolings[index],
                        qubits=self.qubits,
                        channels=channels,
                        init_scale=self.angle_init_scale,
                        dtype=dtype,
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.readout_dim = 2 ** len(self.schedule.measured_wires)
        self.classifier = nn.Linear(channels * self.readout_dim, num_classes, dtype=dtype)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images ``[B, H, W]`` into normalized states ``[B, 2**n]``."""

        _validate_image_spatial_shape(images, image_size=self.image_size, model_name="HurQCNN")
        return encode_image_state(
            images,
            encoding=self.encoding,
            brightness_range=self.brightness_range,
            real_dtype=self.classifier.weight.dtype,
        )

    def _channel_readout(self, state: torch.Tensor, channels: slice) -> torch.Tensor:
        """Map channel-major states ``[C_slice, B, 2**n]`` to probabilities ``[B, C_slice, 2**m]``."""

        for layer in self.layers:
            state = layer(state, channels)
        probabilities = (state.real.square() + state.imag.square()).reshape(*state.shape[:2], *([2] * self.qubits))
        traced_axes = [2 + wire for wire in range(self.qubits) if wire not in self.schedule.measured_wires]
        if traced_axes:
            probabilities = probabilities.sum(dim=traced_axes)
        return probabilities.reshape(*state.shape[:2], self.readout_dim).transpose(0, 1)

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-channel outcome probabilities ``[B, channels, 2**m]`` (measured wires, MSB first).

        Every layer is applied phase by phase (``build_gate_phases``): the
        disjoint gates on each side of a wire split act as one dense operator.
        """

        encoded = self.encode(images)
        chunk = max(1, MAX_CHUNK_AMPLITUDES // encoded.numel())
        readouts = []
        for start in range(0, self.channels, chunk):
            channels = slice(start, min(start + chunk, self.channels))
            state = encoded.expand(channels.stop - start, -1, -1)
            if torch.is_grad_enabled() and chunk < self.channels:
                readouts.append(checkpoint(self._channel_readout, state, channels, use_reentrant=False))
            else:
                readouts.append(self._channel_readout(state, channels))
        return torch.cat(readouts, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the full Hur-QCNN pipeline on images ``[B, image_size, image_size]``."""

        readout = self.exact_quantum_readout_probabilities(images) / self.channels
        return self.classifier(readout.reshape(readout.shape[0], -1))
