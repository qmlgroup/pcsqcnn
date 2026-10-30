"""FRQI adaptation of equivariant split-parallelizing QCNNs.

Sources: Chinzei et al., Phys. Rev. Research 6, 023042 (2024), and
Chinzei et al., Quantum Mach. Intell. 8, 53 (2026).
This is an independent adaptation, not their published phase-classification
circuit. See supplementary.tex, sec:supp_sp_qcnn, for the complete specification.

Wire order (most significant first): x bits, y bits, auxiliary, brightness.
G = C_n x C_n rotates the n bits of each address independently and fixes
the two feature wires. Splitting is 2n+2 -> (n,n,2) -> single wires; no
partial trace, copying, or disentangling is performed. All three stages
commute with G. Orbit-summed probabilities make the classifier G-invariant.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qcnn.layout import is_power_of_two
from qcnn.quantum import FrqiEncoder2D


def binary_rotation_orbits(bits: int) -> tuple[tuple[int, ...], ...]:
    """Binary necklaces, ordered by their smallest integer representative."""
    if bits < 1:
        raise ValueError("bits must be positive.")
    groups: dict[int, list[int]] = {}
    mask = (1 << bits) - 1
    for value in range(1 << bits):
        rotations = [((value << k) | (value >> (bits - k))) & mask for k in range(bits)]
        groups.setdefault(min(rotations), []).append(value)
    return tuple(tuple(groups[key]) for key in sorted(groups))


def _rx(angle: torch.Tensor) -> torch.Tensor:
    """exp(-i angle X); angle is half the usual PennyLane RX argument."""
    c, s = torch.cos(angle), -1j * torch.sin(angle)
    return torch.stack((torch.stack((c, s)), torch.stack((s, c))))


def _euler(angles: torch.Tensor) -> torch.Tensor:
    z = torch.stack((torch.exp(-1j * angles[1]), torch.exp(1j * angles[1]))).diag()
    return _rx(angles[2]) @ z @ _rx(angles[0])


def _tensor_power(matrix: torch.Tensor, count: int) -> torch.Tensor:
    result = matrix
    for _ in range(count - 1):
        result = torch.kron(result, matrix)
    return result


def _local_rotations(state: torch.Tensor, angles: torch.Tensor, bits: int) -> torch.Tensor:
    """Fuse commuting rotations into operators on x, y, and feature registers."""
    size = 1 << bits
    x_gate = _tensor_power(_euler(angles[0]), bits)
    y_gate = _tensor_power(_euler(angles[1]), bits)
    f_gate = torch.kron(_euler(angles[2]), _euler(angles[3]))
    state = state.reshape(-1, size, size, 4)
    state = torch.einsum("ij,bjyk->biyk", x_gate, state)
    state = torch.einsum("ij,bxjk->bxik", y_gate, state)
    state = state @ f_gate.T
    return state.reshape(-1, size * size * 4)


def _edge_orbits(bits: int, *, joint: bool) -> tuple[tuple[tuple[int, int], ...], ...]:
    x, y = tuple(range(bits)), tuple(range(bits, 2 * bits))
    aux, color = 2 * bits, 2 * bits + 1

    def ring(wires: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted({tuple(sorted((wire, wires[(i + 1) % bits]))) for i, wire in enumerate(wires)}))

    edges = (ring(x), ring(y))
    if joint:
        edges += (
            tuple((i, j) for i in x for j in y),
            tuple((i, aux) for i in x), tuple((i, color) for i in x),
            tuple((i, aux) for i in y), tuple((i, color) for i in y),
        )
    return edges + (((aux, color),),)


class SplitParallelStage(nn.Module):
    """Shared RX-RZ-RX gates followed by commuting, orbit-shared ZZ gates."""

    _qcnn_quantum_parameter_module = True

    def __init__(self, bits: int, *, depth: int, stage: str, init_scale: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.bits = bits
        self.stage = stage
        self.edges = () if stage == "single" else _edge_orbits(bits, joint=stage == "joint")
        self.angles = nn.Parameter(init_scale * torch.rand(depth, 12 + len(self.edges), dtype=dtype))
        qubits = 2 * bits + 2
        indices = torch.arange(1 << qubits)
        signs = torch.stack([1 - 2 * ((indices >> (qubits - 1 - wire)) & 1) for wire in range(qubits)])
        zz = torch.stack([sum(signs[i] * signs[j] for i, j in orbit) for orbit in self.edges]) if self.edges else torch.empty(0, 1 << qubits)
        self.register_buffer("zz_eigenvalues", zz.to(dtype), persistent=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        for block in self.angles:
            state = _local_rotations(state, block[:12].reshape(4, 3), self.bits)
            if self.edges:
                # All ZZ generators commute, including those sharing a wire.
                state = state * torch.exp(-1j * (block[12:] @ self.zz_eigenvalues))
        return state


class SpQCNN(nn.Module):
    """Three-stage C_n x C_n-equivariant FRQI circuit and invariant linear head.

    Defaults implement the 32x32 benchmark: 12 qubits, depths (10,10,1),
    362 quantum parameters, orbit readout 8x8x4, 2570 head parameters.
    Smaller power-of-two images are supported for independent reference tests.
    Rotations use exp(-i angle P). Every angle is initialized independently
    from Uniform[0, angle_init_scale); the head uses torch.nn.Linear defaults.
    dtype is serializable as "float32" or "float64". Exact probabilities only.
    """

    def __init__(
        self, *, image_size: int = 32, num_classes: int = 10,
        depth: int = 10, brightness_range: tuple[float, float] = (0.0, math.pi),
        angle_init_scale: float = 2.0 * math.pi, dtype: str = "float32",
    ) -> None:
        super().__init__()
        if not is_power_of_two(image_size) or image_size < 4:
            raise ValueError("image_size must be a power of two of at least 4.")
        if not isinstance(depth, int) or depth < 1 or num_classes < 1:
            raise ValueError("depth and num_classes must be positive integers.")
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'.")
        if not math.isfinite(angle_init_scale) or angle_init_scale <= 0:
            raise ValueError("angle_init_scale must be finite and positive.")
        real_dtype = getattr(torch, dtype)
        self.image_size = image_size
        self.num_classes = num_classes
        self.address_qubits = image_size.bit_length() - 1
        self.feature_qubits = 2
        self.qubits = 2 * self.address_qubits + self.feature_qubits
        self.depth = depth
        self.encoder = FrqiEncoder2D(feature_qubits=2, brightness_range=brightness_range, dtype=real_dtype)
        self.brightness_range = self.encoder.brightness_range
        self.stages = nn.ModuleList([
            SplitParallelStage(self.address_qubits, depth=count, stage=kind, init_scale=angle_init_scale, dtype=real_dtype)
            for kind, count in (("joint", depth), ("split", depth), ("single", 1))
        ])
        self.address_orbits = binary_rotation_orbits(self.address_qubits)
        self.readout_spatial_shape = (len(self.address_orbits), len(self.address_orbits), 4)
        labels = torch.empty(image_size, dtype=torch.long)
        for label, orbit in enumerate(self.address_orbits):
            labels[list(orbit)] = label
        bins = (labels[:, None, None] * len(self.address_orbits) + labels[None, :, None]) * 4 + torch.arange(4)
        self.register_buffer("readout_bins", bins.flatten(), persistent=False)
        self.classifier = nn.Linear(math.prod(self.readout_spatial_shape), num_classes, dtype=real_dtype)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self.encoder.real_dtype = self.classifier.weight.dtype
        self.encoder.complex_dtype = torch.complex128 if self.classifier.weight.dtype == torch.float64 else torch.complex64
        return result

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3 or tuple(images.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError(f"SpQCNN expects [B, {self.image_size}, {self.image_size}] images.")
        # The shared encoder omits 1/sqrt(number of pixels). Normalize here.
        return self.encoder(images).reshape(images.shape[0], -1) / self.image_size

    def quantum_core(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the circuit to arbitrary normalized or unnormalized states."""
        if state.ndim != 2 or state.shape[1] != 1 << self.qubits or not state.is_complex():
            raise ValueError(f"Expected complex states [B, {1 << self.qubits}].")
        for stage in self.stages:
            state = stage(state)
        return state

    def readout_from_state(self, state: torch.Tensor) -> torch.Tensor:
        probabilities = state.real.square() + state.imag.square()
        histogram = probabilities.new_zeros((state.shape[0], math.prod(self.readout_spatial_shape)))
        histogram = histogram.scatter_add(1, self.readout_bins.expand(state.shape[0], -1), probabilities)
        return histogram.reshape(state.shape[0], *self.readout_spatial_shape)

    def exact_quantum_readout_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return self.readout_from_state(self.quantum_core(self.encode(images)))

    def classify_readout_histogram(self, histogram: torch.Tensor) -> torch.Tensor:
        if histogram.ndim != 4 or tuple(histogram.shape[1:]) != self.readout_spatial_shape:
            raise ValueError(f"Expected histograms [B, {self.readout_spatial_shape}].")
        return self.classifier(histogram.flatten(start_dim=1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_readout_histogram(self.exact_quantum_readout_probabilities(images))
