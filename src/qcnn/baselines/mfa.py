"""Ten-class Mordacci--Ferrari--Amoretti QCNN.

Independent implementation of the architecture described in
Mordacci, Ferrari and Amoretti, Multi-Class Quantum Convolutional Neural
Networks (QUASAR 2024, doi:10.1145/3660318.3660326). The author's public
reference is qslab-unipr/qcnn revision 5c320d93664aa02125f18eaab6b33119f08da39a.
No source files from that unlicensed repository are distributed here.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

from qcnn.statistics import LossCollector


def _single_qubit_and_controlled_pattern(
    angles: torch.Tensor, wires: tuple[int, ...]
) -> None:
    """The paper's CC14 convolution component, including its gate ordering."""

    width = len(wires)
    for position, wire in enumerate(wires):
        qml.RY(angles[position], wires=wire)
    for position, wire in enumerate(wires):
        qml.CRX(angles[width + position], wires=(wires[(position - 1) % width], wire))
    for position, wire in enumerate(wires):
        qml.RY(angles[2 * width + position], wires=wire)

    if width % 3 == 0 or width == 2:
        for position in reversed(range(width)):
            qml.CRX(
                angles[3 * width + position],
                wires=(wires[position], wires[(position - 1) % width]),
            )
    else:
        control = width - 1
        target = (control + 3) % width
        for position in reversed(range(width)):
            qml.CRX(
                angles[3 * width + position], wires=(wires[control], wires[target])
            )
            control = target
            target = (control + 3) % width


def _two_qubit_block(angles: torch.Tensor, first: int, second: int) -> None:
    """The shared fifteen-parameter SU(4) block."""

    qml.U3(*angles[0:3], wires=first)
    qml.U3(*angles[3:6], wires=second)
    qml.CNOT(wires=(first, second))
    qml.RY(angles[6], wires=first)
    qml.RZ(angles[7], wires=second)
    qml.CNOT(wires=(second, first))
    qml.RY(angles[8], wires=first)
    qml.CNOT(wires=(first, second))
    qml.U3(*angles[9:12], wires=first)
    qml.U3(*angles[12:15], wires=second)


def _convolution(
    angles: torch.Tensor, wires: tuple[int, ...], *, first_stage: bool
) -> None:
    width = len(wires)
    pattern_angles = angles[: 16 if first_stage else 4 * width]
    pair_angles = angles[16 if first_stage else 4 * width :]
    if first_stage:
        _single_qubit_and_controlled_pattern(pattern_angles, wires[:4])
        _single_qubit_and_controlled_pattern(pattern_angles, wires[4:])
    else:
        _single_qubit_and_controlled_pattern(pattern_angles, wires)
    for start in (*range(0, width, 2), *range(1, width, 2)):
        _two_qubit_block(pair_angles, wires[start], wires[(start + 1) % width])


def _pool(angles: torch.Tensor, control: int, target: int) -> None:
    qml.CRZ(angles[0], wires=(control, target))
    qml.PauliX(wires=control)
    qml.CRX(angles[1], wires=(control, target))


def mfa_circuit(features: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Eight-qubit amplitude circuit with the original sixteen-outcome readout."""

    qml.AmplitudeEmbedding(features, wires=tuple(range(8)), normalize=True)
    _convolution(angles[0:31], tuple(range(8)), first_stage=True)
    _pool(angles[31:33], 7, 6)
    _pool(angles[31:33], 1, 0)
    _convolution(angles[33:72], (0, 2, 3, 4, 5, 6), first_stage=False)
    _pool(angles[72:74], 3, 2)
    _pool(angles[72:74], 5, 4)
    _convolution(angles[74:105], (0, 2, 4, 6), first_stage=False)
    return qml.probs(wires=(0, 2, 4, 6))


class MFAProbabilityNLLLoss(nn.Module):
    """Negative log of the original, unrenormalized ten-class probabilities."""

    def forward(self, probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 2 or probabilities.shape[1] != 10:
            raise ValueError("MFA expects the first ten probabilities for each image.")
        selected = probabilities.gather(1, labels.long().reshape(-1, 1)).squeeze(1)
        return -selected.clamp(1e-15, 1.0 - 1e-15).log().mean()


class MFALossCollector(LossCollector):
    def __init__(self) -> None:
        super().__init__()
        self._loss_fn = MFAProbabilityNLLLoss()

    @property
    def loss_fn(self) -> MFAProbabilityNLLLoss:
        return self._loss_fn

    def export_state(self) -> dict:
        return {**super().export_state(), "loss_name": "MFAProbabilityNLLLoss"}


class MFAClassifier(nn.Module):
    """PCA preprocessing plus the published ten-class quantum classifier.

    Fit PCA once on training images via ``fit_preprocessing``. Its mean and
    components are buffers, so ordinary model checkpoints preserve the full
    classifier, including the fitted preprocessing transform.
    """

    _qcnn_quantum_parameter_module = True

    def __init__(self, *, image_size: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        if image_size < 16 or image_size & (image_size - 1):
            raise ValueError("MFA needs a power-of-two image side of at least 16.")
        if num_classes != 10:
            raise ValueError("The selected published MFA architecture has ten classes.")
        self.image_size = image_size
        self.num_classes = num_classes
        self.angles = nn.Parameter(torch.randn(105, dtype=torch.float64))
        self.register_buffer("pca_mean", torch.zeros(image_size * image_size, dtype=torch.float64))
        self.register_buffer("pca_components", torch.zeros(256, image_size * image_size, dtype=torch.float64))
        self.register_buffer("pca_ready", torch.tensor(False))
        device = qml.device("default.qubit", wires=8)
        self._qnode = qml.QNode(mfa_circuit, device, interface="torch", diff_method="backprop")

    def fit_preprocessing(self, images: torch.Tensor, *, seed: int) -> None:
        """Fit a 256-component PCA using only the supplied training images."""

        from sklearn.decomposition import PCA

        if images.ndim != 3 or tuple(images.shape[1:]) != (self.image_size,) * 2:
            raise ValueError("PCA training images have an unexpected shape.")
        if images.shape[0] < 256:
            raise ValueError("PCA requires at least 256 training images.")
        flat_images = images.detach().cpu().double().reshape(images.shape[0], -1).numpy()
        fitted = PCA(n_components=256, svd_solver="auto", random_state=seed).fit(flat_images)
        self.pca_mean.copy_(torch.from_numpy(fitted.mean_).to(self.pca_mean))
        self.pca_components.copy_(torch.from_numpy(fitted.components_).to(self.pca_components))
        self.pca_ready.fill_(True)

    def build_loss_collector(self) -> MFALossCollector:
        return MFALossCollector()

    def quantum_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        if not bool(self.pca_ready.item()):
            raise RuntimeError("Fit MFA PCA on the training images before evaluating the model.")
        if images.ndim != 3 or tuple(images.shape[1:]) != (self.image_size,) * 2:
            raise ValueError("MFA images have an unexpected shape.")
        flat_images = images.to(dtype=torch.float64).reshape(images.shape[0], -1)
        components = (flat_images - self.pca_mean) @ self.pca_components.T
        return self._qnode(components, self.angles)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantum_probabilities(images)[:, :10]
