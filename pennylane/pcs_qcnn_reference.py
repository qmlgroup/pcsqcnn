"""Readable PennyLane reference implementation of the article PCS-QCNN circuit.

This module is intentionally inference-only. Torch remains the canonical
training/runtime backend, while this PennyLane path exists so a reader can:

- inspect a direct ``QFT -> multiplexer -> QFT†`` circuit rendering;
- check that the reference matches the article architecture at a readable level;
- load Torch PCS-QCNN weights from this repository; and
- reproduce Torch full readout probabilities and final logits.

The reference deliberately avoids Torch-side runtime shortcuts such as the
reduced Fourier junction and does not try to match Torch's intermediate state
conventions exactly. In particular, Torch omits the encoder's global
``1 / sqrt(XY)`` normalization, while this module uses a normalized
``qml.StatePrep`` state. The required parity target is the measured full
readout and the final logits, not raw intermediate amplitudes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag, expm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

_SUPPORTED_REAL_DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "float32": np.float32,
    "float64": np.float64,
}
_TRAIN_SHIFT_SEED_OFFSET = 1_000_003
_TEST_SHIFT_SEED_OFFSET = 2_000_033

_PAULI_SINGLE_QUBIT = (
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
)


@dataclass(frozen=True)
class _LayerGeometry:
    """Static wire bookkeeping for one PennyLane PCS layer."""

    block_parameter_shape: tuple[int, int, int, int, int]
    x_active_wires: tuple[str, ...]
    y_active_wires: tuple[str, ...]
    x_selected_condition_wires: tuple[str, ...]
    y_selected_condition_wires: tuple[str, ...]
    branch_control_wires: tuple[str, ...]
    active_operator_wires: tuple[str, ...]


class TensorImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Simple tensor-backed dataset with image samples of shape ``[X, Y]``."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.ndim != 3:
            raise ValueError(f"images must have shape [N, X, Y], got {tuple(images.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}.")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same leading dimension.")

        self.images = images.contiguous()
        self.labels = labels.to(dtype=torch.long).contiguous()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index].item())


def _normalize_real_dtype(dtype: Any) -> np.dtype:
    """Normalize supported dtype inputs to a NumPy real dtype."""

    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype in (torch.float32, torch.float64):
        return np.dtype(np.float64 if dtype == torch.float64 else np.float32)
    if dtype in (np.float32, np.float64):
        return np.dtype(dtype)
    if isinstance(dtype, str) and dtype in _SUPPORTED_REAL_DTYPES:
        return np.dtype(_SUPPORTED_REAL_DTYPES[dtype])
    raise ValueError(
        "dtype must be one of {'torch.float32', 'torch.float64', float32, float64}, "
        f"got {dtype!r}."
    )


def _dtype_name(dtype: np.dtype) -> str:
    """Return the Torch-style dtype label used by Torch model configs."""

    return "torch.float64" if dtype == np.dtype(np.float64) else "torch.float32"


def _complex_dtype_for(real_dtype: np.dtype) -> np.dtype:
    """Map a supported real dtype to the matching complex dtype."""

    return np.dtype(np.complex128 if real_dtype == np.dtype(np.float64) else np.complex64)


def _validate_brightness_range(
    brightness_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Validate and normalize the encoder brightness interval."""

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
            f"got {(start, end)!r}."
        )
    return start, end


def _is_power_of_two(value: int) -> bool:
    """Return whether an integer is a positive power of two."""

    return value > 0 and value & (value - 1) == 0


def _feature_qubits_from_parameter_count(parameter_count: int) -> int:
    """Infer the feature-register width from ``P = 4**f - 1``."""

    if parameter_count < 3:
        raise ValueError(
            "Expected Pauli coefficient axis length P = 4**f - 1 for some f >= 1, "
            f"got P={parameter_count}."
        )

    reduced = parameter_count + 1
    qubits = 0
    while reduced % 4 == 0:
        reduced //= 4
        qubits += 1
    if reduced != 1:
        raise ValueError(
            "Expected Pauli coefficient axis length P = 4**f - 1 for some f >= 1, "
            f"got P={parameter_count}."
        )
    return qubits


def _pauli_basis(num_qubits: int) -> np.ndarray:
    """Build the reduced Pauli basis in Torch-compatible lexicographic order."""

    if num_qubits < 1:
        raise ValueError(f"num_qubits must be at least 1, got {num_qubits}.")

    basis = []
    for labels in product(range(4), repeat=num_qubits):
        if all(label == 0 for label in labels):
            continue

        matrix = _PAULI_SINGLE_QUBIT[labels[0]]
        for label in labels[1:]:
            matrix = np.kron(matrix, _PAULI_SINGLE_QUBIT[label])
        basis.append(matrix)
    return np.stack(basis, axis=0)


def _infer_real_dtype_from_model_state(model_state: Mapping[str, Any]) -> np.dtype:
    """Infer the real dtype from a Torch-like ``state_dict`` payload."""

    for value in model_state.values():
        value_dtype = getattr(value, "dtype", None)
        if value_dtype is None:
            continue
        try:
            return _normalize_real_dtype(value_dtype)
        except ValueError:
            try:
                return _normalize_real_dtype(str(value_dtype))
            except ValueError:
                continue
    raise ValueError(
        "Could not infer dtype from model_state. Pass an explicit dtype in model_config."
    )


def _negated_mode_order(mode_dim: int) -> tuple[int, ...]:
    """Return the Fourier-mode order induced by switching Torch FFT to ``qml.QFT``."""

    return tuple((-mode_index) % mode_dim for mode_index in range(mode_dim))


def _remap_torch_block_parameters_for_pennylane_qft(
    block_parameters: np.ndarray,
) -> np.ndarray:
    """Reindex Torch Fourier-mode blocks into the built-in PennyLane QFT convention.

    Torch uses ``torch.fft.fft`` on the active spatial axes, while the readable
    reference circuit uses ``qml.QFT`` followed by ``qml.adjoint(qml.QFT)``.
    Those two Fourier conventions label the same modes with opposite signs, so
    exact readout/logit parity requires negating the active-mode indices of the
    loaded Torch block tensor along each active Fourier axis.
    """

    remapped = np.asarray(block_parameters)
    remapped = np.take(remapped, _negated_mode_order(remapped.shape[2]), axis=2)
    remapped = np.take(remapped, _negated_mode_order(remapped.shape[3]), axis=3)
    return remapped


def _branch_index_bits(branch_index: int, *, bit_count: int) -> tuple[int, ...]:
    """Expand one branch index into MSB-left control bits."""

    if bit_count == 0:
        return ()
    return tuple((branch_index >> shift) & 1 for shift in range(bit_count - 1, -1, -1))


def _to_numpy_array(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert NumPy- or Torch-like tensors into detached NumPy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.array(value, copy=True)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_confusion_matrix requires matplotlib. Install with the "
            "'notebook' or 'test' extra to enable plotting."
        ) from exc
    return plt


def _resize_and_normalize_images(images: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Resize MNIST-style grayscale images and normalize them into ``[0, 1]``."""

    if images.ndim != 3:
        raise ValueError(f"images must have shape [N, X, Y], got {tuple(images.shape)}.")
    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    float_images = images.to(dtype=torch.float32)
    if images.dtype == torch.uint8 or torch.amax(float_images) > 1.0 or torch.amin(float_images) < 0.0:
        float_images = float_images / 255.0

    resized = F.interpolate(
        float_images[:, None, :, :],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return resized


def _resolve_scaled_image_size(*, image_size: int, scaled_image_size: int | None) -> int:
    if scaled_image_size is None:
        return image_size
    if scaled_image_size < 1:
        raise ValueError(
            "scaled_image_size must be positive when provided, "
            f"got {scaled_image_size}."
        )
    if scaled_image_size > image_size:
        raise ValueError(
            "scaled_image_size must be less than or equal to image_size, "
            f"got scaled_image_size={scaled_image_size} and image_size={image_size}."
        )
    return int(scaled_image_size)


def _validate_translation_config(
    *,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
) -> None:
    if max_offset < 0:
        raise ValueError(f"max_offset must be non-negative, got {max_offset}.")

    if scaled_image_size > image_size:
        raise ValueError(
            "scaled_image_size must be less than or equal to image_size, "
            f"got scaled_image_size={scaled_image_size} and image_size={image_size}."
        )


def _resolve_effective_max_offset(
    *,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
) -> int:
    _validate_translation_config(
        image_size=image_size,
        scaled_image_size=scaled_image_size,
        max_offset=max_offset,
    )
    max_allowed_offset = (image_size - scaled_image_size) // 2
    return min(max_offset, max_allowed_offset)


def _shift_seed_for_split(*, seed: int, split: str) -> int:
    if split == "train":
        return int(seed) + _TRAIN_SHIFT_SEED_OFFSET
    if split == "test":
        return int(seed) + _TEST_SHIFT_SEED_OFFSET
    raise ValueError(f"Unsupported split {split!r}; expected 'train' or 'test'.")


def _place_images_on_canvas(
    images: torch.Tensor,
    *,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
    seed: int,
    split: str,
) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError(f"images must have shape [N, X, Y], got {tuple(images.shape)}.")
    if tuple(images.shape[1:]) != (scaled_image_size, scaled_image_size):
        raise ValueError(
            "images must already match scaled_image_size before canvas placement, "
            f"got spatial shape {tuple(images.shape[1:])} for scaled_image_size={scaled_image_size}."
        )

    if scaled_image_size == image_size and max_offset == 0:
        return images

    effective_max_offset = _resolve_effective_max_offset(
        image_size=image_size,
        scaled_image_size=scaled_image_size,
        max_offset=max_offset,
    )

    num_images = int(images.shape[0])
    canvas = torch.zeros((num_images, image_size, image_size), dtype=images.dtype)
    base_top = (image_size - scaled_image_size) // 2
    base_left = (image_size - scaled_image_size) // 2

    if effective_max_offset == 0:
        row_offsets = torch.zeros(num_images, dtype=torch.int64)
        col_offsets = torch.zeros(num_images, dtype=torch.int64)
    else:
        generator = torch.Generator()
        generator.manual_seed(_shift_seed_for_split(seed=seed, split=split))
        row_offsets = torch.randint(
            -effective_max_offset,
            effective_max_offset + 1,
            (num_images,),
            generator=generator,
        )
        col_offsets = torch.randint(
            -effective_max_offset,
            effective_max_offset + 1,
            (num_images,),
            generator=generator,
        )

    for index in range(num_images):
        top = base_top + int(row_offsets[index].item())
        left = base_left + int(col_offsets[index].item())
        canvas[index, top : top + scaled_image_size, left : left + scaled_image_size] = images[index]

    return canvas


def prepare_mnist_test_dataset(
    *,
    root: str | Path,
    image_size: int,
    scaled_image_size: int | None = None,
    max_offset: int = 0,
    seed: int = 0,
    download: bool = True,
) -> TensorImageDataset:
    """Prepare the standard MNIST test split with local resize/canvas logic."""

    dataset = MNIST(root=str(root), train=False, download=download)
    resolved_scaled_image_size = _resolve_scaled_image_size(
        image_size=image_size,
        scaled_image_size=scaled_image_size,
    )
    _validate_translation_config(
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
    )
    images = _resize_and_normalize_images(dataset.data, image_size=resolved_scaled_image_size)
    images = _place_images_on_canvas(
        images,
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
        seed=seed,
        split="test",
    )
    labels = dataset.targets.to(dtype=torch.long)
    return TensorImageDataset(images, labels)


def build_mnist_test_loader(
    *,
    dataset: Dataset[tuple[torch.Tensor, int]],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a sequential DataLoader for the reconstructed MNIST test split."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def evaluate_predictions(
    model: "PennyLanePCSQCNN",
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    progress_factory: Callable[
        [Iterable[tuple[torch.Tensor, torch.Tensor]]],
        Iterable[tuple[torch.Tensor, torch.Tensor]],
    ]
    | None = None,
) -> dict[str, Any]:
    """Run batched inference and return transparent prediction artifacts."""

    logits_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    batch_iterator = progress_factory(data_loader) if progress_factory is not None else data_loader

    for images, labels in batch_iterator:
        batch_logits = np.asarray(model.predict_logits(images.detach().cpu().numpy()))
        if batch_logits.ndim == 1:
            batch_logits = batch_logits[None, :]
        batch_predictions = batch_logits.argmax(axis=1).astype(np.int64, copy=False)
        batch_targets = labels.detach().cpu().numpy().astype(np.int64, copy=False)

        logits_batches.append(batch_logits)
        prediction_batches.append(batch_predictions)
        target_batches.append(batch_targets)

    if not logits_batches:
        raise ValueError("data_loader must yield at least one batch.")

    logits = np.concatenate(logits_batches, axis=0)
    predicted_labels = np.concatenate(prediction_batches, axis=0)
    true_labels = np.concatenate(target_batches, axis=0)
    confusion_matrix = np.zeros((model.num_classes, model.num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (true_labels, predicted_labels), 1)
    accuracy = float(np.mean(predicted_labels == true_labels))

    return {
        "logits": logits,
        "predicted_labels": predicted_labels,
        "true_labels": true_labels,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
    }


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
):
    """Plot a confusion matrix and return the created matplotlib axis."""

    plt = _require_matplotlib()

    matrix = np.asarray(confusion_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be a square rank-2 array, "
            f"got shape {tuple(matrix.shape)}."
        )

    num_classes = int(matrix.shape[0])
    labels = (
        tuple(str(index) for index in range(num_classes))
        if class_labels is None
        else tuple(str(label) for label in class_labels)
    )
    if len(labels) < num_classes:
        raise ValueError(
            f"class_labels must have at least {num_classes} entries, got {len(labels)}."
        )

    figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    image = axis.imshow(matrix, cmap="gray_r", interpolation="nearest")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(num_classes), labels=labels[:num_classes])
    axis.set_yticks(range(num_classes), labels=labels[:num_classes])

    threshold = float(matrix.max()) / 2.0 if matrix.size > 0 else 0.0
    for row_index in range(num_classes):
        for column_index in range(num_classes):
            cell_value = int(matrix[row_index, column_index])
            text_color = "white" if cell_value > threshold and threshold > 0.0 else "black"
            axis.text(
                column_index,
                row_index,
                f"{cell_value:d}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return axis


class PennyLanePCSQCNN:
    """Inference-only PennyLane reference for the article PCS-QCNN architecture.

    Args:
        image_size: Square image side length. Must be a positive power of two.
        num_classes: Number of output logits.
        feature_qubits: Number of feature-register qubits.
        quantum_layers: Number of PCS layers.
        brightness_range: Encoder-side angle interval used by the FRQI mapping.
        dtype: Supported real dtype label or NumPy dtype. Only ``float32`` and
            ``float64`` are accepted.

    Notes:
        The reference renders every PCS layer in the readable article order
        ``QFT -> multiplexer -> QFT†`` using PennyLane's built-in ``qml.QFT``.
        Torch checkpoint parity is preserved by remapping the loaded Torch
        Fourier-mode blocks into that convention at load time. The reference
        supports full readout only and intentionally does not implement
        training, finite-shot sampling, reduced-readout mode, or the no-QFT
        ablation.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        feature_qubits: int = 1,
        quantum_layers: int = 1,
        *,
        brightness_range: tuple[float, float] = (0.0, math.pi),
        dtype: Any = np.float32,
    ) -> None:
        if not _is_power_of_two(image_size):
            raise ValueError(f"image_size must be a positive power of two, got {image_size}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_qubits < 1:
            raise ValueError(f"feature_qubits must be at least 1, got {feature_qubits}.")
        if quantum_layers < 1:
            raise ValueError(f"quantum_layers must be positive, got {quantum_layers}.")

        index_qubits_per_axis = int(math.log2(image_size))
        if quantum_layers > index_qubits_per_axis:
            raise ValueError(
                "quantum_layers cannot exceed the number of index qubits per axis: "
                f"got {quantum_layers} > {index_qubits_per_axis}."
            )

        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.feature_qubits = int(feature_qubits)
        self.quantum_layers = int(quantum_layers)
        self.index_qubits_per_axis = index_qubits_per_axis
        self.feature_dim = 2 ** self.feature_qubits
        self.real_dtype = _normalize_real_dtype(dtype)
        self.complex_dtype = _complex_dtype_for(self.real_dtype)
        self.dtype = _dtype_name(self.real_dtype)
        self.brightness_range = _validate_brightness_range(brightness_range)

        self._x_wires = tuple(f"x{wire_idx}" for wire_idx in range(self.index_qubits_per_axis))
        self._y_wires = tuple(f"y{wire_idx}" for wire_idx in range(self.index_qubits_per_axis))
        self._feature_wires = tuple(f"f{wire_idx}" for wire_idx in range(self.feature_qubits))
        self._all_wires = self._x_wires + self._y_wires + self._feature_wires

        self._layer_geometries = self._build_layer_geometries()
        self._layer_block_parameters = [
            np.zeros(geometry.block_parameter_shape, dtype=self.real_dtype)
            for geometry in self._layer_geometries
        ]

        self.final_x_active_dim = self.image_size // (2 ** (self.quantum_layers - 1))
        self.final_y_active_dim = self.final_x_active_dim
        self.final_x_condition_dim = 2 ** (self.quantum_layers - 1)
        self.final_y_condition_dim = self.final_x_condition_dim
        self.classifier_in_features = self.final_x_active_dim * self.final_y_active_dim * self.feature_dim
        self.classifier_weight = np.zeros(
            (self.num_classes, self.classifier_in_features),
            dtype=self.real_dtype,
        )
        self.classifier_bias = np.zeros((self.num_classes,), dtype=self.real_dtype)

        self._measurement_wires = self._layer_geometries[-1].x_active_wires
        self._measurement_wires += self._final_x_condition_wires
        self._measurement_wires += self._layer_geometries[-1].y_active_wires
        self._measurement_wires += self._final_y_condition_wires
        self._measurement_wires += self._feature_wires

        self._device = qml.device("default.qubit", wires=self._all_wires)
        self._qnode = qml.QNode(self._circuit, self._device)

    @classmethod
    def from_torch_state(
        cls,
        model_config: Mapping[str, Any],
        model_state: Mapping[str, Any],
    ) -> "PennyLanePCSQCNN":
        """Create the reference model from Torch constructor kwargs and weights.

        Args:
            model_config: Torch-side constructor kwargs. Full-readout
                ``PCSQCNN`` configs are supported. The reference ignores
                ``use_reduced_fourier_junction`` because it always executes the
                explicit, readable layer stack.
            model_state: Mapping with keys ``multiplexers.<i>.block_parameters``,
                ``classifier.weight``, and ``classifier.bias``.

        Returns:
            A loaded ``PennyLanePCSQCNN`` instance ready for exact inference.
        """

        config = dict(model_config)
        if bool(config.get("reduce_readout_to_feature_distribution", False)):
            raise ValueError(
                "PennyLanePCSQCNN supports only full readout. "
                "Torch configs with reduce_readout_to_feature_distribution=True are not supported."
            )
        shot_budget = config.get("shot_budget")
        if shot_budget not in (None, 0):
            raise ValueError(
                "PennyLanePCSQCNN implements exact full-readout inference only. "
                f"Got unsupported shot_budget={shot_budget!r}."
            )
        if any(
            key.startswith("r_unitary_layer_") or key.startswith("r_adjoint_layer_")
            for key in model_state
        ):
            raise ValueError(
                "PennyLanePCSQCNN only supports Torch PCSQCNN checkpoints, not PCSQCNNNoQFT."
            )

        model = cls(
            image_size=int(config["image_size"]),
            num_classes=int(config["num_classes"]),
            feature_qubits=int(config["feature_qubits"]),
            quantum_layers=int(config["quantum_layers"]),
            brightness_range=tuple(config["brightness_range"]),
            dtype=config.get("dtype", _infer_real_dtype_from_model_state(model_state)),
        )
        model._load_torch_state(model_state)
        return model

    def quantum_readout(self, images: Any) -> np.ndarray:
        """Run the PennyLane circuit and return Torch-equivalent readout marginals.

        Args:
            images: Array-like batch ``[B, X, Y]`` or single image ``[X, Y]``.

        Returns:
            NumPy array with shape ``[B, X_active, Y_active, F]``. This is the
            supported Torch-parity surface for the reference backend.
        """

        batch_images, _ = self._normalize_images(images)
        readouts = [
            self._probs_to_readout(
                np.asarray(self._qnode(self._build_frqi_statevector(image)))
            )
            for image in batch_images
        ]
        return np.stack(readouts, axis=0).astype(self.real_dtype, copy=False)

    def predict_logits(self, images: Any) -> np.ndarray:
        """Run the full PennyLane reference model and return logits.

        Args:
            images: Array-like batch ``[B, X, Y]`` or single image ``[X, Y]``.

        Returns:
            NumPy logits with shape ``[B, C]`` or ``[C]`` for a single image.
        """

        batch_images, single_input = self._normalize_images(images)
        readout = self.quantum_readout(batch_images)
        flattened = readout.reshape(readout.shape[0], -1)
        logits = flattened @ self.classifier_weight.T + self.classifier_bias
        logits = logits.astype(self.real_dtype, copy=False)
        return logits[0] if single_input else logits

    def draw(self, image: Any) -> str:
        """Return a text diagram of the explicit PennyLane circuit for one image.

        Args:
            image: Single image with shape ``[X, Y]``.

        Returns:
            Text circuit diagram generated by ``qml.draw``.
        """

        image_array = np.asarray(image, dtype=self.real_dtype)
        if image_array.ndim != 2:
            raise ValueError(
                "draw(image) expects a single image with shape [X, Y], "
                f"got {tuple(image_array.shape)}."
            )
        if tuple(image_array.shape) != (self.image_size, self.image_size):
            raise ValueError(
                "draw(image) expects spatial shape "
                f"({self.image_size}, {self.image_size}), got {tuple(image_array.shape)}."
            )

        drawer = qml.draw(
            self._qnode,
            wire_order=self._all_wires,
            show_all_wires=True,
            show_matrices=False,
            decimals=None,
            max_length=180,
        )
        return drawer(self._build_frqi_statevector(image_array))

    def _build_layer_geometries(self) -> list[_LayerGeometry]:
        """Precompute wire bookkeeping for every PCS layer."""

        active_x_wires = list(self._x_wires)
        active_y_wires = list(self._y_wires)
        x_condition_wires: list[str] = []
        y_condition_wires: list[str] = []
        geometries: list[_LayerGeometry] = []

        for layer_idx in range(self.quantum_layers):
            x_condition_bits_to_use = 0 if layer_idx == 0 else 1
            y_condition_bits_to_use = 0 if layer_idx == 0 else 1
            x_active_dim = 2 ** len(active_x_wires)
            y_active_dim = 2 ** len(active_y_wires)
            x_selected_condition_wires = tuple(x_condition_wires[:x_condition_bits_to_use])
            y_selected_condition_wires = tuple(y_condition_wires[:y_condition_bits_to_use])

            geometries.append(
                _LayerGeometry(
                    block_parameter_shape=(
                        2 ** x_condition_bits_to_use,
                        2 ** y_condition_bits_to_use,
                        x_active_dim,
                        y_active_dim,
                        4 ** self.feature_qubits - 1,
                    ),
                    x_active_wires=tuple(active_x_wires),
                    y_active_wires=tuple(active_y_wires),
                    x_selected_condition_wires=x_selected_condition_wires,
                    y_selected_condition_wires=y_selected_condition_wires,
                    branch_control_wires=x_selected_condition_wires + y_selected_condition_wires,
                    active_operator_wires=tuple(active_x_wires) + tuple(active_y_wires) + self._feature_wires,
                )
            )

            if layer_idx < self.quantum_layers - 1:
                x_condition_wires.insert(0, active_x_wires.pop())
                y_condition_wires.insert(0, active_y_wires.pop())

        self._final_x_condition_wires = tuple(x_condition_wires)
        self._final_y_condition_wires = tuple(y_condition_wires)
        return geometries

    def _load_torch_state(self, model_state: Mapping[str, Any]) -> None:
        """Load Torch coefficients and remap them into the PennyLane QFT convention."""

        for layer_idx, geometry in enumerate(self._layer_geometries):
            key = f"multiplexers.{layer_idx}.block_parameters"
            if key not in model_state:
                raise ValueError(f"model_state is missing required key {key!r}.")

            parameters = _to_numpy_array(model_state[key], dtype=self.real_dtype)
            if tuple(parameters.shape) != geometry.block_parameter_shape:
                raise ValueError(
                    f"Expected {key} to have shape {geometry.block_parameter_shape}, "
                    f"got {tuple(parameters.shape)}."
                )
            parameter_count = parameters.shape[-1]
            if _feature_qubits_from_parameter_count(parameter_count) != self.feature_qubits:
                raise ValueError(
                    f"{key} implies a different feature_qubit count than configured."
                )
            self._layer_block_parameters[layer_idx] = _remap_torch_block_parameters_for_pennylane_qft(
                parameters
            )

        classifier_weight = _to_numpy_array(model_state.get("classifier.weight"), dtype=self.real_dtype)
        classifier_bias = _to_numpy_array(model_state.get("classifier.bias"), dtype=self.real_dtype)
        if tuple(classifier_weight.shape) != (self.num_classes, self.classifier_in_features):
            raise ValueError(
                "classifier.weight shape mismatch: expected "
                f"{(self.num_classes, self.classifier_in_features)}, "
                f"got {tuple(classifier_weight.shape)}."
            )
        if tuple(classifier_bias.shape) != (self.num_classes,):
            raise ValueError(
                f"classifier.bias shape mismatch: expected {(self.num_classes,)}, "
                f"got {tuple(classifier_bias.shape)}."
            )

        self.classifier_weight = classifier_weight
        self.classifier_bias = classifier_bias

    def _build_frqi_statevector(self, image: np.ndarray) -> np.ndarray:
        """Construct the normalized FRQI statevector used by ``StatePrep``."""

        image = np.asarray(image, dtype=self.real_dtype)
        if tuple(image.shape) != (self.image_size, self.image_size):
            raise ValueError(
                "FRQI encoding expects image shape "
                f"({self.image_size}, {self.image_size}), got {tuple(image.shape)}."
            )

        start, end = self.brightness_range
        angles = start + (end - start) * image
        state = np.zeros(
            (self.image_size, self.image_size, self.feature_dim),
            dtype=self.complex_dtype,
        )
        normalization = 1.0 / math.sqrt(self.image_size * self.image_size)
        state[..., 0] = normalization * np.sin(angles)
        state[..., 1] = normalization * np.cos(angles)
        return state.reshape(-1)

    def _build_block_diagonal_matrix(self, branch_parameters: np.ndarray) -> np.ndarray:
        """Convert one branch of Pauli coefficients into a block-diagonal unitary."""

        branch_parameters = np.asarray(branch_parameters, dtype=self.real_dtype)
        parameter_count = int(branch_parameters.shape[-1])
        num_feature_qubits = _feature_qubits_from_parameter_count(parameter_count)
        basis = _pauli_basis(num_feature_qubits)
        flattened_parameters = branch_parameters.reshape(-1, parameter_count)
        blocks = np.empty(
            (flattened_parameters.shape[0], self.feature_dim, self.feature_dim),
            dtype=np.complex128,
        )

        for block_idx, theta in enumerate(flattened_parameters):
            generator = np.einsum("p,pij->ij", theta.astype(np.float64), basis)
            blocks[block_idx] = expm(1.0j * generator)
        return block_diag(*blocks).astype(self.complex_dtype, copy=False)

    def _encode_frqi_state(self, statevector: np.ndarray) -> None:
        """Prepare the normalized FRQI state on the full PennyLane wire set."""

        qml.StatePrep(
            statevector,
            wires=self._all_wires,
            normalize=False,
            validate_norm=True,
        )

    def _apply_mode_multiplexer(
        self,
        matrix: np.ndarray,
        operator_wires: tuple[str, ...],
    ) -> None:
        """Apply one active-mode block-diagonal multiplexer matrix."""

        qml.QubitUnitary(matrix, wires=operator_wires, unitary_check=False)

    def _apply_pcs_layer(
        self,
        geometry: _LayerGeometry,
        block_parameters: np.ndarray,
    ) -> None:
        """Apply one readable PCS layer using built-in PennyLane QFT gates.

        The visible circuit follows the article order:

        ``QFT -> mode-dependent feature multiplexer -> QFT†``

        Torch checkpoint parity is preserved by remapping the stored Torch
        block tensor at load time so the built-in PennyLane ``qml.QFT``
        convention can be used directly here.
        """

        qml.QFT(wires=geometry.x_active_wires)
        qml.QFT(wires=geometry.y_active_wires)

        if not geometry.branch_control_wires:
            if tuple(block_parameters.shape[:2]) != (1, 1):
                raise ValueError(
                    "Unconditional PCS layers must have exactly one parameter branch."
                )
            matrix = self._build_block_diagonal_matrix(block_parameters[0, 0])
            self._apply_mode_multiplexer(matrix, geometry.active_operator_wires)
        else:
            expected_branch_count = 2 ** len(geometry.branch_control_wires)
            actual_branch_count = block_parameters.shape[0] * block_parameters.shape[1]
            if actual_branch_count != expected_branch_count:
                raise ValueError(
                    "Controlled PCS layer branch count does not match the selected control wires: "
                    f"expected {expected_branch_count}, got {actual_branch_count}."
                )
            for x_branch, y_branch in product(
                range(block_parameters.shape[0]),
                range(block_parameters.shape[1]),
            ):
                control_values = _branch_index_bits(
                    x_branch,
                    bit_count=len(geometry.x_selected_condition_wires),
                ) + _branch_index_bits(
                    y_branch,
                    bit_count=len(geometry.y_selected_condition_wires),
                )
                matrix = self._build_block_diagonal_matrix(block_parameters[x_branch, y_branch])
                controlled_multiplexer = qml.ctrl(
                    self._apply_mode_multiplexer,
                    control=geometry.branch_control_wires,
                    control_values=control_values,
                )
                controlled_multiplexer(matrix, geometry.active_operator_wires)

        qml.adjoint(qml.QFT)(wires=geometry.x_active_wires)
        qml.adjoint(qml.QFT)(wires=geometry.y_active_wires)

    def _probs_to_readout(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert full-wire probabilities into Torch-style marginal readout."""

        reshaped = np.asarray(probabilities).reshape(
            self.final_x_active_dim,
            self.final_x_condition_dim,
            self.final_y_active_dim,
            self.final_y_condition_dim,
            self.feature_dim,
        )
        return reshaped.sum(axis=(1, 3))

    def _normalize_images(self, images: Any) -> tuple[np.ndarray, bool]:
        """Validate image inputs and return a batch tensor."""

        image_array = np.asarray(images, dtype=self.real_dtype)
        single_input = image_array.ndim == 2
        if single_input:
            image_array = image_array[None, ...]
        if image_array.ndim != 3:
            raise ValueError(
                "Expected images with shape [B, X, Y] or [X, Y], "
                f"got {tuple(image_array.shape)}."
            )
        if tuple(image_array.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError(
                "Expected images with spatial shape "
                f"({self.image_size}, {self.image_size}), got {tuple(image_array.shape[1:])}."
            )
        return image_array, single_input

    def _circuit(self, statevector: np.ndarray) -> Any:
        """PennyLane qnode body for one prepared FRQI statevector."""

        self._encode_frqi_state(statevector)
        for geometry, block_parameters in zip(
            self._layer_geometries,
            self._layer_block_parameters,
            strict=True,
        ):
            self._apply_pcs_layer(geometry, block_parameters)
        return qml.probs(wires=self._measurement_wires)
