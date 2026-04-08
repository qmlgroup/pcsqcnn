"""Deterministic MNIST preprocessing utilities for the Torch-first ``qcnn`` library.

The preprocessing contract used by the training pipeline is:

1. Optionally choose a balanced seeded subset from the MNIST training split.
2. Keep the standard MNIST test split membership unchanged.
3. Resize every image to ``scaled_image_size`` with bilinear interpolation.
4. Place every resized image onto a zero-filled ``image_size x image_size``
   canvas with an optional seeded integer translation.
5. Normalize raw grayscale values into ``[0, 1]``.

All tensors produced by this module follow the image contract ``[N, X, Y]`` and
are ready to be consumed directly by classical baselines or by
``FrqiEncoder2D``, which now owns the quantum-specific angle mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

from qcnn.article import warn_for_article_alignment
from qcnn.layout import is_power_of_two

_TRAIN_SHIFT_SEED_OFFSET = 1_000_003
_TEST_SHIFT_SEED_OFFSET = 2_000_033


@dataclass(frozen=True)
class PreparedMnistSplits:
    """Container for prepared train/test datasets.

    Attributes:
        train: Tensor-backed training dataset after optional balanced
            sub-selection, resize/canvas preprocessing, and grayscale
            normalization into ``[0, 1]``.
        test: Tensor-backed test dataset after the same resize/canvas
            preprocessing and grayscale normalization. The canonical MNIST test
            split membership is unchanged.
    """

    train: "TensorImageDataset"
    test: "TensorImageDataset"

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "PreparedMnistSplits":
        """Return prepared splits with both datasets moved onto ``device``."""

        return PreparedMnistSplits(
            train=self.train.to(device, non_blocking=non_blocking),
            test=self.test.to(device, non_blocking=non_blocking),
        )


class TensorImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A tensor-backed dataset with image samples of shape ``[X, Y]``.

    Args:
        images: Tensor with shape ``[N, X, Y]``.
        labels: Tensor with shape ``[N]``. Labels are converted to
            ``torch.long`` on construction.
        metadata: Optional experiment metadata attached to the dataset. This is
            intended for read-only provenance, for example preprocessing
            settings such as ``image_size`` or ``samples_per_class``.

    Returns:
        ``__getitem__(index)`` returns ``(image, label)`` where ``image`` has
        shape ``[X, Y]`` and ``label`` is a scalar ``torch.long`` tensor.

    Raises:
        ValueError: If the input tensors do not satisfy the ``[N, X, Y]`` and
            ``[N]`` contracts or if their leading dimensions disagree.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if images.ndim != 3:
            raise ValueError(f"images must have shape [N, X, Y], got {tuple(images.shape)}.")
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}.")
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same leading dimension.")

        self.images = images.contiguous()
        self.labels = labels.to(dtype=torch.long).contiguous()
        self.metadata = dict(metadata) if metadata is not None else {}

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "TensorImageDataset":
        """Return a copy of the dataset with tensors moved onto ``device``."""

        return TensorImageDataset(
            self.images.to(device=device, non_blocking=non_blocking),
            self.labels.to(device=device, non_blocking=non_blocking),
            metadata=self.metadata,
        )


def _validate_image_size(image_size: int) -> None:
    """Validate that ``image_size`` is a positive power of two.

    Args:
        image_size: Requested square image side length.

    Raises:
        ValueError: If ``image_size`` is not a positive power of two.
    """

    if not is_power_of_two(image_size):
        raise ValueError(f"image_size must be a positive power of two, got {image_size}.")


def _validate_resize_side_length(image_size: int) -> None:
    """Validate that an intermediate resize side length is positive."""

    if image_size < 1:
        raise ValueError(f"resize image_size must be positive, got {image_size}.")


def _resolve_scaled_image_size(*, image_size: int, scaled_image_size: int | None) -> int:
    """Resolve and validate the intermediate resize side length."""

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
    """Validate deterministic canvas placement settings."""

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


def _select_balanced_subset_indices(
    labels: torch.Tensor,
    *,
    samples_per_class: int,
    seed: int,
) -> torch.Tensor:
    """Select a seeded balanced subset from integer class labels.

    Args:
        labels: Tensor with shape ``[N]`` containing class labels.
        samples_per_class: Number of samples to draw for every distinct class
            present in ``labels``.
        seed: Seed for the internal ``torch.Generator`` used by
            ``torch.randperm``.

    Returns:
        A 1D tensor of indices sorted in ascending order. Sorting restores the
        original dataset order after per-class random sampling.

    Raises:
        ValueError: If ``labels`` is not 1D, ``samples_per_class`` is not
            positive, or a class does not contain enough examples.
    """

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}.")
    if samples_per_class < 1:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices: list[torch.Tensor] = []
    for class_id in torch.unique(labels, sorted=True):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_indices.numel() < samples_per_class:
            raise ValueError(
                f"Class {int(class_id)} only has {class_indices.numel()} samples, "
                f"cannot select {samples_per_class}."
            )

        permutation = torch.randperm(class_indices.numel(), generator=generator)
        chosen = class_indices[permutation[:samples_per_class]]
        selected_indices.append(chosen)

    return torch.sort(torch.cat(selected_indices)).values


def _resize_and_normalize_images(
    images: torch.Tensor,
    *,
    image_size: int,
) -> torch.Tensor:
    """Resize images and normalize their brightness into ``[0, 1]``.

    Contract:
        ``images`` must have shape ``[N, X, Y]``. The output has shape
        ``[N, image_size, image_size]``.

    Formula:
        Raw images are represented elementwise as ``x / 255`` when the input
        dtype/range indicates uint8-style MNIST values. The resized output
        remains in ``[0, 1]``.

    Notes:
        Bilinear interpolation is applied through ``torch.nn.functional`` with
        ``align_corners=False`` to match the rest of the project.
    """

    _validate_resize_side_length(image_size)

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


def _shift_seed_for_split(*, seed: int, split: str) -> int:
    """Derive a deterministic split-local translation seed."""

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
    """Center resized images on a zero canvas and apply seeded integer shifts."""

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
    canvas = torch.zeros(
        (num_images, image_size, image_size),
        dtype=images.dtype,
        device=images.device,
    )
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


def _prepare_mnist_splits_from_tensors(
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    samples_per_class: int | None,
    image_size: int,
    scaled_image_size: int | None = None,
    max_offset: int = 0,
    seed: int,
) -> PreparedMnistSplits:
    """Build prepared train/test splits from already-loaded MNIST tensors.

    Args:
        train_images: Training images with shape ``[N_train, X, Y]``.
        train_labels: Training labels with shape ``[N_train]``.
        test_images: Test images with shape ``[N_test, X, Y]``.
        test_labels: Test labels with shape ``[N_test]``.
        samples_per_class: Number of training examples to keep per class. When
            ``None``, the full standard MNIST train split is kept.
        image_size: Final square canvas side length returned by the function.
        scaled_image_size: Intermediate square side length used by the bilinear
            resize step before canvas placement. When omitted, ``image_size``
            is used and the function behaves like the legacy centered pipeline.
        max_offset: Maximum integer translation magnitude applied independently
            along both spatial axes after centering the resized image on the
            final canvas.
        seed: Seed used for balanced training-subset selection when
            ``samples_per_class`` is not ``None`` and for deterministic split-
            local image translations.

    Returns:
        ``PreparedMnistSplits`` where the training split is either balanced or
        full-sized depending on ``samples_per_class`` and both splits are
        resized, placed onto the final canvas, and normalized into the image
        tensor contract ``[N, X, Y]`` with grayscale values in ``[0, 1]``.

    Raises:
        ValueError: If input shapes are inconsistent.
    """

    if train_images.ndim != 3 or test_images.ndim != 3:
        raise ValueError("train_images and test_images must have shape [N, X, Y].")
    if train_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("train_labels and test_labels must have shape [N].")
    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("train_images and train_labels must have the same leading dimension.")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("test_images and test_labels must have the same leading dimension.")

    _validate_image_size(image_size)
    resolved_scaled_image_size = _resolve_scaled_image_size(
        image_size=image_size,
        scaled_image_size=scaled_image_size,
    )
    _validate_translation_config(
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
    )

    if samples_per_class is None:
        selected_indices = torch.arange(train_labels.shape[0])
    else:
        selected_indices = _select_balanced_subset_indices(
            train_labels,
            samples_per_class=samples_per_class,
            seed=seed,
        )
    prepared_train_images = _resize_and_normalize_images(
        train_images[selected_indices],
        image_size=resolved_scaled_image_size,
    )
    prepared_train_images = _place_images_on_canvas(
        prepared_train_images,
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
        seed=seed,
        split="train",
    )
    prepared_test_images = _resize_and_normalize_images(
        test_images,
        image_size=resolved_scaled_image_size,
    )
    prepared_test_images = _place_images_on_canvas(
        prepared_test_images,
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
        seed=seed,
        split="test",
    )

    common_metadata = {
        "dataset_name": "MNIST",
        "image_size": image_size,
        "scaled_image_size": resolved_scaled_image_size,
        "max_offset": max_offset,
        "normalization_range": [0.0, 1.0],
        "resize_mode": "bilinear",
    }

    return PreparedMnistSplits(
        train=TensorImageDataset(
            prepared_train_images,
            train_labels[selected_indices],
            metadata={
                **common_metadata,
                "split": "train",
                "samples_per_class": samples_per_class,
                "seed": seed,
            },
        ),
        test=TensorImageDataset(
            prepared_test_images,
            test_labels,
            metadata={
                **common_metadata,
                "split": "test",
                "test_split": "standard",
            },
        ),
    )


def prepare_mnist_splits(
    *,
    root: str | Path,
    samples_per_class: int | None = 20,
    image_size: int = 16,
    scaled_image_size: int | None = None,
    max_offset: int = 0,
    seed: int = 0,
    download: bool = True,
) -> PreparedMnistSplits:
    """Prepare deterministic MNIST train/test splits for the qcnn workflow.

    Args:
        root: Directory used by ``torchvision.datasets.MNIST``.
        samples_per_class: Number of training examples to keep for every class.
            When ``None``, the full standard MNIST train split is used.
        image_size: Final square canvas side length returned by the function.
            Must be a positive power of two.
        scaled_image_size: Optional intermediate square side length used by the
            bilinear resize step before canvas placement. When omitted,
            ``image_size`` is used.
        max_offset: Maximum integer translation magnitude applied
            independently along both spatial axes after centering the resized
            image on the final canvas.
        seed: Seed for balanced training-subset selection and deterministic
            split-local image translations. When ``samples_per_class`` is
            ``None`` and ``max_offset == 0``, the seed does not affect the
            returned tensors.
        download: Forwarded to ``torchvision.datasets.MNIST``.

    Returns:
        Prepared train/test tensor datasets. The standard MNIST test split
        membership is kept unchanged; the training split is either reduced to a
        balanced seeded subset with ``samples_per_class`` examples per class or
        kept in full when ``samples_per_class`` is ``None``.

    Notes:
        The function emits article-alignment warnings when ``image_size``,
        ``samples_per_class``, or the optional translation pipeline deviate
        from the article-aligned defaults. The article-aligned encoder angle
        range is configured later in ``FrqiEncoder2D`` / ``PCSQCNN``.
    """

    _validate_image_size(image_size)
    resolved_scaled_image_size = _resolve_scaled_image_size(
        image_size=image_size,
        scaled_image_size=scaled_image_size,
    )
    _validate_translation_config(
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
    )

    warn_for_article_alignment(
        image_size=image_size,
        samples_per_class=samples_per_class,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
        stacklevel=2,
    )

    train_dataset = MNIST(root=str(root), train=True, download=download)
    test_dataset = MNIST(root=str(root), train=False, download=download)

    return _prepare_mnist_splits_from_tensors(
        train_images=train_dataset.data,
        train_labels=train_dataset.targets,
        test_images=test_dataset.data,
        test_labels=test_dataset.targets,
        samples_per_class=samples_per_class,
        image_size=image_size,
        scaled_image_size=resolved_scaled_image_size,
        max_offset=max_offset,
        seed=seed,
    )
