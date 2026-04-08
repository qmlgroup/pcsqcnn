import warnings

import pytest
import torch

import qcnn.data as data_module
from qcnn.data import (
    TensorImageDataset,
    _prepare_mnist_splits_from_tensors,
    _resize_and_normalize_images,
    prepare_mnist_splits,
)


class FakeMNIST:
    def __init__(self, root: str, train: bool, download: bool) -> None:
        del root, download
        if train:
            self.data = TRAIN_IMAGES.clone()
            self.targets = TRAIN_LABELS.clone()
        else:
            self.data = TEST_IMAGES.clone()
            self.targets = TEST_LABELS.clone()


TRAIN_IMAGES = torch.arange(12 * 4 * 4, dtype=torch.uint8).reshape(12, 4, 4)
TRAIN_LABELS = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
TEST_IMAGES = torch.arange(6 * 4 * 4, dtype=torch.uint8).reshape(6, 4, 4)
TEST_LABELS = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)


def test_prepare_mnist_splits_is_seeded_and_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with pytest.warns(UserWarning):
        first = prepare_mnist_splits(
            root="unused",
            samples_per_class=2,
            image_size=4,
            seed=7,
            download=False,
        )

    with pytest.warns(UserWarning):
        second = prepare_mnist_splits(
            root="unused",
            samples_per_class=2,
            image_size=4,
            seed=7,
            download=False,
        )

    assert torch.equal(first.train.images, second.train.images)
    assert torch.equal(first.train.labels, second.train.labels)
    assert torch.bincount(first.train.labels, minlength=3).tolist() == [2, 2, 2]


def test_prepare_mnist_splits_keeps_standard_test_split_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with pytest.warns(UserWarning):
        splits = prepare_mnist_splits(
            root="unused",
            samples_per_class=2,
            image_size=4,
            seed=3,
            download=False,
        )

    assert len(splits.test) == len(TEST_LABELS)
    assert torch.equal(splits.test.labels, TEST_LABELS)


def test_prepare_mnist_splits_attaches_dataset_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with pytest.warns(UserWarning):
        splits = prepare_mnist_splits(
            root="unused",
            samples_per_class=2,
            image_size=4,
            seed=11,
            download=False,
        )

    assert splits.train.metadata == {
        "dataset_name": "MNIST",
        "image_size": 4,
        "scaled_image_size": 4,
        "max_offset": 0,
        "normalization_range": [0.0, 1.0],
        "resize_mode": "bilinear",
        "split": "train",
        "samples_per_class": 2,
        "seed": 11,
    }
    assert splits.test.metadata == {
        "dataset_name": "MNIST",
        "image_size": 4,
        "scaled_image_size": 4,
        "max_offset": 0,
        "normalization_range": [0.0, 1.0],
        "resize_mode": "bilinear",
        "split": "test",
        "test_split": "standard",
    }


def test_prepare_mnist_splits_with_none_keeps_full_train_split_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        splits = prepare_mnist_splits(
            root="unused",
            samples_per_class=None,
            image_size=4,
            seed=11,
            download=False,
        )

    assert len(caught) == 1
    assert "image_size deviates" in str(caught[0].message)
    assert "samples_per_class" not in str(caught[0].message)
    assert len(splits.train) == len(TRAIN_LABELS)
    assert torch.equal(splits.train.labels, TRAIN_LABELS)
    assert torch.allclose(splits.train.images, TRAIN_IMAGES.to(dtype=torch.float32) / 255.0)
    assert splits.train.metadata == {
        "dataset_name": "MNIST",
        "image_size": 4,
        "scaled_image_size": 4,
        "max_offset": 0,
        "normalization_range": [0.0, 1.0],
        "resize_mode": "bilinear",
        "split": "train",
        "samples_per_class": None,
        "seed": 11,
    }


def test_prepare_mnist_splits_with_none_ignores_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with pytest.warns(UserWarning, match="image_size deviates"):
        first = prepare_mnist_splits(
            root="unused",
            samples_per_class=None,
            image_size=4,
            seed=1,
            download=False,
        )

    with pytest.warns(UserWarning, match="image_size deviates"):
        second = prepare_mnist_splits(
            root="unused",
            samples_per_class=None,
            image_size=4,
            seed=9,
            download=False,
        )

    assert torch.equal(first.train.images, second.train.images)
    assert torch.equal(first.train.labels, second.train.labels)


def test_resize_and_normalization_send_zero_and_one_to_unit_interval() -> None:
    images = torch.tensor(
        [
            [[0, 0], [0, 0]],
            [[255, 255], [255, 255]],
        ],
        dtype=torch.uint8,
    )

    processed = _resize_and_normalize_images(images, image_size=4)

    assert processed.shape == (2, 4, 4)
    assert torch.allclose(processed[0], torch.zeros((4, 4)))
    assert torch.allclose(processed[1], torch.ones((4, 4)))


def test_prepare_mnist_splits_legacy_mode_matches_resize_only_output() -> None:
    splits = _prepare_mnist_splits_from_tensors(
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        test_images=TEST_IMAGES,
        test_labels=TEST_LABELS,
        samples_per_class=None,
        image_size=4,
        seed=0,
    )

    assert torch.allclose(splits.train.images, TRAIN_IMAGES.to(dtype=torch.float32) / 255.0)
    assert torch.allclose(splits.test.images, TEST_IMAGES.to(dtype=torch.float32) / 255.0)


def test_prepare_mnist_splits_places_resized_images_on_centered_zero_canvas() -> None:
    splits = _prepare_mnist_splits_from_tensors(
        train_images=torch.tensor([[[255, 255], [255, 255]]], dtype=torch.uint8),
        train_labels=torch.tensor([0], dtype=torch.long),
        test_images=torch.tensor([[[255, 255], [255, 255]]], dtype=torch.uint8),
        test_labels=torch.tensor([1], dtype=torch.long),
        samples_per_class=None,
        image_size=4,
        scaled_image_size=2,
        max_offset=0,
        seed=0,
    )

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(splits.train.images[0], expected)
    assert torch.allclose(splits.test.images[0], expected)


def test_prepare_mnist_splits_translations_are_seeded_for_train_and_test() -> None:
    train_images = torch.arange(12 * 4 * 4, dtype=torch.uint8).reshape(12, 4, 4)
    test_images = torch.arange(10 * 4 * 4, dtype=torch.uint8).reshape(10, 4, 4)
    train_labels = torch.arange(12, dtype=torch.long) % 3
    test_labels = torch.arange(10, dtype=torch.long) % 3

    first = _prepare_mnist_splits_from_tensors(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        samples_per_class=None,
        image_size=8,
        scaled_image_size=4,
        max_offset=2,
        seed=17,
    )
    second = _prepare_mnist_splits_from_tensors(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        samples_per_class=None,
        image_size=8,
        scaled_image_size=4,
        max_offset=2,
        seed=17,
    )
    third = _prepare_mnist_splits_from_tensors(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        samples_per_class=None,
        image_size=8,
        scaled_image_size=4,
        max_offset=2,
        seed=23,
    )

    assert torch.equal(first.train.images, second.train.images)
    assert torch.equal(first.test.images, second.test.images)
    assert not torch.equal(first.train.images, third.train.images)
    assert not torch.equal(first.test.images, third.test.images)


def test_prepare_mnist_splits_translation_pipeline_warns_about_article_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "MNIST", FakeMNIST)

    with pytest.warns(UserWarning, match="translation preprocessing deviates"):
        prepare_mnist_splits(
            root="unused",
            samples_per_class=None,
            image_size=16,
            scaled_image_size=12,
            max_offset=2,
            seed=0,
            download=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scaled_image_size": 0}, "scaled_image_size must be positive"),
        ({"scaled_image_size": 5}, "less than or equal to image_size"),
        ({"max_offset": -1}, "max_offset must be non-negative"),
    ],
)
def test_prepare_mnist_splits_rejects_invalid_translation_config(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_mnist_splits_from_tensors(
            train_images=TRAIN_IMAGES,
            train_labels=TRAIN_LABELS,
            test_images=TEST_IMAGES,
            test_labels=TEST_LABELS,
            samples_per_class=2,
            image_size=4,
            seed=0,
            **kwargs,
        )


def test_prepare_mnist_splits_clips_translation_offset_to_canvas_slack() -> None:
    clipped = _prepare_mnist_splits_from_tensors(
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        test_images=TEST_IMAGES,
        test_labels=TEST_LABELS,
        samples_per_class=2,
        image_size=4,
        scaled_image_size=2,
        max_offset=2,
        seed=17,
    )
    reference = _prepare_mnist_splits_from_tensors(
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        test_images=TEST_IMAGES,
        test_labels=TEST_LABELS,
        samples_per_class=2,
        image_size=4,
        scaled_image_size=2,
        max_offset=1,
        seed=17,
    )

    assert torch.equal(clipped.train.images, reference.train.images)
    assert torch.equal(clipped.test.images, reference.test.images)
    assert clipped.train.metadata["max_offset"] == 2
    assert clipped.test.metadata["max_offset"] == 2


def test_prepare_mnist_splits_from_tensors_rejects_non_power_of_two_resize() -> None:
    with pytest.raises(ValueError, match="power of two"):
        _prepare_mnist_splits_from_tensors(
            train_images=TRAIN_IMAGES,
            train_labels=TRAIN_LABELS,
            test_images=TEST_IMAGES,
            test_labels=TEST_LABELS,
            samples_per_class=2,
            image_size=6,
            seed=0,
        )


def test_tensor_image_dataset_returns_tensor_labels_and_supports_device_transfer() -> None:
    dataset = TensorImageDataset(
        torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4),
        torch.tensor([3, 7], dtype=torch.long),
        metadata={"split": "train"},
    )

    image, label = dataset[1]
    moved = dataset.to("cpu")

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.shape == ()
    assert int(label.item()) == 7
    assert torch.equal(moved.images, dataset.images)
    assert torch.equal(moved.labels, dataset.labels)
    assert moved.metadata == dataset.metadata


def test_prepared_mnist_splits_to_moves_both_datasets_and_preserves_metadata() -> None:
    splits = data_module.PreparedMnistSplits(
        train=TensorImageDataset(
            torch.zeros((1, 4, 4), dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            metadata={"split": "train"},
        ),
        test=TensorImageDataset(
            torch.ones((1, 4, 4), dtype=torch.float32),
            torch.tensor([1], dtype=torch.long),
            metadata={"split": "test"},
        ),
    )

    moved = splits.to("cpu")

    assert torch.equal(moved.train.images, splits.train.images)
    assert torch.equal(moved.train.labels, splits.train.labels)
    assert moved.train.metadata == {"split": "train"}
    assert torch.equal(moved.test.images, splits.test.images)
    assert torch.equal(moved.test.labels, splits.test.labels)
    assert moved.test.metadata == {"split": "test"}
