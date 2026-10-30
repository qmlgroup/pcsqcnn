import pytest
import torch

from qcnn.data import TensorImageDataset, stratified_validation_split


def _dataset() -> TensorImageDataset:
    labels = torch.arange(10).repeat_interleave(20)
    images = torch.arange(200, dtype=torch.float32)[:, None, None].expand(200, 4, 4).clone()
    return TensorImageDataset(images, labels, metadata={"split": "train", "seed": 0})


def test_split_is_class_balanced_disjoint_and_order_preserving() -> None:
    dataset = _dataset()

    training, validation = stratified_validation_split(dataset, validation_per_class=3, seed=0)

    assert len(training) == 170 and len(validation) == 30
    assert torch.bincount(validation.labels, minlength=10).tolist() == [3] * 10
    training_ids = training.images[:, 0, 0].long()
    validation_ids = validation.images[:, 0, 0].long()
    assert set(training_ids.tolist()).isdisjoint(validation_ids.tolist())
    assert sorted(training_ids.tolist() + validation_ids.tolist()) == list(range(200))
    assert torch.all(training_ids[1:] > training_ids[:-1])
    assert torch.all(validation_ids[1:] > validation_ids[:-1])
    assert validation.metadata["split"] == "validation"
    assert training.metadata["validation_per_class"] == 3


def test_split_is_deterministic_per_seed() -> None:
    dataset = _dataset()

    first = stratified_validation_split(dataset, validation_per_class=3, seed=0)[1].images
    again = stratified_validation_split(dataset, validation_per_class=3, seed=0)[1].images
    other = stratified_validation_split(dataset, validation_per_class=3, seed=1)[1].images

    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_split_rejects_holding_out_whole_classes() -> None:
    with pytest.raises(ValueError, match="cannot hold out"):
        stratified_validation_split(_dataset(), validation_per_class=20, seed=0)
