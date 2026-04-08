import pytest
import torch
from torch import nn

from qcnn import ClassicalCNN, ClassicalMLP
from qcnn.classic import resolve_classical_mlp_hidden_widths


@pytest.mark.parametrize(
    ("model_cls", "image_size", "num_classes"),
    [
        (ClassicalCNN, 8, 5),
        (ClassicalMLP, 8, 5),
    ],
)
def test_classical_models_return_logits_with_expected_shape(
    model_cls: type[nn.Module],
    image_size: int,
    num_classes: int,
) -> None:
    model = model_cls(image_size=image_size, num_classes=num_classes)
    images = torch.randn(3, image_size, image_size)

    logits = model(images)

    assert logits.shape == (3, num_classes)


def test_classical_cnn_uses_translated_mnist_architecture_and_parameter_count() -> None:
    model = ClassicalCNN(image_size=16, num_classes=10)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert isinstance(model.features[4], nn.AvgPool2d)
    assert isinstance(model.features[9], nn.AvgPool2d)
    assert isinstance(model.features[10], nn.AdaptiveAvgPool2d)
    assert model.features[10].output_size == 1
    assert model.classifier.in_features == 64
    assert model.classifier.bias is not None
    assert parameter_count == 47034


def test_classical_cnn_supports_optional_capacity_and_dropout_kwargs() -> None:
    model = ClassicalCNN(image_size=8, num_classes=5, base_channels=12, dropout=0.25)

    assert model.base_channels == 12
    assert model.dropout_rate == pytest.approx(0.25)
    assert model.features[0].out_channels == 12
    assert model.features[2].out_channels == 24
    assert model.features[5].out_channels == 36
    assert model.features[7].out_channels == 48
    assert model.classifier.in_features == 48
    assert model.dropout.p == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": 3, "num_classes": 5}, "image_size must be at least 4"),
        ({"image_size": 8, "num_classes": 0}, "num_classes must be positive"),
        ({"image_size": 8, "num_classes": 5, "base_channels": 3}, "base_channels must be at least 4"),
        ({"image_size": 8, "num_classes": 5, "dropout": -0.1}, "dropout must be in \\[0, 1\\)"),
        ({"image_size": 8, "num_classes": 5, "dropout": 1.0}, "dropout must be in \\[0, 1\\)"),
    ],
)
def test_classical_cnn_rejects_invalid_constructor_arguments(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassicalCNN(**kwargs)


@pytest.mark.parametrize(
    ("image_size", "expected_widths"),
    [
        (64, (11, 44, 44)),
        (32, (29, 116, 116)),
        (4, (47, 188, 188)),
    ],
)
def test_resolve_classical_mlp_hidden_widths_matches_expected_scales(
    image_size: int,
    expected_widths: tuple[int, int, int],
) -> None:
    assert resolve_classical_mlp_hidden_widths(image_size, num_classes=10) == expected_widths


def test_classical_mlp_uses_adaptive_comparable_budget_dense_architecture() -> None:
    model = ClassicalMLP(image_size=64, num_classes=10)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert model.hidden_widths == (11, 44, 44)
    assert model.input_layer.in_features == 4096
    assert model.input_layer.out_features == 11
    assert model.expansion_layer is not None
    assert model.expansion_layer.in_features == 11
    assert model.expansion_layer.out_features == 44
    assert model.hidden_layer is not None
    assert model.hidden_layer.in_features == 44
    assert model.hidden_layer.out_features == 44
    assert model.classifier.in_features == 44
    assert model.classifier.out_features == 10
    assert isinstance(model.input_activation, nn.GELU)
    assert isinstance(model.expansion_activation, nn.GELU)
    assert isinstance(model.hidden_activation, nn.GELU)
    assert model.input_dropout.p == pytest.approx(0.10)
    assert model.expansion_dropout.p == pytest.approx(0.10)
    assert model.hidden_dropout.p == pytest.approx(0.10)
    assert parameter_count == 48025


def test_classical_mlp_matches_expected_budget_at_32x32() -> None:
    model = ClassicalMLP(image_size=32, num_classes=10)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert model.hidden_widths == (29, 116, 116)
    assert parameter_count == 47947


def test_classical_mlp_supports_optional_hidden_widths_and_dropout_kwargs() -> None:
    model = ClassicalMLP(image_size=8, num_classes=5, hidden_widths=(7, 9, 9), dropout=0.25)

    assert model.hidden_widths == (7, 9, 9)
    assert model.dropout_rate == pytest.approx(0.25)
    assert model.input_layer.in_features == 64
    assert model.input_layer.out_features == 7
    assert model.expansion_layer is not None
    assert model.expansion_layer.in_features == 7
    assert model.expansion_layer.out_features == 9
    assert model.hidden_layer is not None
    assert model.hidden_layer.in_features == 9
    assert model.hidden_layer.out_features == 9
    assert model.classifier.in_features == 9
    assert model.classifier.out_features == 5
    assert model.input_dropout.p == pytest.approx(0.25)
    assert model.expansion_dropout.p == pytest.approx(0.25)
    assert model.hidden_dropout.p == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": 0, "num_classes": 5}, "image_size must be positive"),
        ({"image_size": 8, "num_classes": 0}, "num_classes must be positive"),
        ({"image_size": 8, "num_classes": 5, "target_parameter_budget": 0}, "target_parameter_budget must be positive"),
        ({"image_size": 8, "num_classes": 5, "hidden_widths": ()}, "hidden_widths must not be empty"),
        (
            {"image_size": 8, "num_classes": 5, "hidden_widths": (7, 0)},
            "hidden_widths must contain only positive widths",
        ),
        ({"image_size": 8, "num_classes": 5, "dropout": -0.1}, "dropout must be in \\[0, 1\\)"),
        ({"image_size": 8, "num_classes": 5, "dropout": 1.0}, "dropout must be in \\[0, 1\\)"),
    ],
)
def test_classical_mlp_rejects_invalid_constructor_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassicalMLP(**kwargs)


@pytest.mark.parametrize("model_cls", [ClassicalCNN, ClassicalMLP])
def test_classical_models_support_cross_entropy_backward(model_cls: type[nn.Module]) -> None:
    torch.manual_seed(0)
    model = model_cls(image_size=8, num_classes=4)
    images = torch.randn(2, 8, 8)
    labels = torch.tensor([0, 3], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    first_parameter = next(model.parameters())
    assert first_parameter.grad is not None
    assert torch.count_nonzero(first_parameter.grad) > 0


@pytest.mark.parametrize("model_cls", [ClassicalCNN, ClassicalMLP])
def test_classical_models_reject_non_image_batches(model_cls: type[nn.Module]) -> None:
    model = model_cls(image_size=8, num_classes=4)

    with pytest.raises(ValueError, match=r"\[B, X, Y\]"):
        model(torch.zeros(2, 1, 8, 8))


@pytest.mark.parametrize("model_cls", [ClassicalCNN, ClassicalMLP])
def test_classical_models_reject_wrong_spatial_shape(model_cls: type[nn.Module]) -> None:
    model = model_cls(image_size=8, num_classes=4)

    with pytest.raises(ValueError, match=r"\(8, 8\)"):
        model(torch.zeros(2, 8, 4))
