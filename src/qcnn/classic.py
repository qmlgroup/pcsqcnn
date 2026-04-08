"""Classical MNIST baselines with the PCS-QCNN I/O contract.

The classes in this module implement the two classical reference models used in
the low-data MNIST comparisons while matching the public boundary expected by
``PCSQCNN``:

- inputs are image batches shaped ``[B, image_size, image_size]``
- outputs are logits shaped ``[B, num_classes]``
- softmax and the training loss remain outside the modules

`ClassicalCNN` is a compact translated-MNIST baseline with multiple small
convolutions, average pooling, and a global-average-pooled classifier head.
`ClassicalMLP` is a stronger pure-dense control baseline with multiple GELU
stages and dropout, but still no spatially structured front-end. Its default
hidden widths adapt to the input resolution so its trainable-parameter budget
stays comparable to the CNN baseline.
"""

from __future__ import annotations

import math
import torch
from torch import nn


DEFAULT_CLASSICAL_MLP_TARGET_PARAMETER_BUDGET = 47034


def _validate_image_batch(images: torch.Tensor, image_size: int, model_name: str) -> None:
    if images.ndim != 3:
        raise ValueError(
            f"{model_name} expects image tensors in shape [B, X, Y], got {tuple(images.shape)}."
        )

    if tuple(images.shape[1:]) != (image_size, image_size):
        raise ValueError(
            f"{model_name} expects image tensors with spatial shape "
            f"({image_size}, {image_size}), got {tuple(images.shape[1:])}."
        )


def _classical_mlp_parameter_count_for_widths(
    input_features: int,
    hidden_widths: tuple[int, ...],
    num_classes: int,
) -> int:
    layer_widths = (input_features, *hidden_widths, num_classes)
    return sum(
        input_width * output_width + output_width
        for input_width, output_width in zip(layer_widths, layer_widths[1:])
    )


def resolve_classical_mlp_hidden_widths(
    image_size: int,
    *,
    num_classes: int,
    target_parameter_budget: int = DEFAULT_CLASSICAL_MLP_TARGET_PARAMETER_BUDGET,
) -> tuple[int, int, int]:
    """Resolve adaptive default MLP widths with a comparable CNN budget."""

    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if target_parameter_budget <= 0:
        raise ValueError(
            "target_parameter_budget must be positive, "
            f"got {target_parameter_budget}."
        )

    input_features = image_size * image_size
    linear_term = input_features + 9 + (4 * num_classes)
    discriminant = (linear_term * linear_term) - (80 * (10 - target_parameter_budget))
    approximate_scale = (
        -linear_term + math.sqrt(max(float(discriminant), 0.0))
    ) / 40.0

    candidate_scales = {
        1,
        max(1, math.floor(approximate_scale) - 1),
        max(1, math.floor(approximate_scale)),
        max(1, math.ceil(approximate_scale)),
        max(1, math.ceil(approximate_scale) + 1),
    }
    best_scale = min(
        candidate_scales,
        key=lambda scale: (
            abs(
                _classical_mlp_parameter_count_for_widths(
                    input_features=input_features,
                    hidden_widths=(scale, 4 * scale, 4 * scale),
                    num_classes=num_classes,
                )
                - target_parameter_budget
            ),
            scale,
        ),
    )
    return (best_scale, 4 * best_scale, 4 * best_scale)


class ClassicalCNN(nn.Module):
    """Convolutional baseline for translated MNIST with PCS-QCNN-compatible I/O.

    Args:
        image_size: Square image side length expected at the model input.
        num_classes: Number of logits produced by the classifier head.
        base_channels: Width multiplier controlling convolutional capacity.
        dropout: Dropout probability applied before the final classifier.

    Returns:
        ``forward(images)`` returns logits with shape ``[B, num_classes]``.

    Notes:
        The network keeps convolutional weight sharing deep into the model and
        avoids a large position-specific flattening head. The default
        ``base_channels=16`` stack is:
        ``Conv(1,16,3) -> ReLU -> Conv(16,32,3) -> ReLU -> AvgPool(2) ->``
        ``Conv(32,48,3) -> ReLU -> Conv(48,64,3) -> ReLU -> AvgPool(2) ->``
        ``AdaptiveAvgPool2d(1) -> Dropout(0.10) -> Linear(64, num_classes)``.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        base_channels: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if image_size < 4:
            raise ValueError(f"image_size must be at least 4, got {image_size}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if base_channels < 4:
            raise ValueError(f"base_channels must be at least 4, got {base_channels}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout

        c1 = base_channels
        c2 = 2 * base_channels
        c3 = 3 * base_channels
        c4 = 4 * base_channels

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c3, out_channels=c4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c4, num_classes, bias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the CNN to a batch of prepared images.

        Args:
            images: Real image tensor with shape ``[B, image_size, image_size]``
                containing the normalized grayscale samples produced by
                ``prepare_mnist_splits``.

        Returns:
            Logits with shape ``[B, num_classes]``.

        """

        _validate_image_batch(images, image_size=self.image_size, model_name="ClassicalCNN")

        activations = images.unsqueeze(1)
        activations = self.features(activations)
        activations = activations.flatten(start_dim=1)
        activations = self.dropout(activations)
        return self.classifier(activations)


class ClassicalMLP(nn.Module):
    """Pure-dense translated-MNIST baseline with PCS-QCNN-compatible I/O.

    Args:
        image_size: Square image side length expected at the model input.
        num_classes: Number of logits produced by the classifier head.
        hidden_widths: Widths of the dense hidden stages after flattening. When
            omitted, the widths are resolved automatically from ``image_size``.
        target_parameter_budget: Target trainable-parameter budget used only
            when ``hidden_widths`` is omitted.
        dropout: Dropout probability applied after each hidden dense stage.

    Returns:
        ``forward(images)`` returns logits with shape ``[B, num_classes]``.

    Notes:
        The default dense stack keeps the shape ratio ``1:4:4`` and resolves
        the scale from ``image_size`` so the budget stays near the default CNN.
        For example, ``image_size=64`` resolves to ``(11, 44, 44)`` with
        ``48025`` trainable parameters, and ``image_size=32`` resolves to
        ``(29, 116, 116)`` with ``47947`` parameters.
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        hidden_widths: tuple[int, ...] | None = None,
        target_parameter_budget: int = DEFAULT_CLASSICAL_MLP_TARGET_PARAMETER_BUDGET,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if image_size < 1:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if target_parameter_budget <= 0:
            raise ValueError(
                "target_parameter_budget must be positive, "
                f"got {target_parameter_budget}."
            )
        if hidden_widths is None:
            resolved_hidden_widths = resolve_classical_mlp_hidden_widths(
                image_size,
                num_classes=num_classes,
                target_parameter_budget=target_parameter_budget,
            )
        else:
            resolved_hidden_widths = tuple(hidden_widths)
            if not resolved_hidden_widths:
                raise ValueError("hidden_widths must not be empty.")
            if any(width <= 0 for width in resolved_hidden_widths):
                raise ValueError(
                    "hidden_widths must contain only positive widths, "
                    f"got {resolved_hidden_widths}."
                )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.image_size = image_size
        self.num_classes = num_classes
        self.target_parameter_budget = target_parameter_budget
        self.hidden_widths = resolved_hidden_widths
        self.dropout_rate = dropout
        self.flatten = nn.Flatten(start_dim=1)
        self.input_layer = nn.Linear(image_size * image_size, resolved_hidden_widths[0])
        self.input_activation = nn.GELU()
        self.input_dropout = nn.Dropout(p=dropout)

        self.expansion_layer: nn.Linear | None = None
        self.expansion_activation: nn.GELU | None = None
        self.expansion_dropout: nn.Dropout | None = None
        if len(resolved_hidden_widths) >= 2:
            self.expansion_layer = nn.Linear(resolved_hidden_widths[0], resolved_hidden_widths[1])
            self.expansion_activation = nn.GELU()
            self.expansion_dropout = nn.Dropout(p=dropout)

        self.hidden_layer: nn.Linear | None = None
        self.hidden_activation: nn.GELU | None = None
        self.hidden_dropout: nn.Dropout | None = None
        if len(resolved_hidden_widths) >= 3:
            self.hidden_layer = nn.Linear(resolved_hidden_widths[1], resolved_hidden_widths[2])
            self.hidden_activation = nn.GELU()
            self.hidden_dropout = nn.Dropout(p=dropout)

        self.extra_hidden_layers = nn.ModuleList()
        self.extra_hidden_activations = nn.ModuleList()
        self.extra_hidden_dropouts = nn.ModuleList()
        for previous_width, next_width in zip(resolved_hidden_widths[2:], resolved_hidden_widths[3:]):
            self.extra_hidden_layers.append(nn.Linear(previous_width, next_width))
            self.extra_hidden_activations.append(nn.GELU())
            self.extra_hidden_dropouts.append(nn.Dropout(p=dropout))

        self.classifier = nn.Linear(resolved_hidden_widths[-1], num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to a batch of prepared images.

        Args:
            images: Real image tensor with shape ``[B, image_size, image_size]``
                containing the normalized grayscale samples produced by
                ``prepare_mnist_splits``.

        Returns:
            Logits with shape ``[B, num_classes]``.

        """

        _validate_image_batch(images, image_size=self.image_size, model_name="ClassicalMLP")

        activations = self.flatten(images)
        activations = self.input_layer(activations)
        activations = self.input_activation(activations)
        activations = self.input_dropout(activations)

        if self.expansion_layer is not None:
            activations = self.expansion_layer(activations)
            activations = self.expansion_activation(activations)
            activations = self.expansion_dropout(activations)

        if self.hidden_layer is not None:
            activations = self.hidden_layer(activations)
            activations = self.hidden_activation(activations)
            activations = self.hidden_dropout(activations)

        for layer, activation, dropout_layer in zip(
            self.extra_hidden_layers,
            self.extra_hidden_activations,
            self.extra_hidden_dropouts,
        ):
            activations = layer(activations)
            activations = activation(activations)
            activations = dropout_layer(activations)

        return self.classifier(activations)
