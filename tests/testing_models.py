"""Test-only helper models for qcnn automation and serialization suites."""

from __future__ import annotations

import torch


class CustomTinyClassifier(torch.nn.Module):
    def __init__(self, image_size: int, num_classes: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.classifier = torch.nn.Linear(image_size * image_size, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))
