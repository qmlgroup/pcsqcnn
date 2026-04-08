"""Collector abstractions for training statistics and loss aggregation.

The module provides stateful statistic collectors that own epoch-level history
for train/test phases. Collectors are designed to be driven by
``ImageClassifierRunner`` during ``fit(...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

PhaseName = str


@dataclass(frozen=True)
class StatisticBatchContext:
    """Batch information passed into collector hooks.

    Attributes:
        phase: Current loader phase, conventionally ``"train"`` or ``"test"``.
        epoch: 1-based epoch number associated with the batch.
        batch_size: Number of samples in the batch.
        logits: Model outputs for the batch. These may be detached by the
            caller when gradient tracking is not needed by the collector.
        labels: Target labels for the batch on the runner device.
        loss: Scalar loss tensor associated with ``logits`` and ``labels``.
        model: Wrapped model instance.
        optimizer: Active optimizer when training, otherwise ``None``.
    """

    phase: PhaseName
    epoch: int
    batch_size: int
    logits: torch.Tensor
    labels: torch.Tensor
    loss: torch.Tensor
    model: nn.Module
    optimizer: Optimizer | None


class StatisticCollector:
    """Base class for scalar epoch-history collectors.

    Args:
        name: Stable collector name used in history mappings and serialized
            export payloads.

    Subclasses typically implement ``compute_batch_value(...)``. More advanced
    collectors can override lifecycle hooks and use ``record_batch_value(...)``
    directly to aggregate their own scalar statistic while keeping extra state
    for ``export_state()``.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("StatisticCollector name must be a non-empty string.")
        self.name = name
        self._history: dict[PhaseName, list[float]] = {"train": [], "test": []}
        self._phase_totals: dict[PhaseName, float] = {}
        self._phase_counts: dict[PhaseName, int] = {}

    def on_run_start(self, *, num_epochs: int) -> None:
        """Reset collector state before a new multi-epoch run."""

        del num_epochs
        self._history = {"train": [], "test": []}
        self._phase_totals = {}
        self._phase_counts = {}

    def on_run_end(self) -> None:
        """Finalize a run after the last epoch."""

    def on_epoch_start(self, *, phase: PhaseName, epoch: int) -> None:
        """Prepare per-phase aggregation state for one epoch."""

        del epoch
        self._phase_totals[phase] = 0.0
        self._phase_counts[phase] = 0

    def on_batch_end(self, context: StatisticBatchContext) -> None:
        """Consume one batch.

        The default implementation converts the subclass-provided scalar value
        into a sample-weighted epoch accumulator.
        """

        batch_value = self.compute_batch_value(context)
        if batch_value is None:
            return
        self.record_batch_value(
            phase=context.phase,
            value=batch_value,
            batch_size=context.batch_size,
        )

    def on_epoch_end(self, *, phase: PhaseName, epoch: int) -> float | None:
        """Finalize the current epoch and append the phase history entry."""

        del epoch
        total_samples = self._phase_counts.get(phase, 0)
        if total_samples < 1:
            return None

        epoch_value = self._phase_totals[phase] / total_samples
        self._history.setdefault(phase, []).append(epoch_value)
        return epoch_value

    def compute_batch_value(self, context: StatisticBatchContext) -> float | torch.Tensor | None:
        """Compute the scalar batch statistic.

        Subclasses should override this when the collector is driven by the
        default ``on_batch_end(...)`` implementation.
        """

        del context
        return None

    def record_batch_value(
        self,
        *,
        phase: PhaseName,
        value: float | torch.Tensor,
        batch_size: int,
    ) -> None:
        """Accumulate one sample-weighted batch scalar into the current epoch."""

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        scalar_value = _coerce_scalar(value, name=self.name)
        self._phase_totals[phase] = self._phase_totals.get(phase, 0.0) + scalar_value * batch_size
        self._phase_counts[phase] = self._phase_counts.get(phase, 0) + batch_size

    def history(self) -> dict[PhaseName, list[float]]:
        """Return a copy of the recorded scalar history."""

        return {
            phase: list(values)
            for phase, values in self._history.items()
        }

    def export_state(self) -> dict[str, Any]:
        """Export additional serializable collector state."""

        return {"history": self.history()}


class LossCollector(StatisticCollector):
    """Base class for optimization objectives that also record history.

    Args:
        name: Collector name used for serialized loss state. The default name
            is ``"loss"``.
    """

    def __init__(self, name: str = "loss") -> None:
        super().__init__(name=name)

    @property
    def loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the differentiable optimization loss for one batch."""

        return self.loss_fn(logits, labels)

    def compute_batch_value(self, context: StatisticBatchContext) -> float | torch.Tensor:
        return context.loss


class CrossEntropyLossCollector(LossCollector):
    """Default cross-entropy optimization objective and loss-history collector.

    Args:
        name: Collector/export name for the aggregated loss history.
    """

    def __init__(self, *, name: str = "loss") -> None:
        super().__init__(name=name)
        self._loss_fn = nn.CrossEntropyLoss()

    @property
    def loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self._loss_fn

    def export_state(self) -> dict[str, Any]:
        exported = super().export_state()
        exported["loss_name"] = self._loss_fn.__class__.__name__
        return exported


class AccuracyCollector(StatisticCollector):
    """Default multiclass accuracy collector operating on logits.

    Args:
        name: Collector/export name for the aggregated accuracy history.
    """

    def __init__(self, *, name: str = "accuracy") -> None:
        super().__init__(name=name)

    def compute_batch_value(self, context: StatisticBatchContext) -> float:
        predictions = context.logits.argmax(dim=1)
        return float((predictions == context.labels).to(dtype=torch.float32).mean().item())


def validate_collector_names(
    *,
    loss_collector: LossCollector,
    metric_collectors: Mapping[str, StatisticCollector],
) -> None:
    """Validate that collector names are unique and respect reserved names."""

    if loss_collector.name != "loss":
        metric_name_conflict = loss_collector.name in metric_collectors
        if metric_name_conflict:
            raise ValueError(
                f"Collector name {loss_collector.name!r} is used by both the loss collector and "
                "a metric collector."
            )

    seen_names = {loss_collector.name}
    for metric_name in metric_collectors:
        if metric_name == "loss":
            raise ValueError("Metric collectors cannot use the reserved name 'loss'.")
        if metric_name in seen_names:
            raise ValueError(f"Duplicate collector name {metric_name!r}.")
        seen_names.add(metric_name)


def _coerce_scalar(value: float | torch.Tensor, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Collector {name!r} must return a scalar, got shape {tuple(value.shape)}."
            )
        return float(value.detach().item())
    return float(value)
