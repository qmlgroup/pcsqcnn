"""Training and evaluation helpers for grayscale image classifiers in ``qcnn``.

The module provides a DataLoader-first runner that centralizes the notebook-side
training loop used across the article models. It is intentionally
model-agnostic at the logits boundary: any ``nn.Module`` that accepts image
batches with shape ``[B, X, Y]`` and returns class logits can be wrapped here.

The runner validates the common grayscale-image domain constraints used by the
project, moves batches onto the chosen device, aggregates loss and metrics
sample-wise across loaders, and records compact epoch-level training history.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from qcnn.layout import is_power_of_two
from qcnn.statistics import (
    AccuracyCollector,
    CrossEntropyLossCollector,
    LossCollector,
    StatisticBatchContext,
    StatisticCollector,
    validate_collector_names,
)

ProgressFactory = Callable[[range], Iterable[int]]
EpochEndCallback = Callable[[int, "EvaluationSummary", "EvaluationSummary | None", "TrainingHistory"], None]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated loss and metric values for one loader pass.

    Attributes:
        loss: Sample-weighted mean loss across the evaluated loader.
        metrics: Mapping from metric name to the corresponding sample-weighted
            mean scalar value across the evaluated loader.
    """

    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True, init=False)
class TrainingHistory:
    """Epoch-level training and evaluation history returned by ``fit``.

    Attributes:
        train_epoch: Epoch numbers where train summaries were recorded.
        test_epoch: Epoch numbers where test summaries were recorded.
        train_loss: Sample-weighted mean training loss after each epoch.
        test_loss: Sample-weighted mean test loss after each evaluated epoch.
        train_metrics: Per-metric epoch histories for the training loader.
        test_metrics: Per-metric epoch histories for the test loader.
    """

    train_epoch: list[int]
    test_epoch: list[int]
    train_loss: list[float]
    test_loss: list[float]
    train_metrics: dict[str, list[float]]
    test_metrics: dict[str, list[float]]

    def __init__(
        self,
        *,
        train_epoch: list[int] | None = None,
        test_epoch: list[int] | None = None,
        train_loss: list[float] | None = None,
        test_loss: list[float] | None = None,
        train_metrics: dict[str, list[float]] | None = None,
        test_metrics: dict[str, list[float]] | None = None,
    ) -> None:
        object.__setattr__(self, "train_epoch", list(train_epoch or []))
        object.__setattr__(self, "test_epoch", list(test_epoch or []))
        object.__setattr__(self, "train_loss", list(train_loss or []))
        object.__setattr__(self, "test_loss", list(test_loss or []))
        object.__setattr__(
            self,
            "train_metrics",
            {metric_name: list(metric_values) for metric_name, metric_values in (train_metrics or {}).items()},
        )
        object.__setattr__(
            self,
            "test_metrics",
            {metric_name: list(metric_values) for metric_name, metric_values in (test_metrics or {}).items()},
        )
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        if len(self.train_loss) != len(self.train_epoch):
            raise ValueError(
                f"train_loss must have length {len(self.train_epoch)}, got {len(self.train_loss)}."
            )
        if len(self.test_loss) != len(self.test_epoch):
            raise ValueError(
                f"test_loss must have length {len(self.test_epoch)}, got {len(self.test_loss)}."
            )
        for metric_name, metric_values in self.train_metrics.items():
            if len(metric_values) != len(self.train_epoch):
                raise ValueError(
                    f"train_metrics[{metric_name!r}] must have length {len(self.train_epoch)}, "
                    f"got {len(metric_values)}."
                )
        for metric_name, metric_values in self.test_metrics.items():
            if len(metric_values) != len(self.test_epoch):
                raise ValueError(
                    f"test_metrics[{metric_name!r}] must have length {len(self.test_epoch)}, "
                    f"got {len(metric_values)}."
                )


try:  # pragma: no cover - depends on torch serialization API availability
    from torch.serialization import add_safe_globals

    add_safe_globals([TrainingHistory])
except (ImportError, AttributeError):
    pass


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute multiclass accuracy from logits and integer labels.

    Args:
        logits: Real tensor with shape ``[B, C]`` containing class logits.
        labels: Integer tensor with shape ``[B]`` containing target classes.

    Returns:
        The batch accuracy as a Python ``float`` in ``[0, 1]``.

    Notes:
        The helper assumes the predicted class is ``argmax(logits, dim=1)`` and
        performs no reduction weighting beyond the plain batch mean. Loader-wide
        weighting is handled by ``ImageClassifierRunner``.
    """

    return float((logits.argmax(dim=1) == labels).to(dtype=torch.float32).mean().item())
class ImageClassifierRunner:
    """Stateful training/evaluation wrapper for grayscale logits models.

    Args:
        model: Image classifier that accepts ``[B, X, Y]`` tensors and returns
            logits.
        loss_collector: Preferred optimization/loss collector. When omitted,
            ``CrossEntropyLossCollector`` is used.
        metric_collectors: Preferred sequence of statistic collectors. Built-in
            ``AccuracyCollector`` is included automatically unless a collector
            named ``"accuracy"`` is already present.
        optimizer: Optional optimizer used by the training methods. This may be
            ``None`` when the runner is used only for evaluation.
        device: Optional target device. If provided, the model is moved to it.
            If omitted, the runner infers the device from the model's first
            parameter or buffer and falls back to CPU.

    Attributes:
        model: Wrapped classifier module.
        loss_collector: Optimization/loss collector used by the runner.
        optimizer: Optional optimizer used for training methods.
        device: Device used for batch transfers and forward passes.
        metric_collectors: Ordered mapping of metric collector names to
            collector instances.
        last_train_loader: Latest training iterable passed to ``fit``.
        last_test_loader: Latest evaluation iterable passed to ``fit``.
        last_training_history: Latest ``TrainingHistory`` returned by ``fit``.

    Notes:
        The runner validates the common domain constraints shared by the
        article experiments: image batches must be rank-3, square, grayscale,
        and have power-of-two spatial size. Model-specific validation, such as
        a fixed configured ``image_size``, is still left to the wrapped model.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_collector: LossCollector | None = None,
        metric_collectors: Sequence[StatisticCollector] | None = None,
        optimizer: Optimizer | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.loss_collector = self._build_loss_collector(loss_collector=loss_collector)
        self.optimizer = optimizer
        self.device = self._resolve_device(device)
        self.metric_collectors = self._build_metric_collectors(metric_collectors=metric_collectors)
        validate_collector_names(
            loss_collector=self.loss_collector,
            metric_collectors=self.metric_collectors,
        )
        self.last_train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.last_test_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.last_training_history: TrainingHistory | None = None

    def run_forward_pass(self, images: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on one image batch.

        Args:
            images: Real tensor with shape ``[B, X, Y]``.

        Returns:
            Model logits for the moved batch.

        Notes:
            The method validates the generic grayscale-image batch contract and
            moves the image batch to ``self.device`` before calling the wrapped
            model.
        """

        self._validate_batch(images)
        device_images = images.to(self.device, non_blocking=True)
        return self.model(device_images)

    def run_backward_pass(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        set_to_none: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one backward pass without stepping the optimizer.

        Args:
            images: Real tensor with shape ``[B, X, Y]``.
            labels: Integer tensor with shape ``[B]``.
            set_to_none: Forwarded to ``model.zero_grad``.

        Returns:
            A pair ``(logits, loss)`` for the current batch.

        Notes:
            This method does not require an optimizer. It zeroes model
            gradients, computes the loss, and backpropagates through it.
        """

        self.model.train()
        self.model.zero_grad(set_to_none=set_to_none)
        device_images, device_labels = self._move_batch_to_device(images, labels)
        logits = self.model(device_images)
        loss = self.loss_collector.compute_loss(logits, device_labels)
        loss.backward()
        return logits, loss

    def run_training_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        set_to_none: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one full optimization step on a batch.

        Args:
            images: Real tensor with shape ``[B, X, Y]``.
            labels: Integer tensor with shape ``[B]``.
            set_to_none: Forwarded to ``model.zero_grad``.

        Returns:
            A pair ``(logits, loss)`` computed before the optimizer step.

        Notes:
            The method requires ``self.optimizer``. It performs backward
            propagation and then updates parameters via ``optimizer.step()``.
        """

        optimizer = self._require_optimizer("run_training_step")
        logits, loss = self.run_backward_pass(images, labels, set_to_none=set_to_none)
        optimizer.step()
        return logits, loss

    def evaluate_loader(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        requires_grad: bool = False,
    ) -> EvaluationSummary:
        """Evaluate the wrapped model on a loader.

        Args:
            data_loader: Iterable of ``(images, labels)`` batches.
            requires_grad: Whether to preserve the test-time autograd graph for
                gradient-based collectors.

        Returns:
            ``EvaluationSummary`` with sample-weighted mean loss and metrics.

        Notes:
            The method switches the model into evaluation mode and aggregates
            every scalar by batch size so the final result is independent of the
            chosen batching. By default it runs under ``torch.no_grad()``.
        """

        self.model.eval()
        totals = self._initialize_summary_totals()
        total_samples = 0

        grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
        with grad_context:
            for images, labels in data_loader:
                device_images, device_labels = self._move_batch_to_device(images, labels)
                logits = self.model(device_images)
                loss = self.loss_collector.compute_loss(logits, device_labels)
                batch_context = self._build_batch_context(
                    phase="test",
                    epoch=0,
                    logits=logits if requires_grad else logits.detach(),
                    labels=device_labels.detach(),
                    loss=loss if requires_grad else loss.detach(),
                )
                self._accumulate_summary_batch(totals, batch_context)
                total_samples += batch_context.batch_size

        return self._summary_from_totals(totals, total_samples)

    def train_epoch(
        self,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        set_to_none: bool = True,
    ) -> EvaluationSummary:
        """Train the wrapped model for one epoch over a loader.

        Args:
            train_loader: Iterable of ``(images, labels)`` batches.
            set_to_none: Forwarded to ``run_training_step``.

        Returns:
            ``EvaluationSummary`` with sample-weighted mean loss and metrics for
            the epoch.

        Notes:
            The method requires ``self.optimizer`` and records metrics from the
            pre-update logits of each training batch.
        """

        self._require_optimizer("train_epoch")
        totals = self._initialize_summary_totals()
        total_samples = 0

        for images, labels in train_loader:
            self._validate_batch(images, labels)
            device_labels = labels.to(device=self.device, dtype=torch.long, non_blocking=True)
            logits, loss = self.run_training_step(images, labels, set_to_none=set_to_none)
            batch_context = self._build_batch_context(
                phase="train",
                epoch=0,
                logits=logits.detach(),
                labels=device_labels.detach(),
                loss=loss.detach(),
            )
            self._accumulate_summary_batch(totals, batch_context)
            total_samples += batch_context.batch_size

        return self._summary_from_totals(totals, total_samples)

    def fit(
        self,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        test_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        num_epochs: int,
        set_to_none: bool = True,
        test_requires_grad: bool = False,
        test_evaluation_interval_epochs: int = 10,
        progress_factory: ProgressFactory | None = None,
        epoch_end_callback: EpochEndCallback | None = None,
    ) -> TrainingHistory:
        """Train for multiple epochs and record epoch-level history.

        Args:
            train_loader: Training iterable of ``(images, labels)`` batches.
            test_loader: Evaluation iterable of ``(images, labels)`` batches.
            num_epochs: Number of epochs to run. Must be positive.
            set_to_none: Forwarded to ``train_epoch`` / ``run_training_step``.
            test_requires_grad: Whether epoch-end test evaluation should
                preserve the autograd graph for gradient-based collectors.
            test_evaluation_interval_epochs: Frequency of epoch-end test
                evaluation. Test is still always evaluated on the final epoch.
            progress_factory: Optional callable that receives
                ``range(1, num_epochs + 1)`` and returns the iterable used by
                the epoch loop.
            epoch_end_callback: Optional callback invoked after each epoch is
                recorded in ``TrainingHistory``.

        Returns:
            ``TrainingHistory`` with epoch-level train/test loss and metrics.

        Notes:
            The method requires ``self.optimizer`` and intentionally stores only
            epoch-level aggregates, not per-batch traces.
        """

        self._require_optimizer("fit")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}.")
        if test_evaluation_interval_epochs < 1:
            raise ValueError(
                "test_evaluation_interval_epochs must be positive, "
                f"got {test_evaluation_interval_epochs}."
            )

        metric_names = tuple(self.metric_collectors)
        history = TrainingHistory(
            train_epoch=[],
            test_epoch=[],
            train_loss=[],
            test_loss=[],
            train_metrics={metric_name: [] for metric_name in metric_names},
            test_metrics={metric_name: [] for metric_name in metric_names},
        )

        epoch_range = range(1, num_epochs + 1)
        epoch_iterator = progress_factory(epoch_range) if progress_factory is not None else epoch_range
        all_collectors = self._all_collectors()
        for collector in all_collectors:
            collector.on_run_start(num_epochs=num_epochs)

        try:
            for epoch in epoch_iterator:
                train_summary = self._fit_phase(
                    train_loader,
                    phase="train",
                    epoch=epoch,
                    set_to_none=set_to_none,
                )
                history.train_epoch.append(epoch)
                history.train_loss.append(train_summary.loss)
                for metric_name in metric_names:
                    history.train_metrics[metric_name].append(train_summary.metrics[metric_name])

                test_summary: EvaluationSummary | None = None
                if epoch == num_epochs or epoch % test_evaluation_interval_epochs == 0:
                    test_summary = self._fit_phase(
                        test_loader,
                        phase="test",
                        epoch=epoch,
                        set_to_none=set_to_none,
                        requires_grad=test_requires_grad,
                    )
                    history.test_epoch.append(epoch)
                    history.test_loss.append(test_summary.loss)
                    for metric_name in metric_names:
                        history.test_metrics[metric_name].append(test_summary.metrics[metric_name])

                if epoch_end_callback is not None:
                    epoch_end_callback(epoch, train_summary, test_summary, history)
        finally:
            for collector in all_collectors:
                collector.on_run_end()

        self.last_train_loader = train_loader
        self.last_test_loader = test_loader
        self.last_training_history = history
        return history

    def export_collector_states(self) -> dict[str, object]:
        """Export the current state of the loss and metric collectors."""

        collector_states: dict[str, object] = {
            self.loss_collector.name: self.loss_collector.export_state(),
        }
        for collector_name, collector in self.metric_collectors.items():
            collector_states[collector_name] = collector.export_state()
        return collector_states

    def _fit_phase(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        phase: str,
        epoch: int,
        set_to_none: bool,
        requires_grad: bool = False,
    ) -> EvaluationSummary:
        training = phase == "train"
        if training:
            self._require_optimizer("_fit_phase")
        else:
            self.model.eval()

        all_collectors = self._all_collectors()
        for collector in all_collectors:
            collector.on_epoch_start(phase=phase, epoch=epoch)

        total_samples = 0
        if training:
            for images, labels in data_loader:
                self._validate_batch(images, labels)
                device_labels = labels.to(device=self.device, dtype=torch.long, non_blocking=True)
                logits, loss = self.run_training_step(images, labels, set_to_none=set_to_none)
                batch_context = self._build_batch_context(
                    phase=phase,
                    epoch=epoch,
                    logits=logits.detach(),
                    labels=device_labels.detach(),
                    loss=loss.detach(),
                )
                for collector in all_collectors:
                    collector.on_batch_end(batch_context)
                total_samples += batch_context.batch_size
        else:
            grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
            with grad_context:
                for images, labels in data_loader:
                    device_images, device_labels = self._move_batch_to_device(images, labels)
                    logits = self.model(device_images)
                    loss = self.loss_collector.compute_loss(logits, device_labels)
                    batch_context = self._build_batch_context(
                        phase=phase,
                        epoch=epoch,
                        logits=logits if requires_grad else logits.detach(),
                        labels=device_labels.detach(),
                        loss=loss if requires_grad else loss.detach(),
                    )
                    for collector in all_collectors:
                        collector.on_batch_end(batch_context)
                    total_samples += batch_context.batch_size

        if total_samples < 1:
            raise ValueError("data_loader must yield at least one sample.")

        loss_value = self.loss_collector.on_epoch_end(phase=phase, epoch=epoch)
        if loss_value is None:
            raise ValueError(f"Loss collector {self.loss_collector.name!r} did not record a value.")

        metric_values: dict[str, float] = {}
        for collector_name, collector in self.metric_collectors.items():
            metric_value = collector.on_epoch_end(phase=phase, epoch=epoch)
            if metric_value is None:
                raise ValueError(f"Metric collector {collector_name!r} did not record a value.")
            metric_values[collector_name] = metric_value

        return EvaluationSummary(
            loss=loss_value,
            metrics=metric_values,
        )

    def _resolve_device(self, explicit_device: torch.device | str | None) -> torch.device:
        if explicit_device is not None:
            resolved_device = torch.device(explicit_device)
            self.model = self.model.to(resolved_device)
            return resolved_device

        first_parameter = next(self.model.parameters(), None)
        if first_parameter is not None:
            return first_parameter.device

        first_buffer = next(self.model.buffers(), None)
        if first_buffer is not None:
            return first_buffer.device

        return torch.device("cpu")

    def _build_loss_collector(
        self,
        *,
        loss_collector: LossCollector | None,
    ) -> LossCollector:
        if loss_collector is not None:
            return loss_collector
        return CrossEntropyLossCollector()

    def _build_metric_collectors(
        self,
        *,
        metric_collectors: Sequence[StatisticCollector] | None,
    ) -> OrderedDict[str, StatisticCollector]:
        if metric_collectors is not None:
            collector_sequence = list(metric_collectors)
            collector_names = [collector.name for collector in collector_sequence]
            if len(set(collector_names)) != len(collector_names):
                duplicate_names = [
                    name
                    for index, name in enumerate(collector_names)
                    if name in collector_names[:index]
                ]
                duplicate_name = duplicate_names[0]
                raise ValueError(f"Duplicate collector name {duplicate_name!r}.")
            if "accuracy" not in collector_names:
                collector_sequence.insert(0, AccuracyCollector())
            return OrderedDict((collector.name, collector) for collector in collector_sequence)

        metric_mapping: OrderedDict[str, StatisticCollector] = OrderedDict()
        metric_mapping["accuracy"] = AccuracyCollector()
        return metric_mapping

    def _validate_batch(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        if images.ndim != 3:
            raise ValueError(
                "ImageClassifierRunner expects image tensors with shape [B, X, Y], "
                f"got {tuple(images.shape)}."
            )

        _, x_dim, y_dim = images.shape
        if x_dim != y_dim:
            raise ValueError(f"Input images must be square, got X={x_dim}, Y={y_dim}.")
        if not is_power_of_two(x_dim):
            raise ValueError(f"image side length must be a positive power of two, got {x_dim}.")

        if labels is None:
            return

        if labels.ndim != 1:
            raise ValueError(
                "ImageClassifierRunner expects label tensors with shape [B], "
                f"got {tuple(labels.shape)}."
            )
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "images and labels must have matching batch dimensions, "
                f"got {images.shape[0]} and {labels.shape[0]}."
            )

    def _move_batch_to_device(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_batch(images, labels)
        return (
            images.to(self.device, non_blocking=True),
            labels.to(device=self.device, dtype=torch.long, non_blocking=True),
        )

    def _build_batch_context(
        self,
        *,
        phase: str,
        epoch: int,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> StatisticBatchContext:
        return StatisticBatchContext(
            phase=phase,
            epoch=epoch,
            batch_size=int(labels.shape[0]),
            logits=logits,
            labels=labels,
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
        )

    def _all_collectors(self) -> tuple[StatisticCollector, ...]:
        return (self.loss_collector, *self.metric_collectors.values())

    def _initialize_summary_totals(self) -> dict[str, float]:
        totals = {"loss": 0.0}
        for metric_name in self.metric_collectors:
            totals[metric_name] = 0.0
        return totals

    def _accumulate_summary_batch(
        self,
        totals: dict[str, float],
        context: StatisticBatchContext,
    ) -> None:
        totals["loss"] += self._coerce_collector_value(
            self.loss_collector.compute_batch_value(context),
            collector_name=self.loss_collector.name,
        ) * context.batch_size
        for metric_name, collector in self.metric_collectors.items():
            batch_value = collector.compute_batch_value(context)
            if batch_value is None:
                raise ValueError(f"Metric collector {metric_name!r} did not produce a scalar batch value.")
            totals[metric_name] += self._coerce_collector_value(
                batch_value,
                collector_name=metric_name,
            ) * context.batch_size

    def _coerce_collector_value(
        self,
        metric_value: float | torch.Tensor | None,
        *,
        collector_name: str,
    ) -> float:
        if metric_value is None:
            raise ValueError(f"Collector {collector_name!r} did not produce a scalar value.")
        if isinstance(metric_value, torch.Tensor):
            if metric_value.numel() != 1:
                raise ValueError(
                    f"Collector {collector_name!r} must return a scalar, "
                    f"got shape {tuple(metric_value.shape)}."
                )
            return float(metric_value.detach().item())
        return float(metric_value)

    def _summary_from_totals(self, totals: dict[str, float], total_samples: int) -> EvaluationSummary:
        if total_samples < 1:
            raise ValueError("data_loader must yield at least one sample.")

        averaged_metrics = {
            metric_name: metric_total / total_samples
            for metric_name, metric_total in totals.items()
            if metric_name != "loss"
        }
        return EvaluationSummary(
            loss=totals["loss"] / total_samples,
            metrics=averaged_metrics,
        )

    def _require_optimizer(self, method_name: str) -> Optimizer:
        if self.optimizer is None:
            raise RuntimeError(f"ImageClassifierRunner.{method_name} requires an optimizer.")
        return self.optimizer
