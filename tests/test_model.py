import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from qcnn import (
    AccuracyCollector,
    ClassicalMLP,
    CrossEntropyLossCollector,
    ImageClassifierRunner,
    PCSQCNN,
    StatisticBatchContext,
    StatisticCollector,
    TensorImageDataset,
    TrainingHistory,
    accuracy_from_logits,
)


class MeanLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2))
        return torch.stack((self.scale * means, -self.scale * means), dim=1)


class MeanConfidenceCollector(StatisticCollector):
    def __init__(self) -> None:
        super().__init__("confidence_gap")
        self.batch_trace: list[tuple[str, int]] = []

    def on_run_start(self, *, num_epochs: int) -> None:
        super().on_run_start(num_epochs=num_epochs)
        self.batch_trace = []

    def on_batch_end(self, context: StatisticBatchContext) -> None:
        super().on_batch_end(context)
        self.batch_trace.append((context.phase, context.epoch))

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        return (context.logits[:, 0] - context.logits[:, 1]).mean()

    def export_state(self) -> dict[str, object]:
        exported = super().export_state()
        exported["batch_trace"] = list(self.batch_trace)
        return exported


class GapCollector(StatisticCollector):
    def __init__(self, name: str = "gap") -> None:
        super().__init__(name)

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        return (context.logits[:, 0] - context.logits[:, 1]).mean()


class LossScaleGradientCollector(StatisticCollector):
    def __init__(self) -> None:
        super().__init__("test_loss_scale_grad")
        self.test_batch_requires_grad: list[bool] = []
        self.test_grad_values: list[float] = []

    def on_run_start(self, *, num_epochs: int) -> None:
        super().on_run_start(num_epochs=num_epochs)
        self.test_batch_requires_grad = []
        self.test_grad_values = []

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        if context.phase != "test":
            return context.loss.detach()

        gradient = torch.autograd.grad(context.loss, context.model.scale)[0]
        self.test_batch_requires_grad.append(bool(context.loss.requires_grad))
        self.test_grad_values.append(float(gradient.detach().item()))
        return gradient.abs()


def make_loader(*, batch_size: int = 2) -> DataLoader:
    images = torch.tensor(
        [
            [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            [[0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8]],
            [[0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    dataset = TensorImageDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def make_runner(
    *,
    optimizer: bool = True,
    loss_collector: CrossEntropyLossCollector | None = None,
    metric_collectors: list[StatisticCollector] | None = None,
) -> tuple[ImageClassifierRunner, MeanLogitModel]:
    model = MeanLogitModel()
    optimizer_obj = torch.optim.SGD(model.parameters(), lr=0.1) if optimizer else None
    runner = ImageClassifierRunner(
        model=model,
        loss_collector=loss_collector,
        optimizer=optimizer_obj,
        metric_collectors=metric_collectors,
    )
    return runner, model


def test_accuracy_from_logits_matches_argmax_fraction() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5], [0.7, 0.2]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)

    accuracy = accuracy_from_logits(logits, labels)

    assert accuracy == pytest.approx(2.0 / 3.0)


def test_runner_defaults_to_cross_entropy_loss() -> None:
    runner, _ = make_runner()

    assert isinstance(runner.loss_collector.loss_fn, nn.CrossEntropyLoss)
    assert isinstance(runner.loss_collector, CrossEntropyLossCollector)


def test_collectors_compute_sample_weighted_epoch_values() -> None:
    loss_collector = CrossEntropyLossCollector()
    accuracy_collector = AccuracyCollector()
    model = MeanLogitModel()

    loss_collector.on_run_start(num_epochs=1)
    accuracy_collector.on_run_start(num_epochs=1)
    loss_collector.on_epoch_start(phase="train", epoch=1)
    accuracy_collector.on_epoch_start(phase="train", epoch=1)

    batch_one = StatisticBatchContext(
        phase="train",
        epoch=1,
        batch_size=2,
        logits=torch.tensor([[3.0, 1.0], [0.2, 0.8]], dtype=torch.float32),
        labels=torch.tensor([0, 1], dtype=torch.long),
        loss=torch.tensor(0.5, dtype=torch.float32),
        model=model,
        optimizer=None,
    )
    batch_two = StatisticBatchContext(
        phase="train",
        epoch=1,
        batch_size=1,
        logits=torch.tensor([[0.4, 0.6]], dtype=torch.float32),
        labels=torch.tensor([0], dtype=torch.long),
        loss=torch.tensor(1.5, dtype=torch.float32),
        model=model,
        optimizer=None,
    )

    loss_collector.on_batch_end(batch_one)
    loss_collector.on_batch_end(batch_two)
    accuracy_collector.on_batch_end(batch_one)
    accuracy_collector.on_batch_end(batch_two)

    assert loss_collector.on_epoch_end(phase="train", epoch=1) == pytest.approx((0.5 * 2 + 1.5) / 3.0)
    assert accuracy_collector.on_epoch_end(phase="train", epoch=1) == pytest.approx((1.0 * 2 + 0.0) / 3.0)


def test_runner_rejects_invalid_batch_shapes() -> None:
    runner, _ = make_runner()

    with pytest.raises(ValueError, match=r"\[B, X, Y\]"):
        runner.run_forward_pass(torch.zeros(2, 1, 4, 4))

    with pytest.raises(ValueError, match="square"):
        runner.run_forward_pass(torch.zeros(2, 4, 8))

    with pytest.raises(ValueError, match="power of two"):
        runner.run_forward_pass(torch.zeros(2, 6, 6))

    with pytest.raises(ValueError, match=r"\[B\]"):
        runner.run_backward_pass(torch.zeros(2, 4, 4), torch.zeros(2, 1, dtype=torch.long))

    with pytest.raises(ValueError, match="matching batch dimensions"):
        runner.run_backward_pass(torch.zeros(2, 4, 4), torch.zeros(3, dtype=torch.long))


def test_runner_forward_backward_and_training_step_work() -> None:
    runner, model = make_runner()
    images = torch.full((2, 4, 4), 0.25, dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    logits = runner.run_forward_pass(images)
    before_step = model.scale.detach().clone()
    backward_logits, loss = runner.run_backward_pass(images, labels)
    assert logits.shape == (2, 2)
    assert backward_logits.shape == (2, 2)
    assert loss.ndim == 0
    assert model.scale.grad is not None

    training_logits, training_loss = runner.run_training_step(images, labels)

    assert training_logits.shape == (2, 2)
    assert training_loss.ndim == 0
    assert not torch.equal(before_step, model.scale.detach())


def test_evaluate_loader_uses_sample_weighted_loss_and_metrics() -> None:
    runner, model = make_runner(metric_collectors=[GapCollector()])
    loader = make_loader(batch_size=2)
    dataset = loader.dataset
    images = dataset.images
    labels = dataset.labels

    summary = runner.evaluate_loader(loader)
    logits = model(images)
    expected_loss = float(runner.loss_collector.loss_fn(logits, labels).item())
    expected_accuracy = accuracy_from_logits(logits, labels)
    expected_gap = float((logits[:, 0] - logits[:, 1]).mean().item())

    assert summary.loss == pytest.approx(expected_loss)
    assert summary.metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert summary.metrics["gap"] == pytest.approx(expected_gap)


def test_evaluate_loader_default_no_grad_rejects_gradient_based_collectors() -> None:
    collector = LossScaleGradientCollector()
    runner, _ = make_runner()
    runner = ImageClassifierRunner(
        model=runner.model,
        metric_collectors=[collector],
    )

    with pytest.raises(RuntimeError, match="does not require grad"):
        runner.evaluate_loader(make_loader(batch_size=2))


def test_evaluate_loader_requires_grad_preserves_graph_for_collectors() -> None:
    collector = LossScaleGradientCollector()
    runner, model = make_runner()
    runner = ImageClassifierRunner(
        model=model,
        metric_collectors=[collector],
    )

    summary = runner.evaluate_loader(make_loader(batch_size=2), requires_grad=True)

    assert "test_loss_scale_grad" in summary.metrics
    assert collector.test_batch_requires_grad == [True, True]
    assert collector.test_grad_values
    assert model.scale.grad is None


def test_runner_rejects_metric_collectors_using_reserved_loss_name() -> None:
    with pytest.raises(ValueError, match="reserved name 'loss'"):
        ImageClassifierRunner(
            model=MeanLogitModel(),
            metric_collectors=[GapCollector(name="loss")],
        )

    with pytest.raises(ValueError, match="Duplicate collector name"):
        ImageClassifierRunner(
            model=MeanLogitModel(),
            metric_collectors=[GapCollector(name="dup"), GapCollector(name="dup")],
        )


def test_runner_rejects_legacy_loss_fn_and_metrics_arguments() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'loss_fn'"):
        ImageClassifierRunner(model=MeanLogitModel(), loss_fn=nn.CrossEntropyLoss())

    with pytest.raises(TypeError, match="unexpected keyword argument 'metrics'"):
        ImageClassifierRunner(model=MeanLogitModel(), metrics={"gap": accuracy_from_logits})


def test_training_methods_require_optimizer() -> None:
    runner, _ = make_runner(optimizer=False)
    images = torch.full((2, 4, 4), 0.25, dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = make_loader(batch_size=2)

    with pytest.raises(RuntimeError, match="run_training_step requires an optimizer"):
        runner.run_training_step(images, labels)

    with pytest.raises(RuntimeError, match="train_epoch requires an optimizer"):
        runner.train_epoch(loader)

    with pytest.raises(RuntimeError, match="fit requires an optimizer"):
        runner.fit(loader, loader, num_epochs=1)


def test_fit_records_epoch_level_history() -> None:
    runner, _ = make_runner()
    train_loader = make_loader(batch_size=2)
    test_loader = make_loader(batch_size=3)

    history = runner.fit(train_loader, test_loader, num_epochs=2)

    assert history.train_epoch == [1, 2]
    assert history.test_epoch == [2]
    assert len(history.train_loss) == 2
    assert len(history.test_loss) == 1
    assert set(history.train_metrics) == {"accuracy"}
    assert set(history.test_metrics) == {"accuracy"}
    assert len(history.train_metrics["accuracy"]) == 2
    assert len(history.test_metrics["accuracy"]) == 1
    assert runner.last_train_loader is train_loader
    assert runner.last_test_loader is test_loader
    assert runner.last_training_history == history


def test_fit_records_sparse_test_history_and_always_evaluates_final_epoch() -> None:
    runner, _ = make_runner()

    history = runner.fit(
        make_loader(batch_size=2),
        make_loader(batch_size=3),
        num_epochs=12,
        test_evaluation_interval_epochs=10,
    )

    assert history.train_epoch == list(range(1, 13))
    assert history.test_epoch == [10, 12]
    assert len(history.train_loss) == 12
    assert len(history.test_loss) == 2
    assert len(history.train_metrics["accuracy"]) == 12
    assert len(history.test_metrics["accuracy"]) == 2


def test_runner_records_metric_collector_history_and_export_state() -> None:
    collector = MeanConfidenceCollector()
    runner, _ = make_runner()
    runner = ImageClassifierRunner(
        model=runner.model,
        optimizer=runner.optimizer,
        metric_collectors=[collector],
    )
    history = runner.fit(make_loader(batch_size=2), make_loader(batch_size=3), num_epochs=2)

    assert "accuracy" in history.train_metrics
    assert "confidence_gap" in history.train_metrics
    assert len(history.train_metrics["confidence_gap"]) == 2
    collector_states = runner.export_collector_states()
    assert collector_states["confidence_gap"]["history"]["train"] == history.train_metrics["confidence_gap"]
    assert collector_states["confidence_gap"]["batch_trace"]


def test_fit_test_requires_grad_enables_gradient_based_test_collectors() -> None:
    collector = LossScaleGradientCollector()
    runner, model = make_runner()
    runner = ImageClassifierRunner(
        model=model,
        optimizer=runner.optimizer,
        metric_collectors=[collector],
    )

    history = runner.fit(
        make_loader(batch_size=2),
        make_loader(batch_size=3),
        num_epochs=1,
        test_requires_grad=True,
    )

    assert "test_loss_scale_grad" in history.train_metrics
    assert "test_loss_scale_grad" in history.test_metrics
    assert collector.test_batch_requires_grad == [True]
    assert collector.test_grad_values
    assert model.scale.grad is not None


def test_training_history_rejects_legacy_epoch_argument() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'epoch'"):
        TrainingHistory(epoch=[1], train_loss=[1.0], test_loss=[1.0])


def test_runner_integrates_with_classical_model() -> None:
    torch.manual_seed(0)
    model = ClassicalMLP(image_size=4, num_classes=2)
    runner = ImageClassifierRunner(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    )
    images = torch.rand(4, 4, 4)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorImageDataset(images, labels), batch_size=2, shuffle=False)

    summary = runner.train_epoch(loader)

    assert isinstance(summary.loss, float)
    assert "accuracy" in summary.metrics


def test_runner_integrates_with_pcsqcnn() -> None:
    torch.manual_seed(0)
    with pytest.warns(UserWarning):
        model = PCSQCNN(
            image_size=4,
            num_classes=2,
            feature_qubits=1,
            quantum_layers=1,
        )
    runner = ImageClassifierRunner(
        model=model,
    )
    images = torch.rand(2, 4, 4)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorImageDataset(images, labels), batch_size=1, shuffle=False)

    summary = runner.evaluate_loader(loader)

    assert isinstance(summary.loss, float)
    assert "accuracy" in summary.metrics
