from dataclasses import replace
from pathlib import Path
import warnings

import pytest
import torch

from qcnn import (
    AutoTrainingConfig,
    EvaluationSummary,
    MnistDatasetConfig,
    ModelSpec,
    OptimizerConfig,
    OutputConfig,
    SeedConfig,
    StatisticCollector,
    TensorImageDataset,
    TrainingConfig,
    TrainingHistory,
    load_auto_training_run,
    load_model_checkpoint,
    run_mnist_auto_training,
)
from qcnn import automation as automation_module
from qcnn.data import PreparedMnistSplits
from qcnn.model_stats import format_trainable_parameter_stats_line
from qcnn.script_tasks import run_manifest_tasks
from qcnn.statistics import StatisticBatchContext
from tests.testing_models import CustomTinyClassifier


class MarginCollector(StatisticCollector):
    def __init__(self) -> None:
        super().__init__("margin_gap")
        self.max_gap = float("-inf")

    def on_run_start(self, *, num_epochs: int) -> None:
        super().on_run_start(num_epochs=num_epochs)
        self.max_gap = float("-inf")

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        gap = (context.logits[:, 0] - context.logits[:, 1]).mean()
        self.max_gap = max(self.max_gap, float(gap.detach().item()))
        return gap

    def export_state(self) -> dict[str, object]:
        exported = super().export_state()
        exported["max_gap"] = self.max_gap
        return exported


class LossScaleGradientCollector(StatisticCollector):
    def __init__(self) -> None:
        super().__init__("test_loss_scale_grad")
        self.observed_test_requires_grad = False

    def on_run_start(self, *, num_epochs: int) -> None:
        super().on_run_start(num_epochs=num_epochs)
        self.observed_test_requires_grad = False

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        if context.phase != "test":
            return context.loss.detach()

        gradient = torch.autograd.grad(context.loss, context.model.classifier.weight)[0]
        self.observed_test_requires_grad = self.observed_test_requires_grad or bool(context.loss.requires_grad)
        return gradient.abs().mean()

    def export_state(self) -> dict[str, object]:
        exported = super().export_state()
        exported["observed_test_requires_grad"] = self.observed_test_requires_grad
        return exported


class BadExportCollector(StatisticCollector):
    def __init__(self) -> None:
        super().__init__("bad_export")

    def compute_batch_value(self, context: StatisticBatchContext) -> torch.Tensor:
        return context.logits.mean()

    def export_state(self) -> dict[str, object]:
        return {"history": self.history(), "bad": lambda value: value}


def fake_prepare_mnist_splits(
    *,
    root: str | Path,
    samples_per_class: int | None,
    image_size: int,
    scaled_image_size: int | None,
    max_offset: int,
    seed: int,
    download: bool,
) -> PreparedMnistSplits:
    del root, download
    train_images = torch.tensor(
        [
            [[0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3]],
            [[0.9, 0.9, 0.8, 0.8], [0.9, 0.9, 0.8, 0.8], [0.7, 0.7, 0.6, 0.6], [0.7, 0.7, 0.6, 0.6]],
            [[0.1, 0.2, 0.1, 0.2], [0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4], [0.3, 0.4, 0.3, 0.4]],
            [[0.8, 0.7, 0.8, 0.7], [0.8, 0.7, 0.8, 0.7], [0.6, 0.5, 0.6, 0.5], [0.6, 0.5, 0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    test_images = torch.tensor(
        [
            [[0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            [[0.95, 0.95, 0.95, 0.95], [0.95, 0.95, 0.95, 0.95], [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]],
        ],
        dtype=torch.float32,
    )
    if image_size != 4:
        raise AssertionError(f"Expected image_size=4 in the fake dataset, got {image_size}.")
    if scaled_image_size not in (None, 4):
        raise AssertionError(
            f"Expected scaled_image_size to resolve to legacy 4x4 preprocessing, got {scaled_image_size}."
        )
    if max_offset != 0:
        raise AssertionError(f"Expected max_offset=0 in the fake dataset, got {max_offset}.")

    return PreparedMnistSplits(
        train=TensorImageDataset(
            train_images,
            torch.tensor([0, 1, 0, 1], dtype=torch.long),
            metadata={
                "dataset_name": "MNIST",
                "split": "train",
                "image_size": image_size,
                "scaled_image_size": image_size,
                "max_offset": 0,
                "samples_per_class": samples_per_class,
                "seed": seed,
            },
        ),
        test=TensorImageDataset(
            test_images,
            torch.tensor([0, 1], dtype=torch.long),
            metadata={
                "dataset_name": "MNIST",
                "split": "test",
                "image_size": image_size,
                "scaled_image_size": image_size,
                "max_offset": 0,
                "test_split": "standard",
            },
        ),
    )


def make_config(
    tmp_path: Path,
    *,
    model_kind: str = "classical_mlp",
    base_seed: int = 0,
    seed_count: int = 1,
    snapshot_epochs: tuple[int, ...] = (1,),
    num_epochs: int = 2,
    test_requires_grad: bool = False,
    test_evaluation_interval_epochs: int = 10,
) -> AutoTrainingConfig:
    return AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=tmp_path / "data",
            samples_per_class=2,
            image_size=4,
            train_batch_size=2,
            test_batch_size=3,
            download=False,
        ),
        model=make_model_spec(model_kind),
        optimizer=OptimizerConfig(
            kind="adam",
            learning_rate=1e-2,
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            snapshot_epochs=snapshot_epochs,
            device="cpu",
            torch_matmul_precision=None,
            test_requires_grad=test_requires_grad,
            test_evaluation_interval_epochs=test_evaluation_interval_epochs,
        ),
        seeds=SeedConfig(
            base_seed=base_seed,
            seed_count=seed_count,
        ),
        output=OutputConfig(
            root=tmp_path / "outputs",
            use_timestamp_dir=False,
        ),
    )


def make_model_spec(model_kind: str) -> ModelSpec:
    if model_kind == "pcsqcnn":
        return ModelSpec(
            module="qcnn.hybrid",
            class_name="PCSQCNN",
            constructor_kwargs={
                "image_size": 4,
                "num_classes": 2,
                "feature_qubits": 1,
                "quantum_layers": 1,
            },
        )
    if model_kind == "classical_cnn":
        return ModelSpec(
            module="qcnn.classic",
            class_name="ClassicalCNN",
            constructor_kwargs={
                "image_size": 4,
                "num_classes": 2,
            },
        )
    if model_kind == "classical_mlp":
        return ModelSpec(
            module="qcnn.classic",
            class_name="ClassicalMLP",
            constructor_kwargs={
                "image_size": 4,
                "num_classes": 2,
            },
        )
    if model_kind == "custom_tiny":
        return ModelSpec(
            module="tests.testing_models",
            class_name="CustomTinyClassifier",
            constructor_kwargs={
                "image_size": 4,
                "num_classes": 2,
            },
        )
    raise ValueError(f"Unsupported test model kind {model_kind!r}.")


def make_named_config(
    tmp_path: Path,
    *,
    directory_name: str,
    **kwargs,
) -> AutoTrainingConfig:
    base_config = make_config(tmp_path, **kwargs)
    return replace(
        base_config,
        output=replace(base_config.output, directory_name=directory_name),
    )


def write_completed_output(config: AutoTrainingConfig) -> None:
    output_directory = automation_module.resolve_auto_training_output_directory(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "manifest.json").write_text("{}", encoding="utf-8")


def expected_seeds(config: AutoTrainingConfig) -> list[int]:
    return list(range(config.seeds.base_seed, config.seeds.base_seed + config.seeds.seed_count))


def expected_epoch_range(config: AutoTrainingConfig) -> list[int]:
    return list(range(1, config.training.num_epochs + 1))


def test_run_mnist_auto_training_single_seed_creates_expected_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    config = make_config(tmp_path, snapshot_epochs=(1, 2))
    result = run_mnist_auto_training(config)
    seed_run = result.runs[0]

    assert result.output_directory.name == "run"
    assert result.manifest_path.is_file()
    assert seed_run.artifacts.result_path.name == "result_seed0.pt"
    assert seed_run.artifacts.final_checkpoint_path.name == "checkpoint_final_seed0.pt"
    assert set(path.name for path in seed_run.artifacts.snapshot_paths.values()) == {
        "checkpoint_epoch1_seed0.pt",
        "checkpoint_epoch2_seed0.pt",
    }
    payload = torch.load(seed_run.artifacts.result_path)
    checkpoint_payload = torch.load(seed_run.artifacts.final_checkpoint_path)
    assert isinstance(payload["training_history"], TrainingHistory)
    assert seed_run.train_loader.batch_size == config.dataset.train_batch_size
    assert seed_run.test_loader.batch_size == config.dataset.test_batch_size
    assert payload["loss_name"] == "CrossEntropyLoss"
    assert payload["parameter_stats_line"] == format_trainable_parameter_stats_line(seed_run.runner.model)
    assert payload["files"]["checkpoint_final"] == "checkpoint_final_seed0.pt"
    assert payload["final_summaries"]["test"]["loss"] == pytest.approx(seed_run.history.test_loss[-1])
    assert payload["resolved_config"]["dataset"]["train_batch_size"] == config.dataset.train_batch_size
    assert payload["resolved_config"]["dataset"]["test_batch_size"] == config.dataset.test_batch_size
    assert payload["resolved_config"]["dataset"]["scaled_image_size"] == (
        config.dataset.image_size if config.dataset.scaled_image_size is None else config.dataset.scaled_image_size
    )
    assert payload["resolved_config"]["dataset"]["max_offset"] == config.dataset.max_offset
    assert checkpoint_payload["checkpoint_type"] == "qcnn_model_checkpoint"
    assert checkpoint_payload["model_spec"]["class_name"] == config.model.class_name
    assert "model_state" in checkpoint_payload


def test_run_mnist_auto_training_multi_seed_uses_flat_seed_suffixed_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    config = make_config(tmp_path, base_seed=4, seed_count=3)
    result = run_mnist_auto_training(config)

    assert result.seeds == expected_seeds(config)
    assert not any(path.is_dir() for path in result.output_directory.iterdir())
    for seed in result.seeds:
        assert (result.output_directory / f"result_seed{seed}.pt").is_file()
        assert (result.output_directory / f"checkpoint_final_seed{seed}.pt").is_file()
        for snapshot_epoch in config.training.snapshot_epochs:
            assert (result.output_directory / f"checkpoint_epoch{snapshot_epoch}_seed{seed}.pt").is_file()


def test_run_mnist_auto_training_reports_seed_lifecycle_and_epoch_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    seed_start_calls: list[tuple[int, int, int]] = []
    seed_end_calls: list[tuple[int, int, int]] = []
    progress_ranges: list[list[int]] = []

    def progress_factory(epoch_range: range) -> range:
        progress_ranges.append(list(epoch_range))
        return epoch_range

    config = make_config(tmp_path, base_seed=3, seed_count=2, num_epochs=3)
    run_mnist_auto_training(
        config,
        progress_factory=progress_factory,
        seed_start_callback=lambda seed, seed_index, seed_count: seed_start_calls.append(
            (seed, seed_index, seed_count)
        ),
        seed_end_callback=lambda seed, seed_index, seed_count: seed_end_calls.append(
            (seed, seed_index, seed_count)
        ),
    )

    expected_lifecycle_calls = [
        (seed, index, config.seeds.seed_count)
        for index, seed in enumerate(expected_seeds(config), start=1)
    ]
    assert seed_start_calls == expected_lifecycle_calls
    assert seed_end_calls == expected_lifecycle_calls
    assert progress_ranges == [expected_epoch_range(config) for _ in expected_seeds(config)]


def test_run_mnist_auto_training_reports_seed_epoch_end_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    epoch_end_calls: list[dict[str, object]] = []

    def on_seed_epoch_end(
        seed: int,
        epoch: int,
        train_summary,
        test_summary,
        history: TrainingHistory,
    ) -> None:
        epoch_end_calls.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_summary.loss,
                "train_accuracy": train_summary.metrics["accuracy"],
                "test_loss": None if test_summary is None else test_summary.loss,
                "test_accuracy": None if test_summary is None else test_summary.metrics["accuracy"],
                "train_epoch": list(history.train_epoch),
                "test_epoch": list(history.test_epoch),
            }
        )

    config = make_config(
        tmp_path,
        base_seed=3,
        seed_count=2,
        num_epochs=3,
        test_evaluation_interval_epochs=2,
    )
    run_mnist_auto_training(config, seed_epoch_end_callback=on_seed_epoch_end)

    assert [(call["seed"], call["epoch"]) for call in epoch_end_calls] == [
        (seed, epoch)
        for seed in expected_seeds(config)
        for epoch in expected_epoch_range(config)
    ]

    for call in epoch_end_calls:
        current_epoch = call["epoch"]
        assert call["train_epoch"] == list(range(1, current_epoch + 1))
        expected_test_epochs = [
            epoch
            for epoch in range(1, current_epoch + 1)
            if epoch % config.training.test_evaluation_interval_epochs == 0 or epoch == config.training.num_epochs
        ]
        assert call["test_epoch"] == expected_test_epochs
        if current_epoch in expected_test_epochs:
            assert call["test_loss"] is not None
            assert call["test_accuracy"] is not None
        else:
            assert call["test_loss"] is None
            assert call["test_accuracy"] is None
        assert 0.0 <= call["train_accuracy"] <= 1.0


def test_run_mnist_auto_training_allows_full_train_split_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    config = make_config(tmp_path)
    config = AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=config.dataset.root,
            samples_per_class=None,
            image_size=config.dataset.image_size,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
            download=config.dataset.download,
        ),
        model=config.model,
        optimizer=config.optimizer,
        training=config.training,
        seeds=config.seeds,
        output=config.output,
    )

    result = run_mnist_auto_training(config)

    payload = torch.load(result.runs[0].artifacts.result_path)
    assert payload["resolved_config"]["dataset"]["samples_per_class"] == config.dataset.samples_per_class
    assert result.runs[0].train_loader.dataset.metadata["samples_per_class"] == config.dataset.samples_per_class


def test_run_mnist_auto_training_passes_test_requires_grad_to_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    collector = LossScaleGradientCollector()

    result = run_mnist_auto_training(
        make_config(tmp_path, model_kind="pcsqcnn", test_requires_grad=True),
        collectors=[collector],
    )

    payload = torch.load(result.runs[0].artifacts.result_path)
    assert collector.observed_test_requires_grad is True
    assert payload["collector_states"]["test_loss_scale_grad"]["observed_test_requires_grad"] is True


def test_run_mnist_auto_training_accepts_custom_importable_model_specs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    result = run_mnist_auto_training(
        make_config(tmp_path, model_kind="custom_tiny"),
    )

    loaded_checkpoint = load_model_checkpoint(result.runs[0].artifacts.final_checkpoint_path)
    assert isinstance(loaded_checkpoint.model, CustomTinyClassifier)


def test_run_mnist_auto_training_without_timestamp_avoids_directory_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    first = run_mnist_auto_training(make_config(tmp_path))
    second = run_mnist_auto_training(make_config(tmp_path))

    assert first.output_directory.name == "run"
    assert second.output_directory.name == "run_1"


def test_run_mnist_auto_training_creates_nested_explicit_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    base_config = make_config(tmp_path)
    nested_config = AutoTrainingConfig(
        dataset=base_config.dataset,
        model=base_config.model,
        optimizer=base_config.optimizer,
        training=base_config.training,
        seeds=base_config.seeds,
        output=OutputConfig(
            root=tmp_path / "outputs",
            directory_name="sweeps/fq1_ql1_u2by25pi",
            use_timestamp_dir=False,
        ),
    )

    result = run_mnist_auto_training(nested_config)

    assert result.output_directory == tmp_path / "outputs" / "sweeps" / "fq1_ql1_u2by25pi"
    assert result.output_directory.parent == tmp_path / "outputs" / "sweeps"
    assert result.manifest_path.is_file()
    assert (result.output_directory / "result_seed0.pt").is_file()
    assert (result.output_directory / "checkpoint_final_seed0.pt").is_file()


def test_run_mnist_auto_training_pcsqcnn_writes_loadable_checkpoint_and_collector_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        result = run_mnist_auto_training(
            make_config(tmp_path, model_kind="pcsqcnn"),
            collectors=[MarginCollector()],
        )

    seed_run = result.runs[0]
    with pytest.warns(UserWarning, match="1/sqrt\\(XY\\)"):
        loaded_run = load_auto_training_run(result.output_directory, seed_run.seed)
    assert loaded_run.training_history == seed_run.history
    assert loaded_run.model.__class__.__name__ == "PCSQCNN"

    payload = torch.load(seed_run.artifacts.result_path)
    assert payload["parameter_stats_line"] == format_trainable_parameter_stats_line(seed_run.runner.model)
    assert "margin_gap" in payload["collector_states"]
    assert payload["collector_states"]["margin_gap"]["history"]["train"] == seed_run.history.train_metrics["margin_gap"]
    assert payload["final_summaries"]["train"]["loss"] == pytest.approx(seed_run.history.train_loss[-1])
    assert payload["final_summaries"]["test"]["loss"] == pytest.approx(seed_run.history.test_loss[-1])
    assert payload["files"].get("pcsqcnn_artifact") is None


def test_run_mnist_auto_training_rejects_non_serializable_collector_exports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(automation_module, "prepare_mnist_splits", fake_prepare_mnist_splits)

    with pytest.raises(ValueError, match="not torch.save-serializable"):
        run_mnist_auto_training(
            make_config(tmp_path),
            collectors=[BadExportCollector()],
        )


def test_loader_settings_for_cuda_disable_workers_and_pin_memory() -> None:
    assert automation_module._loader_settings_for_device(
        device=torch.device("cuda"),
        num_workers=4,
        pin_memory=True,
    ) == (0, False)
    assert automation_module._loader_settings_for_device(
        device=torch.device("cpu"),
        num_workers=4,
        pin_memory=True,
    ) == (4, True)


def test_resolve_auto_training_output_directory_requires_explicit_directory_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output.directory_name"):
        automation_module.resolve_auto_training_output_directory(make_config(tmp_path))


def test_resolve_auto_training_output_directory_joins_nested_directory_name(tmp_path: Path) -> None:
    config = make_named_config(
        tmp_path,
        directory_name="sweeps/fq1_ql1_u2by25pi",
    )

    assert automation_module.resolve_auto_training_output_directory(config) == (
        tmp_path / "outputs" / "sweeps" / "fq1_ql1_u2by25pi"
    ).resolve()


def test_run_auto_training_manifest_tasks_builds_tasks_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_a = make_named_config(tmp_path, directory_name="family/a")
    config_b = make_named_config(tmp_path, directory_name="family/b")
    captured: dict[str, object] = {}

    def fake_run_manifest_tasks(tasks, *, rebuild: bool) -> None:
        captured["tasks"] = tuple(tasks)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(automation_module, "run_manifest_tasks", fake_run_manifest_tasks)

    automation_module.run_auto_training_manifest_tasks((config_a, config_b), rebuild=True)

    assert captured["rebuild"] is True
    assert [task.output_directory for task in captured["tasks"]] == [
        (tmp_path / "outputs" / "family" / "a").resolve(),
        (tmp_path / "outputs" / "family" / "b").resolve(),
    ]


def test_build_auto_training_manifest_task_updates_secondary_progress_with_latest_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTaskContext:
        def __init__(self) -> None:
            self.primary_descriptions: list[str] = []
            self.secondary_descriptions: list[str] = []
            self.status_lines: list[str] = []
            self.primary_completed: list[int] = []
            self.secondary_advance_steps = 0
            self.secondary_hidden_count = 0
            self.status_hidden_count = 0

        def show_primary_progress(self, *, description: str, total: int, completed: int = 0) -> None:
            del total, completed
            self.primary_descriptions.append(description)

        def show_secondary_progress(self, *, description: str, total: int, completed: int = 0) -> None:
            del total, completed
            self.secondary_descriptions.append(description)

        def update_primary_progress(self, *, description=None, total=None, completed=None) -> None:
            del total
            if description is not None:
                self.primary_descriptions.append(description)
            if completed is not None:
                self.primary_completed.append(completed)

        def update_secondary_progress(self, *, description=None, total=None, completed=None) -> None:
            del total, completed
            if description is not None:
                self.secondary_descriptions.append(description)

        def advance_secondary_progress(self, steps: int = 1) -> None:
            self.secondary_advance_steps += steps

        def hide_secondary_progress(self) -> None:
            self.secondary_hidden_count += 1

        def show_status_line(self, text: str) -> None:
            self.status_lines.append(text)

        def update_status_line(self, text: str) -> None:
            self.status_lines.append(text)

        def hide_status_line(self) -> None:
            self.status_hidden_count += 1

    def build_history() -> TrainingHistory:
        return TrainingHistory(
            train_epoch=[],
            test_epoch=[],
            train_loss=[],
            test_loss=[],
            train_metrics={"accuracy": []},
            test_metrics={"accuracy": []},
        )

    first_seed_summaries = [
        (EvaluationSummary(loss=0.5, metrics={"accuracy": 0.75}), None),
        (
            EvaluationSummary(loss=0.4, metrics={"accuracy": 0.80}),
            EvaluationSummary(loss=0.3, metrics={"accuracy": 0.90}),
        ),
        (EvaluationSummary(loss=0.35, metrics={"accuracy": 0.825}), None),
    ]
    second_seed_summaries = [
        (EvaluationSummary(loss=0.24, metrics={"accuracy": 0.93}), None),
        (EvaluationSummary(loss=0.19, metrics={"accuracy": 0.95}), None),
        (
            EvaluationSummary(loss=0.15, metrics={"accuracy": 0.97}),
            EvaluationSummary(loss=0.11, metrics={"accuracy": 0.98}),
        ),
    ]

    def fake_run_mnist_auto_training(config, **kwargs) -> None:
        assert config is training_config
        seed_start_callback = kwargs["seed_start_callback"]
        seed_end_callback = kwargs["seed_end_callback"]
        seed_epoch_end_callback = kwargs["seed_epoch_end_callback"]
        progress_factory = kwargs["progress_factory"]

        first_seed_history = build_history()
        seed_start_callback(7, 1, 2)
        for epoch in progress_factory(range(1, 4)):
            train_summary, test_summary = first_seed_summaries[epoch - 1]
            first_seed_history.train_epoch.append(epoch)
            first_seed_history.train_loss.append(train_summary.loss)
            first_seed_history.train_metrics["accuracy"].append(train_summary.metrics["accuracy"])
            if test_summary is not None:
                first_seed_history.test_epoch.append(epoch)
                first_seed_history.test_loss.append(test_summary.loss)
                first_seed_history.test_metrics["accuracy"].append(test_summary.metrics["accuracy"])
            seed_epoch_end_callback(7, epoch, train_summary, test_summary, first_seed_history)
        seed_end_callback(7, 1, 2)

        second_seed_history = build_history()
        seed_start_callback(8, 2, 2)
        for epoch in progress_factory(range(1, 4)):
            train_summary, test_summary = second_seed_summaries[epoch - 1]
            second_seed_history.train_epoch.append(epoch)
            second_seed_history.train_loss.append(train_summary.loss)
            second_seed_history.train_metrics["accuracy"].append(train_summary.metrics["accuracy"])
            if test_summary is not None:
                second_seed_history.test_epoch.append(epoch)
                second_seed_history.test_loss.append(test_summary.loss)
                second_seed_history.test_metrics["accuracy"].append(test_summary.metrics["accuracy"])
            seed_epoch_end_callback(8, epoch, train_summary, test_summary, second_seed_history)
        seed_end_callback(8, 2, 2)

    monkeypatch.setattr(automation_module, "run_mnist_auto_training", fake_run_mnist_auto_training)
    perf_counter_values = iter([100.0, 110.0, 130.0, 160.0, 165.0, 175.0, 190.0])
    monkeypatch.setattr(automation_module.time, "perf_counter", lambda: next(perf_counter_values))
    training_config = make_named_config(
        tmp_path,
        directory_name="fixed/classical_mlp_spc20",
        num_epochs=3,
        seed_count=2,
    )
    task_spec = automation_module.build_auto_training_manifest_task(training_config)
    task_context = FakeTaskContext()

    task_spec.run(task_context)

    placeholder = automation_module._format_epoch_status_line()
    seed7_train_only = automation_module._format_epoch_status_line(
        train_summary=first_seed_summaries[0][0],
    )
    seed7_with_test = automation_module._format_epoch_status_line(
        train_summary=first_seed_summaries[1][0],
        test_summary=first_seed_summaries[1][1],
    )
    seed7_cached_test = automation_module._format_epoch_status_line(
        train_summary=first_seed_summaries[2][0],
        test_summary=first_seed_summaries[1][1],
    )
    seed8_final = automation_module._format_epoch_status_line(
        train_summary=second_seed_summaries[2][0],
        test_summary=second_seed_summaries[2][1],
    )

    assert task_context.primary_descriptions
    assert task_context.secondary_descriptions
    assert all(description.startswith("Seeds ETA") for description in task_context.primary_descriptions)
    assert all(description.startswith("Epochs ETA") for description in task_context.secondary_descriptions)
    assert task_context.status_lines[0] == placeholder
    assert seed7_train_only in task_context.status_lines
    assert seed7_with_test in task_context.status_lines
    assert seed7_cached_test in task_context.status_lines
    assert task_context.status_lines.count(placeholder) == training_config.seeds.seed_count
    assert task_context.status_lines[-1] == seed8_final
    assert task_context.primary_completed == list(range(1, training_config.seeds.seed_count + 1))
    assert task_context.secondary_advance_steps == training_config.training.num_epochs * training_config.seeds.seed_count
    assert task_context.secondary_hidden_count == training_config.seeds.seed_count
    assert task_context.status_hidden_count == training_config.seeds.seed_count


def test_build_auto_training_manifest_task_formats_duplicate_user_warnings_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    def fake_run_mnist_auto_training(config, **kwargs) -> None:
        del kwargs
        warnings.warn("duplicated warning", UserWarning)
        warnings.warn("duplicated warning", UserWarning)
        warnings.warn("another warning", UserWarning)
        write_completed_output(config)

    monkeypatch.setattr(automation_module, "run_mnist_auto_training", fake_run_mnist_auto_training)
    training_config = make_named_config(
        tmp_path,
        directory_name="fixed/classical_mlp_spc20",
    )
    task_spec = automation_module.build_auto_training_manifest_task(training_config)

    run_manifest_tasks((task_spec,))

    captured = capsys.readouterr()
    assert captured.out.count("Warning duplicated warning") == 1
    assert captured.out.count("Warning another warning") == 1
    assert "UserWarning" not in captured.out
