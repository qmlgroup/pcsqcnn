from __future__ import annotations

from pathlib import Path

import pytest

from tests.script_loading import load_run_script


prepare_full_mnist_classical_baselines = load_run_script(
    "prepare_full_mnist_classical_baselines_data_module",
    "prepare_full_mnist_classical_baselines_data.py",
)


def test_parse_args_accepts_rebuild_flag() -> None:
    args = prepare_full_mnist_classical_baselines.parse_args(["--rebuild"])
    assert args.rebuild is True


def test_training_defaults_accept_inherited_shared_kwargs(tmp_path: Path) -> None:
    baseline = prepare_full_mnist_classical_baselines.FullMnistClassicalBaselinesTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    defaults = prepare_full_mnist_classical_baselines.FullMnistClassicalBaselinesTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        optimizer_kind="sgd" if baseline.optimizer_kind != "sgd" else "adam",
        num_workers=baseline.num_workers + 1,
        pin_memory=not baseline.pin_memory,
        download=not baseline.download,
    )

    assert defaults.optimizer_kind != baseline.optimizer_kind
    assert defaults.num_workers != baseline.num_workers
    assert defaults.pin_memory != baseline.pin_memory
    assert defaults.download != baseline.download
    assert defaults.samples_per_class is None
    assert defaults.scaled_image_size == 32
    assert defaults.image_size == 32
    assert defaults.max_offset == 0
    assert defaults.num_epochs == 1000
    assert defaults.learning_rate == pytest.approx(0.005)


def test_iter_full_mnist_classical_baseline_configs_returns_exactly_two_runs(tmp_path: Path) -> None:
    defaults = prepare_full_mnist_classical_baselines.FullMnistClassicalBaselinesTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    configs = prepare_full_mnist_classical_baselines.iter_full_mnist_classical_baseline_configs(defaults)

    assert len(configs) == 2
    assert [config.model.class_name for config in configs] == ["ClassicalMLP", "ClassicalCNN"]
    assert [config.output.directory_name for config in configs] == [
        "full_mnist_classical_baselines/classical_mlp/32on32",
        "full_mnist_classical_baselines/classical_cnn/32on32",
    ]
    assert all(config.dataset.samples_per_class is None for config in configs)
    assert [config.dataset.scaled_image_size for config in configs] == [32, 32]
    assert [config.dataset.image_size for config in configs] == [32, 32]
    assert [config.dataset.max_offset for config in configs] == [0, 0]
    assert all(config.training.num_epochs == 1000 for config in configs)
    assert all(config.optimizer.learning_rate == pytest.approx(0.005) for config in configs)
    assert all(
        config.optimizer.learning_rate == pytest.approx(defaults.learning_rate)
        for config in configs
    )


def test_main_dispatches_only_full_mnist_classical_configs_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_full_mnist_classical_baselines.FullMnistClassicalBaselinesTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_full_mnist_classical_baselines,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_full_mnist_classical_baselines.main(defaults=defaults, argv=["--rebuild"])

    assert captured["rebuild"] is True
    assert len(captured["configs"]) == 2
