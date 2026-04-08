from __future__ import annotations

from pathlib import Path

import pytest

from qcnn.article_training import build_figure_2_directory_name
from tests.script_loading import load_run_script


prepare_figure_2_data = load_run_script(
    "prepare_translated_mnist_baselines_data_module",
    "prepare_translated_mnist_baselines_data.py",
)


def test_parse_args_accepts_rebuild_flag() -> None:
    args = prepare_figure_2_data.parse_args(["--rebuild"])
    assert args.rebuild is True


def test_figure_2_training_defaults_accept_inherited_shared_kwargs(tmp_path: Path) -> None:
    baseline = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    defaults = prepare_figure_2_data.Figure2TrainingDefaults(
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


def test_iter_figure_2_training_configs_returns_multi_size_classical_and_fixed_quantum_runs(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    configs = prepare_figure_2_data.iter_figure_2_training_configs(defaults)

    assert len(configs) == 5
    assert [config.model.class_name for config in configs] == [
        "ClassicalMLP",
        "ClassicalMLP",
        "ClassicalCNN",
        "ClassicalCNN",
        "PCSQCNNNoQFT",
    ]
    assert [config.output.directory_name for config in configs] == [
        "classical_mlp/16on32",
        "classical_mlp/32on32",
        "classical_cnn/16on32",
        "classical_cnn/32on32",
        build_figure_2_directory_name("pcsqcnn_no_qft", samples_per_class=defaults.samples_per_class),
    ]
    assert all(config.dataset.samples_per_class == defaults.samples_per_class for config in configs)
    assert [config.dataset.scaled_image_size for config in configs] == [16, 32, 16, 32, 16]
    assert [config.dataset.max_offset for config in configs] == [8, 0, 8, 0, 8]
    assert configs[0].optimizer.learning_rate == pytest.approx(defaults.learning_rate)
    assert configs[4].optimizer.learning_rate == pytest.approx(defaults.hybrid_learning_rate)
    assert configs[4].model.constructor_kwargs["feature_qubits"] == 2
    assert configs[4].model.constructor_kwargs["quantum_layers"] == 3


def test_iter_figure_2_training_configs_rejects_mismatched_preprocessing_option_lengths(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        scaled_image_size_options=(16, 32),
        max_offset_options=(8,),
    )

    with pytest.raises(ValueError, match="same length"):
        prepare_figure_2_data.iter_figure_2_training_configs(defaults)


def test_main_dispatches_only_figure_2_configs_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_figure_2_data,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_figure_2_data.main(defaults=defaults, argv=["--rebuild"])

    assert captured["rebuild"] is True
    assert len(captured["configs"]) == 5
