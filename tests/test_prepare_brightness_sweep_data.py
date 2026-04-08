from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from qcnn import article_training
from tests.script_loading import load_run_script


prepare_brightness_sweep_data = load_run_script(
    "prepare_brightness_sweep_data_module",
    "prepare_brightness_sweep_data.py",
)


def test_parse_args_accepts_rebuild_flag() -> None:
    args = prepare_brightness_sweep_data.parse_args(["--rebuild"])
    assert args.rebuild is True


def test_brightness_sweep_training_defaults_accept_inherited_shared_kwargs(tmp_path: Path) -> None:
    baseline = prepare_brightness_sweep_data.BrightnessSweepTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    defaults = prepare_brightness_sweep_data.BrightnessSweepTrainingDefaults(
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
    assert defaults.fixed_quantum_reduce_readout_to_feature_distribution is False
    assert defaults.test_evaluation_interval_epochs == 150


def test_iter_brightness_sweep_training_configs_returns_ordered_unique_runs(tmp_path: Path) -> None:
    defaults = prepare_brightness_sweep_data.BrightnessSweepTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        feature_qubits_options=(2,),
        quantum_layers_options=(1, 3),
        num_interior_points=2,
    )
    configs = prepare_brightness_sweep_data.iter_brightness_sweep_training_configs(defaults)
    brightness_pi_coefficients = article_training.generate_brightness_pi_coefficients(defaults)

    assert len(configs) == (
        len(defaults.feature_qubits_options)
        * len(defaults.quantum_layers_options)
        * len(brightness_pi_coefficients)
    )
    assert [config.output.directory_name for config in configs] == [
        article_training.build_brightness_sweep_directory_name(
            defaults,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
            brightness_stop_pi=brightness_stop_pi,
        )
        for feature_qubits in defaults.feature_qubits_options
        for quantum_layers in defaults.quantum_layers_options
        for brightness_stop_pi in brightness_pi_coefficients
    ]
    assert all(config.model.class_name == "PCSQCNN" for config in configs)
    assert all(config.optimizer.learning_rate == pytest.approx(defaults.hybrid_learning_rate) for config in configs)
    assert all(config.training.test_evaluation_interval_epochs == 150 for config in configs)
    assert all(
        config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is False
        for config in configs
    )


def test_main_dispatches_only_brightness_sweep_configs_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_brightness_sweep_data.BrightnessSweepTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        feature_qubits_options=(1,),
        quantum_layers_options=(1,),
        num_interior_points=1,
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_brightness_sweep_data,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_brightness_sweep_data.main(defaults=defaults, argv=["--rebuild"])

    assert captured["rebuild"] is True
    assert len(captured["configs"]) == 1
