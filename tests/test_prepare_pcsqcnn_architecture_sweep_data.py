from __future__ import annotations

from pathlib import Path

import pytest

from qcnn import article_training
from tests.script_loading import load_run_script


prepare_figure_5a_data = load_run_script(
    "prepare_pcsqcnn_architecture_sweep_data_module",
    "prepare_pcsqcnn_architecture_sweep_data.py",
)


def test_parse_args_accepts_rebuild_flag() -> None:
    args = prepare_figure_5a_data.parse_args(["--rebuild"])
    assert args.rebuild is True


def test_figure_5a_training_defaults_accept_inherited_shared_kwargs(tmp_path: Path) -> None:
    baseline = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
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


def test_iter_figure_5a_training_configs_returns_architecture_sweep_runs_in_legend_order(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    configs = prepare_figure_5a_data.iter_figure_5a_training_configs(defaults)

    assert len(configs) == 15
    assert [config.output.directory_name for config in configs] == [
        article_training.build_architecture_sweep_directory_name(
            defaults,
            feature_qubits=feature_qubits,
            quantum_layers=quantum_layers,
        )
        for quantum_layers in defaults.quantum_layers_options
        for feature_qubits in defaults.feature_qubits_options
    ]
    assert all(config.model.class_name == "PCSQCNN" for config in configs)
    assert all(config.optimizer.learning_rate == pytest.approx(defaults.hybrid_learning_rate) for config in configs)
    assert all(
        config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is False
        for config in configs
    )


def test_iter_figure_5a_training_configs_switches_to_full_readout_root_when_configured(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        feature_qubits_options=(1,),
        quantum_layers_options=(1,),
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )
    (config,) = prepare_figure_5a_data.iter_figure_5a_training_configs(defaults)

    assert config.output.directory_name == "pcsqcnn_architecture_sweep_full_readout/fq1_ql1"


def test_resolve_figure_5a_training_defaults_forces_full_readout(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        fixed_quantum_reduce_readout_to_feature_distribution=True,
    )

    inherited_defaults = prepare_figure_5a_data.resolve_figure_5a_training_defaults(defaults)

    assert inherited_defaults.fixed_quantum_reduce_readout_to_feature_distribution is False

    (full_config,) = prepare_figure_5a_data.iter_figure_5a_training_configs(
        prepare_figure_5a_data.Figure5aTrainingDefaults(
            data_root=tmp_path / "data",
            artifacts_root=tmp_path / "artifacts",
            feature_qubits_options=(1,),
            quantum_layers_options=(1,),
            fixed_quantum_reduce_readout_to_feature_distribution=True,
        )
    )

    assert full_config.output.directory_name == "pcsqcnn_architecture_sweep_full_readout/fq1_ql1"
    assert full_config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is False


def test_main_dispatches_only_figure_5a_configs_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        feature_qubits_options=(1,),
        quantum_layers_options=(1, 2),
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_figure_5a_data,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_figure_5a_data.main(defaults=defaults, argv=["--rebuild"])

    assert captured["rebuild"] is True
    assert len(captured["configs"]) == 2


def test_main_forces_full_readout_for_dispatched_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5a_data.Figure5aTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        feature_qubits_options=(1,),
        quantum_layers_options=(1,),
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_figure_5a_data,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_figure_5a_data.main(defaults=defaults, argv=[])

    assert captured["rebuild"] is False
    assert len(captured["configs"]) == 1
    (config,) = captured["configs"]
    assert config.output.directory_name == "pcsqcnn_architecture_sweep_full_readout/fq1_ql1"
    assert config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is False
