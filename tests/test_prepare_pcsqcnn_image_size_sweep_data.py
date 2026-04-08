from __future__ import annotations

from pathlib import Path

import pytest

from qcnn import article_training
from tests.script_loading import load_run_script


prepare_figure_5b_data = load_run_script(
    "prepare_pcsqcnn_image_size_sweep_data_module",
    "prepare_pcsqcnn_image_size_sweep_data.py",
)


def test_parse_args_accepts_rebuild_flag() -> None:
    args = prepare_figure_5b_data.parse_args(["--rebuild"])
    assert args.rebuild is True


def test_figure_5b_training_defaults_accept_inherited_shared_kwargs(tmp_path: Path) -> None:
    baseline = prepare_figure_5b_data.Figure5bTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    defaults = prepare_figure_5b_data.Figure5bTrainingDefaults(
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
    assert defaults.scaled_image_size_options == (8, 16, 28, 32)
    assert defaults.image_size_options == (8, 16, 32, 32)
    assert defaults.reference_scaled_image_size == 16
    assert defaults.reference_image_size == 16
    assert defaults.reference_snapshot_epochs == (10,) + tuple(range(100, 2001, 100))


def test_iter_figure_5b_training_configs_returns_image_size_sweep_runs(tmp_path: Path) -> None:
    defaults = prepare_figure_5b_data.Figure5bTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    configs = prepare_figure_5b_data.iter_figure_5b_training_configs(defaults)

    assert [config.output.directory_name for config in configs] == [
        article_training.build_image_size_sweep_directory_name(
            defaults,
            scaled_image_size=scaled_image_size,
            image_size=image_size,
        )
        for scaled_image_size, image_size in zip(
            defaults.scaled_image_size_options,
            defaults.image_size_options,
            strict=True,
        )
    ]
    assert [config.dataset.scaled_image_size for config in configs] == [8, 16, 28, 32]
    assert [config.dataset.image_size for config in configs] == [8, 16, 32, 32]
    assert [config.training.snapshot_epochs for config in configs] == [
        (),
        defaults.reference_snapshot_epochs,
        (),
        (),
    ]
    assert all(config.model.class_name == "PCSQCNN" for config in configs)
    assert all(config.optimizer.learning_rate == pytest.approx(defaults.hybrid_learning_rate) for config in configs)
    assert all(
        config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is False
        for config in configs
    )


def test_main_dispatches_only_figure_5b_configs_and_forwards_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5b_data.Figure5bTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        scaled_image_size_options=(8, 16, 28),
        image_size_options=(8, 16, 32),
    )
    captured: dict[str, object] = {}

    def fake_run_auto_training_manifest_tasks(configs, *, rebuild: bool) -> None:
        captured["configs"] = tuple(configs)
        captured["rebuild"] = rebuild

    monkeypatch.setattr(
        prepare_figure_5b_data,
        "run_auto_training_manifest_tasks",
        fake_run_auto_training_manifest_tasks,
    )

    prepare_figure_5b_data.main(defaults=defaults, argv=["--rebuild"])

    assert captured["rebuild"] is True
    assert [config.dataset.scaled_image_size for config in captured["configs"]] == [8, 16, 28]
    assert [config.dataset.image_size for config in captured["configs"]] == [8, 16, 32]


def test_iter_figure_5b_training_configs_rejects_mismatched_option_lengths(tmp_path: Path) -> None:
    defaults = prepare_figure_5b_data.Figure5bTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        scaled_image_size_options=(8, 16, 28),
        image_size_options=(8, 16),
    )

    with pytest.raises(ValueError, match="same length"):
        prepare_figure_5b_data.iter_figure_5b_training_configs(defaults)


def test_iter_figure_5b_training_configs_requires_exactly_one_canonical_reference_pair(
    tmp_path: Path,
) -> None:
    defaults = prepare_figure_5b_data.Figure5bTrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        scaled_image_size_options=(8, 28, 32),
        image_size_options=(8, 32, 32),
    )

    with pytest.raises(ValueError, match="canonical reference pair"):
        prepare_figure_5b_data.iter_figure_5b_training_configs(defaults)
