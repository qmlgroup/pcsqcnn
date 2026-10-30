from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.script_loading import load_run_script


prepare_henderson_data = load_run_script(
    "prepare_henderson_qcnn_translated_mnist_data_module",
    "prepare_henderson_qcnn_translated_mnist_data.py",
)
prepare_figure_2_data = load_run_script(
    "prepare_translated_mnist_baselines_data_module_for_henderson",
    "prepare_translated_mnist_baselines_data.py",
)


def test_parse_args_accepts_rebuild_runs_and_no_compile() -> None:
    args = prepare_henderson_data.parse_args(["--rebuild", "--runs", "ch12_ql3", "ch1_ql1", "--no-compile"])

    assert args.rebuild is True
    assert args.runs == ["ch12_ql3", "ch1_ql1"]
    assert args.no_compile is True


def test_default_configs_follow_the_figure_2_translated_protocol(tmp_path: Path) -> None:
    defaults = prepare_henderson_data.HendersonQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    figure_2_defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    configs = prepare_henderson_data.iter_henderson_qcnn_training_configs(defaults)

    assert [config.output.directory_name for config in configs] == [
        "henderson_qcnn_translated_mnist/ch12_ql3",
        "henderson_qcnn_translated_mnist/ch1_ql1",
        "henderson_qcnn_translated_mnist/ch1_ql3",
    ]
    for config in configs:
        assert config.model.module == "qcnn.henderson"
        assert config.model.class_name == "HendersonQCNN"
        assert config.model.constructor_kwargs["brightness_range"] == (0.0, math.pi)
        assert config.model.constructor_kwargs["block_init_scale"] == 2.0 * math.pi
        assert config.optimizer.kind == "adam"
        assert config.optimizer.learning_rate == figure_2_defaults.hybrid_learning_rate
        assert config.dataset.samples_per_class == figure_2_defaults.samples_per_class
        assert config.dataset.image_size == figure_2_defaults.image_size
        assert config.dataset.scaled_image_size == figure_2_defaults.scaled_image_size
        assert config.dataset.max_offset == figure_2_defaults.max_offset
        assert config.dataset.train_batch_size == figure_2_defaults.train_batch_size
        assert config.dataset.test_batch_size == figure_2_defaults.test_batch_size
        assert config.training.num_epochs == figure_2_defaults.num_epochs
        assert config.training.test_evaluation_interval_epochs == 10
        assert config.seeds.base_seed == 0
        assert config.seeds.seed_count == figure_2_defaults.seed_count
    assert [
        (config.model.constructor_kwargs["channels"], config.model.constructor_kwargs["quantum_layers"])
        for config in configs
    ] == [(12, 3), (1, 1), (1, 3)]


def test_runs_filter_selects_a_subset_in_the_requested_order(tmp_path: Path) -> None:
    defaults = prepare_henderson_data.HendersonQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    configs = prepare_henderson_data.iter_henderson_qcnn_training_configs(defaults, runs=["ch1_ql3", "ch12_ql3"])

    assert [config.output.directory_name for config in configs] == [
        "henderson_qcnn_translated_mnist/ch1_ql3",
        "henderson_qcnn_translated_mnist/ch12_ql3",
    ]


def test_runs_filter_rejects_unknown_names(tmp_path: Path) -> None:
    defaults = prepare_henderson_data.HendersonQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    with pytest.raises(ValueError, match="Unknown Henderson-QCNN runs"):
        prepare_henderson_data.iter_henderson_qcnn_training_configs(defaults, runs=["ch3_ql3"])
