from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.script_loading import load_run_script


prepare_qcsqcnn_data = load_run_script(
    "prepare_qcsqcnn_translated_mnist_data_module",
    "prepare_qcsqcnn_translated_mnist_data.py",
)
prepare_figure_2_data = load_run_script(
    "prepare_translated_mnist_baselines_data_module_for_qcs",
    "prepare_translated_mnist_baselines_data.py",
)


def test_parse_args_accepts_rebuild_and_runs() -> None:
    args = prepare_qcsqcnn_data.parse_args(["--rebuild", "--runs", "fq2_ql3_sl8"])

    assert args.rebuild is True
    assert args.runs == ["fq2_ql3_sl8"]


def test_default_configs_follow_the_figure_2_quantum_protocol(tmp_path: Path) -> None:
    defaults = prepare_qcsqcnn_data.QcsQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    figure_2_defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    configs = prepare_qcsqcnn_data.iter_qcsqcnn_training_configs(defaults)

    assert [config.output.directory_name for config in configs] == [
        "qcsqcnn_translated_mnist/fq2_ql3_sl1",
        "qcsqcnn_translated_mnist/fq2_ql3_sl8",
    ]
    for config, sublayers in zip(configs, (1, 8), strict=True):
        kwargs = config.model.constructor_kwargs
        assert (config.model.module, config.model.class_name) == ("qcnn.qcs", "QCSQCNN")
        assert kwargs["feature_qubits"] == figure_2_defaults.fixed_feature_qubits
        assert kwargs["quantum_layers"] == figure_2_defaults.fixed_quantum_layers
        assert kwargs["sublayers"] == sublayers
        assert kwargs["brightness_range"] == figure_2_defaults.fixed_quantum_brightness_range == (0.0, math.pi)
        assert kwargs["generator_init_scale"] == 2.0 * math.pi
        assert config.optimizer.learning_rate == figure_2_defaults.hybrid_learning_rate
        assert config.dataset.samples_per_class == figure_2_defaults.samples_per_class
        assert config.dataset.scaled_image_size == figure_2_defaults.scaled_image_size
        assert config.dataset.max_offset == figure_2_defaults.max_offset
        assert config.dataset.train_batch_size == figure_2_defaults.train_batch_size
        assert config.dataset.test_batch_size == figure_2_defaults.test_batch_size
        assert config.training.num_epochs == figure_2_defaults.num_epochs
        assert config.seeds.seed_count == figure_2_defaults.seed_count


def test_runs_filter_rejects_unknown_names(tmp_path: Path) -> None:
    defaults = prepare_qcsqcnn_data.QcsQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    assert [
        config.output.directory_name
        for config in prepare_qcsqcnn_data.iter_qcsqcnn_training_configs(defaults, runs=["fq2_ql3_sl8"])
    ] == ["qcsqcnn_translated_mnist/fq2_ql3_sl8"]
    with pytest.raises(ValueError, match="Unknown QCS-QCNN runs"):
        prepare_qcsqcnn_data.iter_qcsqcnn_training_configs(defaults, runs=["fq2_ql3_sl2"])


def test_overrides_change_learning_rate_init_scale_and_directory(tmp_path: Path) -> None:
    defaults = prepare_qcsqcnn_data.QcsQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    args = prepare_qcsqcnn_data.parse_args(
        [
            "--runs",
            "fq2_ql3_sl8",
            "--learning-rate",
            "0.003",
            "--generator-init-scale",
            "0.05",
            "--root-directory",
            "qcsqcnn_translated_mnist_tuned",
        ]
    )

    (config,) = prepare_qcsqcnn_data.iter_qcsqcnn_training_configs(
        prepare_qcsqcnn_data.apply_overrides(defaults, args),
        runs=args.runs,
    )

    assert config.optimizer.learning_rate == 0.003
    assert config.model.constructor_kwargs["generator_init_scale"] == 0.05
    assert config.output.directory_name == "qcsqcnn_translated_mnist_tuned/fq2_ql3_sl8"


def test_no_overrides_keep_the_protocol_values(tmp_path: Path) -> None:
    defaults = prepare_qcsqcnn_data.QcsQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    assert prepare_qcsqcnn_data.apply_overrides(defaults, prepare_qcsqcnn_data.parse_args([])) == defaults
