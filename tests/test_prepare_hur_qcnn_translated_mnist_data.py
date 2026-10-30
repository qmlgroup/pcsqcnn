from __future__ import annotations

from pathlib import Path

import pytest

from tests.script_loading import load_run_script


prepare_hur_data = load_run_script(
    "prepare_hur_qcnn_translated_mnist_data_module",
    "prepare_hur_qcnn_translated_mnist_data.py",
)
prepare_figure_2_data = load_run_script(
    "prepare_translated_mnist_baselines_data_module_for_hur",
    "prepare_translated_mnist_baselines_data.py",
)


def make_defaults(tmp_path: Path):
    return prepare_hur_data.HurQcnnTranslatedMnistDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )


def test_default_config_follows_the_figure_2_quantum_protocol(tmp_path: Path) -> None:
    figure_2_defaults = prepare_figure_2_data.Figure2TrainingDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )

    configs = prepare_hur_data.iter_hur_qcnn_training_configs(make_defaults(tmp_path))

    assert [config.output.directory_name for config in configs] == [
        "hur_qcnn_translated_mnist/su4_l1",
        "hur_qcnn_translated_mnist/su4_l4",
        "hur_qcnn_translated_mnist/su4_l16",
        "hur_qcnn_translated_mnist/su4_l64",
        "hur_qcnn_translated_mnist/ch8_su4_l1",
        "hur_qcnn_translated_mnist/ch112_su4_l1",
    ]
    # Single-channel specs keep the form of the saved su4_l1 runs.
    assert "channels" not in configs[0].model.constructor_kwargs
    assert [config.model.constructor_kwargs.get("channels", 1) for config in configs] == [1, 1, 1, 1, 8, 112]
    assert [config.model.constructor_kwargs["filters"] for config in configs] == [1, 4, 16, 64, 1, 1]
    config = configs[0]
    assert (config.model.module, config.model.class_name) == ("qcnn.hur", "HurQCNN")
    assert config.model.constructor_kwargs["filters"] == 1
    assert config.model.constructor_kwargs["angle_init_scale"] == figure_2_defaults.multiplexer_init_scale
    assert config.optimizer.learning_rate == figure_2_defaults.hybrid_learning_rate
    assert config.dataset.samples_per_class == figure_2_defaults.samples_per_class
    assert config.dataset.scaled_image_size == figure_2_defaults.scaled_image_size
    assert config.dataset.max_offset == figure_2_defaults.max_offset
    assert config.dataset.train_batch_size == figure_2_defaults.train_batch_size
    assert config.dataset.test_batch_size == figure_2_defaults.test_batch_size
    assert config.training.num_epochs == figure_2_defaults.num_epochs
    assert config.seeds.seed_count == figure_2_defaults.seed_count


def test_runs_filter_rejects_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Hur-QCNN runs"):
        prepare_hur_data.iter_hur_qcnn_training_configs(make_defaults(tmp_path), runs=["su4_l2"])


def test_overrides_change_learning_rate_init_scale_and_directory(tmp_path: Path) -> None:
    args = prepare_hur_data.parse_args(
        ["--learning-rate", "0.003", "--angle-init-scale", "0.5", "--root-directory", "hur_qcnn_translated_mnist_tuned"]
    )

    (config,) = prepare_hur_data.iter_hur_qcnn_training_configs(
        prepare_hur_data.apply_overrides(make_defaults(tmp_path), args),
        runs=["su4_l1"],
    )

    assert config.optimizer.learning_rate == 0.003
    assert config.model.constructor_kwargs["angle_init_scale"] == 0.5
    assert config.output.directory_name == "hur_qcnn_translated_mnist_tuned/su4_l1"
    assert prepare_hur_data.apply_overrides(make_defaults(tmp_path), prepare_hur_data.parse_args([])) == make_defaults(
        tmp_path
    )
