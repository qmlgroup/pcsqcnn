from __future__ import annotations

from pathlib import Path

import pytest

from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from tests.script_loading import load_run_script


generalization = load_run_script(
    "prepare_translation_generalization_data_module",
    "prepare_translation_generalization_data.py",
)
prepare_figure_2_data = load_run_script(
    "prepare_translated_mnist_baselines_data_module_for_generalization",
    "prepare_translated_mnist_baselines_data.py",
)


def make_defaults(tmp_path: Path):
    return generalization.TranslationGeneralizationDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )


def test_runs_train_on_small_offsets_and_test_on_the_benchmark_offsets(tmp_path: Path) -> None:
    figure_2 = prepare_figure_2_data.Figure2TrainingDefaults(data_root=tmp_path / "data", artifacts_root=tmp_path / "a")

    configs = generalization.iter_translation_generalization_configs(make_defaults(tmp_path))

    assert [config.output.directory_name for config in configs] == [
        "translation_generalization/pcsqcnn_fq2_ql3_train2",
        "translation_generalization/pcsqcnn_fq2_ql3_reduced_train2",
        "translation_generalization/hur_ch8_su4_l1_train2",
        "translation_generalization/pcsqcnn_fq2_ql3_reduced_train8",
    ]
    assert [config.dataset.max_offset for config in configs] == [2, 2, 2, 8]
    for config in configs:
        assert config.dataset.test_max_offset == figure_2.max_offset == 8
        assert config.dataset.samples_per_class == figure_2.samples_per_class
        assert config.dataset.scaled_image_size == figure_2.scaled_image_size == 16
        assert config.dataset.image_size == 32
        assert config.optimizer.learning_rate == figure_2.hybrid_learning_rate
        assert config.training.num_epochs == figure_2.num_epochs
        assert config.seeds.seed_count == figure_2.seed_count


def test_models_match_the_benchmark_models(tmp_path: Path) -> None:
    configs = generalization.iter_translation_generalization_configs(make_defaults(tmp_path))
    figure_2_pcs = prepare_figure_2_data.build_figure_2_model_spec(
        "pcsqcnn",
        defaults=prepare_figure_2_data.Figure2TrainingDefaults(data_root=tmp_path / "d", artifacts_root=tmp_path / "a"),
    )

    assert configs[0].model == figure_2_pcs
    assert configs[1].model.constructor_kwargs["reduce_readout_to_feature_distribution"] is True
    lines = [format_trainable_parameter_stats_line(instantiate_model(config.model)) for config in configs]
    assert lines[0].endswith("Q36864 C2570")
    assert lines[1].endswith("Q36864 C50")
    assert lines[2] == "q120 q16 q120 c2570 Q256 C2570"


def test_unknown_runs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown translation-generalization runs"):
        generalization.iter_translation_generalization_configs(make_defaults(tmp_path), runs=["pcsqcnn_train4"])
