from __future__ import annotations

from pathlib import Path

import pytest

from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from tests.script_loading import load_run_script


encoding_swap = load_run_script("prepare_encoding_swap_data_module", "prepare_encoding_swap_data.py")
generalization = load_run_script(
    "prepare_translation_generalization_data_module_for_encoding_swap",
    "prepare_translation_generalization_data.py",
)
prepare_hur = load_run_script(
    "prepare_hur_qcnn_translated_mnist_data_module_for_encoding_swap",
    "prepare_hur_qcnn_translated_mnist_data.py",
)


def make_defaults(tmp_path: Path):
    return encoding_swap.EncodingSwapDefaults(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts")


def test_runs_cover_both_protocols(tmp_path: Path) -> None:
    configs = encoding_swap.iter_encoding_swap_configs(make_defaults(tmp_path))

    assert [config.output.directory_name for config in configs] == [
        "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8",
        "encoding_swap/hur_su4_l1_frqi_train8",
        "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train2",
        "encoding_swap/hur_su4_l1_frqi_train2",
        "encoding_swap/hur_su4_l1_train2",
        "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train8",
        "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train2",
    ]
    assert [config.dataset.max_offset for config in configs] == [8, 8, 2, 2, 2, 8, 2]
    assert [config.dataset.test_max_offset for config in configs] == [None, None, 8, 8, 8, None, 8]


def test_protocols_match_the_figure_2_and_generalization_runs(tmp_path: Path) -> None:
    configs = encoding_swap.iter_encoding_swap_configs(make_defaults(tmp_path))
    (hur_figure_2,) = prepare_hur.iter_hur_qcnn_training_configs(
        prepare_hur.HurQcnnTranslatedMnistDefaults(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts"),
        runs=["su4_l1"],
    )
    (hur_generalization,) = generalization.iter_translation_generalization_configs(
        generalization.TranslationGeneralizationDefaults(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts"),
        runs=["hur_ch8_su4_l1_train2"],
    )

    for config in (*configs[:2], configs[5]):
        assert (config.dataset, config.optimizer, config.training, config.seeds) == (
            hur_figure_2.dataset,
            hur_figure_2.optimizer,
            hur_figure_2.training,
            hur_figure_2.seeds,
        )
    for config in (*configs[2:5], configs[6]):
        assert (config.dataset, config.optimizer, config.training, config.seeds) == (
            hur_generalization.dataset,
            hur_generalization.optimizer,
            hur_generalization.training,
            hur_generalization.seeds,
        )


def test_models_swap_only_the_encoding(tmp_path: Path) -> None:
    configs = encoding_swap.iter_encoding_swap_configs(make_defaults(tmp_path))
    (pcs_generalization,) = generalization.iter_translation_generalization_configs(
        generalization.TranslationGeneralizationDefaults(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts"),
        runs=["pcsqcnn_fq2_ql3_train2"],
    )

    pcs_kwargs = dict(configs[0].model.constructor_kwargs)
    assert pcs_kwargs.pop("encoding") == "amplitude"
    assert pcs_kwargs == pcs_generalization.model.constructor_kwargs
    assert configs[1].model.constructor_kwargs["encoding"] == "frqi"
    assert "encoding" not in configs[4].model.constructor_kwargs

    lines = [format_trainable_parameter_stats_line(instantiate_model(config.model)) for config in configs]
    assert lines[0].endswith("Q36864 C2570")
    assert lines[1] == lines[3] == "q15 q2 q15 c650 Q32 C650"
    assert lines[4] == "q15 q2 q15 c330 Q32 C330"

    reduced_kwargs = dict(configs[5].model.constructor_kwargs)
    assert reduced_kwargs.pop("reduce_readout_to_feature_distribution") is True
    assert configs[0].model.constructor_kwargs.pop("reduce_readout_to_feature_distribution") is False
    assert reduced_kwargs == configs[0].model.constructor_kwargs
    assert lines[5].endswith("Q36864 C50")
    assert configs[6].model == configs[5].model


def test_warm_continuation_runs_only_when_named(tmp_path: Path) -> None:
    defaults = make_defaults(tmp_path)
    configs = encoding_swap.iter_encoding_swap_configs(defaults)
    (warm,) = encoding_swap.iter_encoding_swap_configs(defaults, runs=["hur_su4_l1_frqi_train8_e6000_warm"])

    assert "encoding_swap/hur_su4_l1_frqi_train8_e6000_warm" not in [config.output.directory_name for config in configs]
    assert warm.model == configs[1].model
    assert (warm.dataset, warm.optimizer, warm.seeds) == (configs[1].dataset, configs[1].optimizer, configs[1].seeds)
    assert warm.training.num_epochs == 4000
    assert warm.training.init_checkpoint == str(
        tmp_path / "artifacts/encoding_swap/hur_su4_l1_frqi_train8/checkpoint_final_seed{seed}.pt"
    )


def test_unknown_runs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown encoding-swap runs"):
        encoding_swap.iter_encoding_swap_configs(make_defaults(tmp_path), runs=["pcsqcnn_frqi_train8"])


def test_base_seed_and_directory_suffix_train_extra_seeds_apart(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(encoding_swap, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))

    encoding_swap.main(
        defaults=make_defaults(tmp_path),
        argv=[
            "--runs", "hur_su4_l1_frqi_train8_e6000_warm",
            "--base-seed", "1", "--seed-count", "2", "--directory-suffix", "_seeds1-2",
        ],
    )

    (config,) = captured["configs"]
    assert (config.seeds.base_seed, config.seeds.seed_count) == (1, 2)
    assert config.output.directory_name == "encoding_swap/hur_su4_l1_frqi_train8_e6000_warm_seeds1-2"
    assert config.training.num_epochs == 4000
