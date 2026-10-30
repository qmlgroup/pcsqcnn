from __future__ import annotations

from pathlib import Path

import pytest

from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from tests.script_loading import load_run_script


local_pcs = load_run_script(
    "prepare_local_pcsqcnn_translated_mnist_data_module",
    "prepare_local_pcsqcnn_translated_mnist_data.py",
)
encoding_swap = load_run_script(
    "prepare_encoding_swap_data_module_for_local_pcsqcnn",
    "prepare_encoding_swap_data.py",
)
generalization = load_run_script(
    "prepare_translation_generalization_data_module_for_local_pcsqcnn",
    "prepare_translation_generalization_data.py",
)


def make_defaults(module, tmp_path: Path, name: str):
    return getattr(module, name)(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts")


def test_runs_cover_both_protocols_like_the_encoding_swap(tmp_path: Path) -> None:
    configs = local_pcs.iter_local_pcsqcnn_training_configs(
        make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults")
    )
    swap_train8, _, swap_train2, *_ = encoding_swap.iter_encoding_swap_configs(
        make_defaults(encoding_swap, tmp_path, "EncodingSwapDefaults")
    )

    assert [config.output.directory_name for config in configs] == [
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train2",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_reduced_train8",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_reduced_train2",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e4000_warm",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e6000_warm",
    ]
    for config, reference in zip(configs, (swap_train8, swap_train2) * 2 + (swap_train8,) * 2, strict=True):
        assert (config.dataset, config.optimizer, config.seeds) == (
            reference.dataset,
            reference.optimizer,
            reference.seeds,
        )
        assert config.training.num_epochs == reference.training.num_epochs
        assert config.training.torch_matmul_precision == "highest"


def test_model_is_the_figure_2_pcsqcnn_with_the_local_multiplexer(tmp_path: Path) -> None:
    (config, _, reduced_train8, reduced_train2, warm, warm6000) = local_pcs.iter_local_pcsqcnn_training_configs(
        make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults")
    )
    (pcs_generalization,) = generalization.iter_translation_generalization_configs(
        make_defaults(generalization, tmp_path, "TranslationGeneralizationDefaults"),
        runs=["pcsqcnn_fq2_ql3_train2"],
    )

    kwargs = dict(config.model.constructor_kwargs)
    assert kwargs.pop("multiplexer_kind") == "local_3x3"
    assert kwargs == pcs_generalization.model.constructor_kwargs
    assert format_trainable_parameter_stats_line(instantiate_model(config.model)) == "q32 q128 q128 c2570 Q288 C2570"
    assert config.training.init_checkpoint is None
    assert warm.model == config.model
    assert warm.training.init_checkpoint == str(
        tmp_path / "artifacts/local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8/checkpoint_final_seed{seed}.pt"
    )
    assert warm6000.model == config.model
    assert warm6000.training.init_checkpoint == str(
        tmp_path / "artifacts/local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e4000_warm/checkpoint_final_seed{seed}.pt"
    )

    (reduced_generalization,) = generalization.iter_translation_generalization_configs(
        make_defaults(generalization, tmp_path, "TranslationGeneralizationDefaults"),
        runs=["pcsqcnn_fq2_ql3_reduced_train2"],
    )
    for reduced in (reduced_train8, reduced_train2):
        reduced_kwargs = dict(reduced.model.constructor_kwargs)
        assert reduced_kwargs.pop("multiplexer_kind") == "local_3x3"
        assert reduced_kwargs == reduced_generalization.model.constructor_kwargs
        assert format_trainable_parameter_stats_line(instantiate_model(reduced.model)) == "q32 q128 q128 c50 Q288 C50"


def test_seed_count_override(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(local_pcs, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))

    local_pcs.main(
        defaults=make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults"),
        argv=["--runs", "pcsqcnn_local_fq2_ql3_reduced_train2", "--seed-count", "1"],
    )

    (config,) = captured["configs"]
    assert config.seeds.seed_count == 1
    assert config.model.constructor_kwargs["reduce_readout_to_feature_distribution"] is True


def test_num_epochs_override_names_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(local_pcs, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))

    local_pcs.main(
        defaults=make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults"),
        argv=["--runs", "pcsqcnn_local_fq2_ql3_reduced_train8", "pcsqcnn_local_fq2_ql3_reduced_train2", "--num-epochs", "4000"],
    )

    assert [config.output.directory_name for config in captured["configs"]] == [
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_reduced_train8_e4000",
        "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_reduced_train2_e4000",
    ]
    assert [config.training.num_epochs for config in captured["configs"]] == [4000, 4000]
    assert [config.seeds.seed_count for config in captured["configs"]] == [3, 3]


def test_unknown_runs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown local PCS-QCNN runs"):
        local_pcs.iter_local_pcsqcnn_training_configs(
            make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults"), runs=["pcsqcnn_local_train4"]
        )


def test_base_seed_and_directory_suffix_train_extra_seeds_apart(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(local_pcs, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))

    local_pcs.main(
        defaults=make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults"),
        argv=[
            "--runs", "pcsqcnn_local_fq2_ql3_train8_e4000_warm",
            "--base-seed", "1", "--seed-count", "2", "--directory-suffix", "_seeds1-2",
        ],
    )

    (config,) = captured["configs"]
    assert (config.seeds.base_seed, config.seeds.seed_count) == (1, 2)
    assert config.output.directory_name == "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e4000_warm_seeds1-2"
    assert config.training.init_checkpoint.endswith("pcsqcnn_local_fq2_ql3_train8/checkpoint_final_seed{seed}.pt")


@pytest.mark.parametrize("options", [
    ["--num-epochs", "10"],
    ["--directory-suffix", "_pilot"],
    ["--num-epochs", "10", "--directory-suffix", "_pilot"],
])
def test_warm_restart_chain_uses_selected_predecessor_paths(tmp_path: Path, monkeypatch, options) -> None:
    captured = {}
    monkeypatch.setattr(local_pcs, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))
    local_pcs.main(
        defaults=make_defaults(local_pcs, tmp_path, "LocalPcsqcnnTranslatedMnistDefaults"),
        argv=["--runs", "pcsqcnn_local_fq2_ql3_train8", "pcsqcnn_local_fq2_ql3_train8_e4000_warm",
              "pcsqcnn_local_fq2_ql3_train8_e6000_warm", *options],
    )
    first, second, third = captured["configs"]
    for source, destination in [(first, second), (second, third)]:
        assert destination.training.init_checkpoint == str(
            Path(source.output.root) / source.output.directory_name / "checkpoint_final_seed{seed}.pt"
        )
