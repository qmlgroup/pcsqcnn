from __future__ import annotations

import math
from pathlib import Path

import pytest

from qcnn.model_spec import instantiate_model
from qcnn.model_stats import format_trainable_parameter_stats_line
from tests.script_loading import load_run_script


sp_qcnn = load_run_script("prepare_sp_qcnn_translated_mnist_data_module", "prepare_sp_qcnn_translated_mnist_data.py")
encoding_swap = load_run_script(
    "prepare_encoding_swap_data_module_for_sp_qcnn",
    "prepare_encoding_swap_data.py",
)


def make_defaults(module, tmp_path: Path, name: str):
    return getattr(module, name)(data_root=tmp_path / "data", artifacts_root=tmp_path / "artifacts")


def test_runs_cover_both_protocols_like_the_encoding_swap(tmp_path: Path) -> None:
    configs = sp_qcnn.iter_sp_qcnn_training_configs(make_defaults(sp_qcnn, tmp_path, "SpQcnnTranslatedMnistDefaults"))
    swap_train8, _, swap_train2, *_ = encoding_swap.iter_encoding_swap_configs(
        make_defaults(encoding_swap, tmp_path, "EncodingSwapDefaults")
    )

    assert [config.output.directory_name for config in configs] == [
        "sp_qcnn_translated_mnist/sp_qcnn_train8",
        "sp_qcnn_translated_mnist/sp_qcnn_train2",
    ]
    for config, reference in zip(configs, (swap_train8, swap_train2), strict=True):
        assert (config.dataset, config.optimizer, config.seeds) == (
            reference.dataset,
            reference.optimizer,
            reference.seeds,
        )
        assert config.training.num_epochs == reference.training.num_epochs
        assert config.training.torch_matmul_precision == "highest"


def test_model_matches_the_source_branch_configuration(tmp_path: Path) -> None:
    (config, _) = sp_qcnn.iter_sp_qcnn_training_configs(make_defaults(sp_qcnn, tmp_path, "SpQcnnTranslatedMnistDefaults"))

    # make_config(model="sp_qcnn") of run/prepare_qcs_baseline_data.py for the common FRQI benchmark.
    assert config.model.module == "qcnn.baselines.sp_qcnn"
    assert config.model.class_name == "SpQCNN"
    assert config.model.constructor_kwargs == {
        "image_size": 32,
        "num_classes": 10,
        "depth": 10,
        "brightness_range": (0.0, math.pi),
        "angle_init_scale": 2 * math.pi,
        "dtype": "float32",
    }
    assert format_trainable_parameter_stats_line(instantiate_model(config.model)) == "q200 q150 q12 c2570 Q362 C2570"


def test_seed_count_override(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(sp_qcnn, "run_auto_training_manifest_tasks", lambda configs, rebuild: captured.update(configs=configs))

    sp_qcnn.main(
        defaults=make_defaults(sp_qcnn, tmp_path, "SpQcnnTranslatedMnistDefaults"),
        argv=["--seed-count", "2"],
    )

    assert [config.seeds.seed_count for config in captured["configs"]] == [2, 2]


def test_unknown_runs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown sp-QCNN runs"):
        sp_qcnn.iter_sp_qcnn_training_configs(
            make_defaults(sp_qcnn, tmp_path, "SpQcnnTranslatedMnistDefaults"), runs=["sp_qcnn_train4"]
        )
