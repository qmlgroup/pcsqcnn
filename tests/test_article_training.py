from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import pytest

from qcnn import article_training
from tests.script_loading import load_run_script


@dataclass(frozen=True)
class _Figure2Defaults:
    data_root: Path
    artifacts_root: Path
    samples_per_class: int | None = 20
    image_size: int = 32
    scaled_image_size: int | None = 28
    max_offset: int = 2
    num_classes: int = 10
    optimizer_kind: str = "adam"
    learning_rate: float = 1e-2
    hybrid_learning_rate: float = 5e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 300
    set_to_none: bool = True
    test_evaluation_interval_epochs: int = 10
    snapshot_epochs: tuple[int, ...] = ()
    device: str | None = None
    torch_matmul_precision: str | None = "high"
    multiplexer_init_scale: float = 2.0 * math.pi
    base_seed: int = 0
    seed_count: int = 1
    use_timestamp_dir: bool = False
    fixed_feature_qubits: int = 2
    fixed_quantum_layers: int = 1
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False
    fixed_quantum_brightness_range: tuple[float, float] = (0.0, math.pi)


@dataclass(frozen=True)
class _BrightnessSweepDefaults(_Figure2Defaults):
    sweep_root_directory: str = "pcsqcnn_brightness_sweep"
    feature_qubits_options: tuple[int, ...] = (1, 2)
    quantum_layers_options: tuple[int, ...] = (1, 2)
    brightness_start_pi: Fraction = Fraction(0, 1)
    brightness_stop_pi: Fraction = Fraction(2, 1)
    num_interior_points: int = 24


@dataclass(frozen=True)
class _ArchitectureSweepDefaults(_Figure2Defaults):
    sweep_root_directory: str = "pcsqcnn_architecture_sweep"
    feature_qubits_options: tuple[int, ...] = (1, 2, 3)
    quantum_layers_options: tuple[int, ...] = (1, 2, 3, 4, 5)
    brightness_range: tuple[float, float] = (0.0, math.pi)


@dataclass(frozen=True)
class _ImageSizeSweepDefaults:
    data_root: Path
    artifacts_root: Path
    samples_per_class: int | None = None
    num_classes: int = 10
    optimizer_kind: str = "adam"
    learning_rate: float = 1e-2
    hybrid_learning_rate: float = 5e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True
    scaled_image_size: int | None = None
    max_offset: int = 0
    train_batch_size: int = 512
    test_batch_size: int = 16000
    num_epochs: int = 150
    set_to_none: bool = True
    test_evaluation_interval_epochs: int = 10
    snapshot_epochs: tuple[int, ...] = ()
    device: str | None = None
    torch_matmul_precision: str | None = "high"
    multiplexer_init_scale: float = 2.0 * math.pi
    base_seed: int = 0
    seed_count: int = 1
    use_timestamp_dir: bool = False
    scaled_image_size: int | None = 16
    feature_qubits: int = 2
    quantum_layers: int = 1
    brightness_range: tuple[float, float] = (0.0, math.pi)
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False
    sweep_root_directory: str = "pcsqcnn_image_size_sweep"


def test_generate_brightness_pi_coefficients_matches_default_24_point_grid(tmp_path: Path) -> None:
    defaults = _BrightnessSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    coefficients = article_training.generate_brightness_pi_coefficients(defaults)
    start_pi = Fraction(defaults.brightness_start_pi)
    stop_pi = Fraction(defaults.brightness_stop_pi)
    steps = [right - left for left, right in zip(coefficients, coefficients[1:], strict=False)]

    assert len(coefficients) == defaults.num_interior_points
    assert all(start_pi < coefficient < stop_pi for coefficient in coefficients)
    assert coefficients == tuple(sorted(coefficients))
    if steps:
        assert all(step == steps[0] for step in steps)
    assert coefficients[0] - start_pi == stop_pi - coefficients[-1]


def test_format_pi_fraction_suffix_uses_reduced_exact_pi_fraction() -> None:
    assert article_training.format_pi_fraction_suffix(Fraction(2, 25)) == "u2by25pi"
    assert article_training.format_pi_fraction_suffix(Fraction(10, 25)) == "u2by5pi"
    assert article_training.format_pi_fraction_suffix(Fraction(26, 25)) == "u26by25pi"


def test_build_readout_mode_suffix_matches_reduction_setting() -> None:
    assert (
        article_training.build_readout_mode_suffix(
            reduce_readout_to_feature_distribution=False
        )
        == "full_readout"
    )
    assert (
        article_training.build_readout_mode_suffix(
            reduce_readout_to_feature_distribution=True
        )
        == "reduced_readout"
    )


def test_build_brightness_sweep_directory_name_nests_run_under_sweep_root(tmp_path: Path) -> None:
    defaults = _BrightnessSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    feature_qubits = 1
    quantum_layers = 2
    brightness_stop_pi = Fraction(2, 25)
    directory_name = article_training.build_brightness_sweep_directory_name(
        defaults,
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
        brightness_stop_pi=brightness_stop_pi,
    )
    expected_leaf = (
        f"fq{feature_qubits}_ql{quantum_layers}_"
        f"{article_training.format_pi_fraction_suffix(brightness_stop_pi)}"
    )

    assert directory_name == str(Path(defaults.sweep_root_directory) / expected_leaf)
    assert Path(directory_name).parts == (defaults.sweep_root_directory, expected_leaf)


def test_build_architecture_sweep_root_directory_name_appends_readout_mode_suffix(tmp_path: Path) -> None:
    defaults = _ArchitectureSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )

    assert (
        article_training.build_architecture_sweep_root_directory_name(defaults)
        == "pcsqcnn_architecture_sweep_full_readout"
    )


def test_build_architecture_sweep_directory_name_nests_run_under_mode_root(tmp_path: Path) -> None:
    defaults = _ArchitectureSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    directory_name = article_training.build_architecture_sweep_directory_name(
        defaults,
        feature_qubits=2,
        quantum_layers=4,
    )

    assert directory_name == str(Path("pcsqcnn_architecture_sweep_full_readout") / "fq2_ql4")
    assert Path(directory_name).parts == ("pcsqcnn_architecture_sweep_full_readout", "fq2_ql4")


def test_build_image_size_sweep_directory_name_nests_run_under_sweep_root(tmp_path: Path) -> None:
    defaults = _ImageSizeSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    scaled_image_size = 28
    image_size = 32
    directory_name = article_training.build_image_size_sweep_directory_name(
        defaults,
        scaled_image_size=scaled_image_size,
        image_size=image_size,
    )

    assert directory_name == str(Path(defaults.sweep_root_directory) / f"{scaled_image_size}on{image_size}")
    assert Path(directory_name).parts == (defaults.sweep_root_directory, f"{scaled_image_size}on{image_size}")


def test_build_canonical_reference_run_directory_name_uses_16on16_pair() -> None:
    assert (
        article_training.build_canonical_reference_run_directory_name()
        == "pcsqcnn_image_size_sweep/16on16"
    )


def test_build_figure_2_model_spec_returns_expected_model_classes(tmp_path: Path) -> None:
    defaults = _Figure2Defaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
    )
    class_names = [
        article_training.build_figure_2_model_spec(model_kind, defaults=defaults).class_name
        for model_kind in article_training.FIXED_ARTICLE_MODEL_KINDS
    ]

    assert class_names == [
        "ClassicalMLP",
        "ClassicalCNN",
        "PCSQCNN",
        "PCSQCNNNoQFT",
    ]


def test_build_architecture_sweep_model_spec_forwards_readout_reduction(tmp_path: Path) -> None:
    defaults = _ArchitectureSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )
    model_spec = article_training.build_architecture_sweep_model_spec(
        defaults,
        feature_qubits=3,
        quantum_layers=5,
    )

    assert model_spec.class_name == "PCSQCNN"
    assert model_spec.constructor_kwargs["feature_qubits"] == 3
    assert model_spec.constructor_kwargs["quantum_layers"] == 5
    assert model_spec.constructor_kwargs["reduce_readout_to_feature_distribution"] is False


def test_build_brightness_sweep_model_spec_forwards_readout_reduction(tmp_path: Path) -> None:
    defaults = _BrightnessSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )

    model_spec = article_training.build_brightness_sweep_model_spec(
        defaults,
        feature_qubits=2,
        quantum_layers=1,
        brightness_stop_pi=Fraction(1, 2),
    )

    assert model_spec.class_name == "PCSQCNN"
    assert model_spec.constructor_kwargs["reduce_readout_to_feature_distribution"] is False


def test_build_image_size_sweep_model_spec_forwards_readout_reduction(tmp_path: Path) -> None:
    defaults = _ImageSizeSweepDefaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )

    model_spec = article_training.build_image_size_sweep_model_spec(
        defaults,
        image_size=32,
    )

    assert model_spec.class_name == "PCSQCNN"
    assert model_spec.constructor_kwargs["reduce_readout_to_feature_distribution"] is False


def test_build_article_auto_training_config_routes_hybrid_learning_rate(tmp_path: Path) -> None:
    defaults = _Figure2Defaults(
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        learning_rate=0.3,
        hybrid_learning_rate=0.07,
    )
    classical_config = article_training.build_article_auto_training_config(
        defaults,
        model_spec=article_training.build_figure_2_model_spec("classical_mlp", defaults=defaults),
        directory_name="classical_mlp_spc20",
    )
    hybrid_config = article_training.build_article_auto_training_config(
        defaults,
        model_spec=article_training.build_figure_2_model_spec("pcsqcnn", defaults=defaults),
        directory_name="pcsqcnn_spc20",
    )

    assert classical_config.optimizer.learning_rate == pytest.approx(defaults.learning_rate)
    assert hybrid_config.optimizer.learning_rate == pytest.approx(defaults.hybrid_learning_rate)


def test_article_training_defaults_are_inherited_by_prepare_scripts() -> None:
    shared_defaults = article_training.ArticleTrainingDefaults()
    shared_field_names = (
        "optimizer_kind",
        "learning_rate",
        "hybrid_learning_rate",
        "num_workers",
        "pin_memory",
        "download",
        "set_to_none",
        "test_evaluation_interval_epochs",
        "device",
        "torch_matmul_precision",
        "base_seed",
        "use_timestamp_dir",
    )
    prepare_modules = (
        load_run_script("prepare_translated_mnist_baselines_data_defaults_module", "prepare_translated_mnist_baselines_data.py"),
        load_run_script(
            "prepare_full_mnist_classical_baselines_data_defaults_module",
            "prepare_full_mnist_classical_baselines_data.py",
        ),
        load_run_script("prepare_brightness_sweep_data_defaults_module", "prepare_brightness_sweep_data.py"),
        load_run_script("prepare_pcsqcnn_architecture_sweep_data_defaults_module", "prepare_pcsqcnn_architecture_sweep_data.py"),
        load_run_script("prepare_pcsqcnn_image_size_sweep_data_defaults_module", "prepare_pcsqcnn_image_size_sweep_data.py"),
    )

    for module in prepare_modules:
        module_defaults = next(
            value
            for name, value in module.__dict__.items()
            if name.startswith("DEFAULT_") and name.endswith("_TRAINING_DEFAULTS")
        )
        expected_shared_values = {
            field_name: getattr(shared_defaults, field_name)
            for field_name in shared_field_names
        }
        if module.__name__ == "prepare_brightness_sweep_data_defaults_module":
            expected_shared_values["test_evaluation_interval_epochs"] = 150
        if module.__name__ == "prepare_full_mnist_classical_baselines_data_defaults_module":
            expected_shared_values["learning_rate"] = 5e-3
        assert {
            field_name: getattr(module_defaults, field_name)
            for field_name in shared_field_names
        } == expected_shared_values
