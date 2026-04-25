"""Shared article-training helpers built on top of generic automation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
from typing import Protocol

from qcnn.article_figures import figure_2_fixed_run_directory_name
from qcnn.automation import (
    AutoTrainingConfig,
    MnistDatasetConfig,
    OptimizerConfig,
    OutputConfig,
    SeedConfig,
    TrainingConfig,
)
from qcnn.classic import resolve_classical_mlp_hidden_widths
from qcnn.model_spec import ModelSpec


FIXED_ARTICLE_MODEL_KINDS: tuple[str, ...] = (
    "classical_mlp",
    "classical_cnn",
    "pcsqcnn",
    "pcsqcnn_no_qft",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_IMAGE_SIZE_SWEEP_ROOT_DIRECTORY = "pcsqcnn_image_size_sweep"
DEFAULT_CANONICAL_REFERENCE_SCALED_IMAGE_SIZE = 16
DEFAULT_CANONICAL_REFERENCE_IMAGE_SIZE = 16
DEFAULT_CANONICAL_REFERENCE_SNAPSHOT_EPOCHS: tuple[int, ...] = (10,) + tuple(
    range(100, 2001, 100)
)
DEFAULT_S2_REFERENCE_EPOCHS: tuple[int, ...] = tuple(range(100, 2001, 100))
DEFAULT_S2_ENTROPY_SHOT_BUDGETS: tuple[int, ...] = (128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class ArticleTrainingDefaults:
    """Shared defaults reused by article-oriented training entrypoints."""

    data_root: Path = DEFAULT_DATA_ROOT
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT
    num_classes: int = 10
    optimizer_kind: str = "adam"
    learning_rate: float = 1e-2
    hybrid_learning_rate: float = 3e-2
    weight_decay: float = 0.0
    momentum: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True
    set_to_none: bool = True
    test_evaluation_interval_epochs: int = 10
    snapshot_epochs: tuple[int, ...] = ()
    device: str | None = None
    torch_matmul_precision: str | None = "high"
    multiplexer_init_scale: float = 2.0 * math.pi
    base_seed: int = 0
    seed_count: int = 1
    use_timestamp_dir: bool = False


class _CommonTrainingDefaults(Protocol):
    data_root: str | Path
    artifacts_root: str | Path
    samples_per_class: int | None
    num_classes: int
    optimizer_kind: str
    learning_rate: float
    hybrid_learning_rate: float
    weight_decay: float
    momentum: float
    num_workers: int
    pin_memory: bool
    download: bool
    scaled_image_size: int | None
    max_offset: int
    train_batch_size: int
    test_batch_size: int
    num_epochs: int
    set_to_none: bool
    test_evaluation_interval_epochs: int
    snapshot_epochs: tuple[int, ...]
    device: str | None
    torch_matmul_precision: str | None
    multiplexer_init_scale: float
    base_seed: int
    seed_count: int
    use_timestamp_dir: bool


class _FixedFigure2Defaults(_CommonTrainingDefaults, Protocol):
    image_size: int
    fixed_feature_qubits: int
    fixed_quantum_layers: int
    fixed_quantum_reduce_readout_to_feature_distribution: bool
    fixed_quantum_brightness_range: tuple[float, float]


class _BrightnessSweepDefaults(_CommonTrainingDefaults, Protocol):
    image_size: int
    sweep_root_directory: str
    feature_qubits_options: tuple[int, ...]
    quantum_layers_options: tuple[int, ...]
    fixed_quantum_reduce_readout_to_feature_distribution: bool
    brightness_start_pi: Fraction
    brightness_stop_pi: Fraction
    num_interior_points: int


class _ArchitectureSweepDefaults(_CommonTrainingDefaults, Protocol):
    image_size: int
    sweep_root_directory: str
    feature_qubits_options: tuple[int, ...]
    quantum_layers_options: tuple[int, ...]
    brightness_range: tuple[float, float]
    fixed_quantum_reduce_readout_to_feature_distribution: bool


class _ImageSizeSweepDefaults(_CommonTrainingDefaults, Protocol):
    feature_qubits: int
    quantum_layers: int
    brightness_range: tuple[float, float]
    fixed_quantum_reduce_readout_to_feature_distribution: bool
    sweep_root_directory: str


def _normalize_pi_coefficient(value: Fraction | int) -> Fraction:
    return Fraction(value)


def generate_brightness_pi_coefficients(defaults: _BrightnessSweepDefaults) -> tuple[Fraction, ...]:
    start_pi = _normalize_pi_coefficient(defaults.brightness_start_pi)
    stop_pi = _normalize_pi_coefficient(defaults.brightness_stop_pi)
    step = (stop_pi - start_pi) / (defaults.num_interior_points + 1)
    return tuple(start_pi + step * index for index in range(1, defaults.num_interior_points + 1))


def format_pi_fraction_suffix(pi_coefficient: Fraction | int) -> str:
    normalized = _normalize_pi_coefficient(pi_coefficient)
    if normalized <= 0:
        raise ValueError(f"pi_coefficient must be positive, got {normalized}.")
    return f"u{normalized.numerator}by{normalized.denominator}pi"


def build_readout_mode_suffix(*, reduce_readout_to_feature_distribution: bool) -> str:
    return "reduced_readout" if reduce_readout_to_feature_distribution else "full_readout"


def build_pcsqcnn_model_spec(
    defaults: (
        _FixedFigure2Defaults
        | _BrightnessSweepDefaults
        | _ArchitectureSweepDefaults
        | _ImageSizeSweepDefaults
    ),
    *,
    image_size: int,
    feature_qubits: int,
    quantum_layers: int,
    brightness_range: tuple[float, float],
    reduce_readout_to_feature_distribution: bool,
) -> ModelSpec:
    return ModelSpec(
        module="qcnn.hybrid",
        class_name="PCSQCNN",
        constructor_kwargs={
            "image_size": image_size,
            "num_classes": defaults.num_classes,
            "feature_qubits": feature_qubits,
            "quantum_layers": quantum_layers,
            "brightness_range": brightness_range,
            "shot_budget": None,
            "reduce_readout_to_feature_distribution": reduce_readout_to_feature_distribution,
            "use_reduced_fourier_junction": True,
            "multiplexer_init_scale": defaults.multiplexer_init_scale,
        },
    )


def build_fixed_pcsqcnn_model_spec(defaults: _FixedFigure2Defaults) -> ModelSpec:
    return build_pcsqcnn_model_spec(
        defaults,
        image_size=defaults.image_size,
        feature_qubits=defaults.fixed_feature_qubits,
        quantum_layers=defaults.fixed_quantum_layers,
        brightness_range=defaults.fixed_quantum_brightness_range,
        reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        ),
    )


def build_fixed_pcsqcnn_no_qft_model_spec(defaults: _FixedFigure2Defaults) -> ModelSpec:
    return ModelSpec(
        module="qcnn.hybrid",
        class_name="PCSQCNNNoQFT",
        constructor_kwargs={
            "image_size": defaults.image_size,
            "num_classes": defaults.num_classes,
            "feature_qubits": defaults.fixed_feature_qubits,
            "quantum_layers": defaults.fixed_quantum_layers,
            "brightness_range": defaults.fixed_quantum_brightness_range,
            "shot_budget": None,
            "reduce_readout_to_feature_distribution": (
                defaults.fixed_quantum_reduce_readout_to_feature_distribution
            ),
            "multiplexer_init_scale": defaults.multiplexer_init_scale,
        },
    )


def build_figure_2_model_spec(
    model_kind: str,
    *,
    defaults: _FixedFigure2Defaults,
) -> ModelSpec:
    if model_kind == "pcsqcnn":
        return build_fixed_pcsqcnn_model_spec(defaults)
    if model_kind == "pcsqcnn_no_qft":
        return build_fixed_pcsqcnn_no_qft_model_spec(defaults)
    if model_kind == "classical_mlp":
        return ModelSpec(
            module="qcnn.classic",
            class_name="ClassicalMLP",
            constructor_kwargs={
                "image_size": defaults.image_size,
                "num_classes": defaults.num_classes,
                "hidden_widths": resolve_classical_mlp_hidden_widths(
                    defaults.image_size,
                    num_classes=defaults.num_classes,
                ),
                "dropout": 0.10,
            },
        )
    if model_kind == "classical_cnn":
        return ModelSpec(
            module="qcnn.classic",
            class_name="ClassicalCNN",
            constructor_kwargs={
                "image_size": defaults.image_size,
                "num_classes": defaults.num_classes,
                "base_channels": 16,
                "dropout": 0.10,
            },
        )
    raise ValueError(
        "Figure 2 training only supports 'classical_mlp', 'classical_cnn', "
        f"'pcsqcnn', and 'pcsqcnn_no_qft', got {model_kind!r}."
    )


def build_brightness_sweep_model_spec(
    defaults: _BrightnessSweepDefaults,
    *,
    feature_qubits: int,
    quantum_layers: int,
    brightness_stop_pi: Fraction,
) -> ModelSpec:
    return build_pcsqcnn_model_spec(
        defaults,
        image_size=defaults.image_size,
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
        brightness_range=(0.0, float(brightness_stop_pi) * math.pi),
        reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        ),
    )


def build_architecture_sweep_model_spec(
    defaults: _ArchitectureSweepDefaults,
    *,
    feature_qubits: int,
    quantum_layers: int,
) -> ModelSpec:
    return build_pcsqcnn_model_spec(
        defaults,
        image_size=defaults.image_size,
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
        brightness_range=defaults.brightness_range,
        reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        ),
    )


def build_image_size_sweep_model_spec(
    defaults: _ImageSizeSweepDefaults,
    *,
    image_size: int,
) -> ModelSpec:
    return build_pcsqcnn_model_spec(
        defaults,
        image_size=image_size,
        feature_qubits=defaults.feature_qubits,
        quantum_layers=defaults.quantum_layers,
        brightness_range=defaults.brightness_range,
        reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        ),
    )


def _is_article_hybrid_model_spec(model_spec: ModelSpec) -> bool:
    return (model_spec.module, model_spec.class_name) in {
        ("qcnn.hybrid", "PCSQCNN"),
        ("qcnn.hybrid", "PCSQCNNNoQFT"),
    }


def resolve_article_learning_rate(
    *,
    defaults: _CommonTrainingDefaults,
    model_spec: ModelSpec,
) -> float:
    if _is_article_hybrid_model_spec(model_spec):
        return defaults.hybrid_learning_rate
    return defaults.learning_rate


def build_article_auto_training_config(
    defaults: _CommonTrainingDefaults,
    *,
    model_spec: ModelSpec,
    directory_name: str,
    dataset_image_size: int | None = None,
) -> AutoTrainingConfig:
    resolved_image_size = dataset_image_size
    if resolved_image_size is None:
        resolved_image_size = getattr(defaults, "image_size", None)
    if resolved_image_size is None:
        raise ValueError("build_article_auto_training_config requires an explicit dataset_image_size.")

    resolved_learning_rate = resolve_article_learning_rate(defaults=defaults, model_spec=model_spec)
    return AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=defaults.data_root,
            samples_per_class=defaults.samples_per_class,
            image_size=resolved_image_size,
            scaled_image_size=defaults.scaled_image_size,
            max_offset=defaults.max_offset,
            train_batch_size=defaults.train_batch_size,
            test_batch_size=defaults.test_batch_size,
            num_workers=defaults.num_workers,
            pin_memory=defaults.pin_memory,
            download=defaults.download,
        ),
        model=model_spec,
        optimizer=OptimizerConfig(
            kind=defaults.optimizer_kind,
            learning_rate=resolved_learning_rate,
            weight_decay=defaults.weight_decay,
            momentum=defaults.momentum,
        ),
        training=TrainingConfig(
            num_epochs=defaults.num_epochs,
            set_to_none=defaults.set_to_none,
            test_evaluation_interval_epochs=defaults.test_evaluation_interval_epochs,
            snapshot_epochs=defaults.snapshot_epochs,
            device=defaults.device,
            torch_matmul_precision=defaults.torch_matmul_precision,
        ),
        seeds=SeedConfig(
            base_seed=defaults.base_seed,
            seed_count=defaults.seed_count,
        ),
        output=OutputConfig(
            root=defaults.artifacts_root,
            directory_name=directory_name,
            use_timestamp_dir=defaults.use_timestamp_dir,
        ),
    )


def build_figure_2_directory_name(
    model_kind: str,
    *,
    samples_per_class: int,
) -> str:
    return figure_2_fixed_run_directory_name(model_kind, samples_per_class=samples_per_class)


def build_brightness_sweep_run_subdirectory_name(
    *,
    feature_qubits: int,
    quantum_layers: int,
    brightness_stop_pi: Fraction,
) -> str:
    return f"fq{feature_qubits}_ql{quantum_layers}_{format_pi_fraction_suffix(brightness_stop_pi)}"


def build_brightness_sweep_directory_name(
    defaults: _BrightnessSweepDefaults,
    *,
    feature_qubits: int,
    quantum_layers: int,
    brightness_stop_pi: Fraction,
) -> str:
    run_subdirectory = build_brightness_sweep_run_subdirectory_name(
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
        brightness_stop_pi=brightness_stop_pi,
    )
    return f"{defaults.sweep_root_directory}/{run_subdirectory}"


def build_architecture_sweep_root_directory_name(defaults: _ArchitectureSweepDefaults) -> str:
    readout_mode_suffix = build_readout_mode_suffix(
        reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        )
    )
    return f"{defaults.sweep_root_directory}_{readout_mode_suffix}"


def build_architecture_sweep_run_subdirectory_name(
    *,
    feature_qubits: int,
    quantum_layers: int,
) -> str:
    return f"fq{feature_qubits}_ql{quantum_layers}"


def build_architecture_sweep_directory_name(
    defaults: _ArchitectureSweepDefaults,
    *,
    feature_qubits: int,
    quantum_layers: int,
) -> str:
    root_directory = build_architecture_sweep_root_directory_name(defaults)
    run_subdirectory = build_architecture_sweep_run_subdirectory_name(
        feature_qubits=feature_qubits,
        quantum_layers=quantum_layers,
    )
    return str(Path(root_directory) / run_subdirectory)


def build_image_size_sweep_run_subdirectory_name(
    *,
    scaled_image_size: int,
    image_size: int,
) -> str:
    return f"{scaled_image_size}on{image_size}"


def build_image_size_sweep_directory_name_from_root(
    *,
    sweep_root_directory: str,
    scaled_image_size: int,
    image_size: int,
) -> str:
    run_subdirectory = build_image_size_sweep_run_subdirectory_name(
        scaled_image_size=scaled_image_size,
        image_size=image_size,
    )
    return f"{sweep_root_directory}/{run_subdirectory}"


def build_image_size_sweep_directory_name(
    defaults: _ImageSizeSweepDefaults,
    *,
    scaled_image_size: int,
    image_size: int,
) -> str:
    return build_image_size_sweep_directory_name_from_root(
        sweep_root_directory=defaults.sweep_root_directory,
        scaled_image_size=scaled_image_size,
        image_size=image_size,
    )


def build_canonical_reference_run_directory_name(
    *,
    sweep_root_directory: str = DEFAULT_IMAGE_SIZE_SWEEP_ROOT_DIRECTORY,
) -> str:
    return build_image_size_sweep_directory_name_from_root(
        sweep_root_directory=sweep_root_directory,
        scaled_image_size=DEFAULT_CANONICAL_REFERENCE_SCALED_IMAGE_SIZE,
        image_size=DEFAULT_CANONICAL_REFERENCE_IMAGE_SIZE,
    )


def resolve_canonical_reference_run_directory(
    artifacts_root: str | Path,
    *,
    sweep_root_directory: str = DEFAULT_IMAGE_SIZE_SWEEP_ROOT_DIRECTORY,
) -> Path:
    return Path(artifacts_root) / build_canonical_reference_run_directory_name(
        sweep_root_directory=sweep_root_directory,
    )
