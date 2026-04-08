from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    DEFAULT_CANONICAL_REFERENCE_IMAGE_SIZE,
    DEFAULT_CANONICAL_REFERENCE_SCALED_IMAGE_SIZE,
    DEFAULT_CANONICAL_REFERENCE_SNAPSHOT_EPOCHS,
    build_article_auto_training_config,
    build_image_size_sweep_directory_name,
    build_image_size_sweep_model_spec,
)
from qcnn.automation import run_auto_training_manifest_tasks


@dataclass(frozen=True)
class Figure5bTrainingDefaults(ArticleTrainingDefaults):
    """Figure 5b size sweep defaults.

    `scaled_image_size_options[i]` and `image_size_options[i]` are applied
    synchronously and together define the `i`-th sweep configuration.

    The canonical reference configuration is the `16on16` pair. Only that run
    receives the dense snapshot schedule used by Figures 6, 7, S2, and S3.
    """

    samples_per_class: int | None = None
    max_offset: int = 0
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False
    train_batch_size: int = 512
    test_batch_size: int = 16000
    num_epochs: int = 2000
    seed_count: int = 3
    scaled_image_size_options: tuple[int, ...] = (8, 16, 28, 32)
    image_size_options: tuple[int, ...] = (8, 16, 32, 32)
    reference_scaled_image_size: int = DEFAULT_CANONICAL_REFERENCE_SCALED_IMAGE_SIZE
    reference_image_size: int = DEFAULT_CANONICAL_REFERENCE_IMAGE_SIZE
    reference_snapshot_epochs: tuple[int, ...] = DEFAULT_CANONICAL_REFERENCE_SNAPSHOT_EPOCHS
    feature_qubits: int = 3
    quantum_layers: int = 1
    brightness_range: tuple[float, float] = (0.0, math.pi)
    sweep_root_directory: str = "pcsqcnn_image_size_sweep"


@dataclass(frozen=True)
class _ResolvedFigure5bTrainingDefaults:
    data_root: str
    artifacts_root: str
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
    scaled_image_size: int
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
    image_size: int
    feature_qubits: int
    quantum_layers: int
    brightness_range: tuple[float, float]
    fixed_quantum_reduce_readout_to_feature_distribution: bool
    sweep_root_directory: str


DEFAULT_FIGURE_5B_TRAINING_DEFAULTS = Figure5bTrainingDefaults()


def _resolve_figure_5b_pair_defaults(
    defaults: Figure5bTrainingDefaults,
    *,
    scaled_image_size: int,
    image_size: int,
) -> _ResolvedFigure5bTrainingDefaults:
    snapshot_epochs = (
        defaults.reference_snapshot_epochs
        if (
            scaled_image_size == defaults.reference_scaled_image_size
            and image_size == defaults.reference_image_size
        )
        else ()
    )
    return _ResolvedFigure5bTrainingDefaults(
        data_root=defaults.data_root,
        artifacts_root=defaults.artifacts_root,
        samples_per_class=defaults.samples_per_class,
        num_classes=defaults.num_classes,
        optimizer_kind=defaults.optimizer_kind,
        learning_rate=defaults.learning_rate,
        hybrid_learning_rate=defaults.hybrid_learning_rate,
        weight_decay=defaults.weight_decay,
        momentum=defaults.momentum,
        num_workers=defaults.num_workers,
        pin_memory=defaults.pin_memory,
        download=defaults.download,
        scaled_image_size=scaled_image_size,
        max_offset=defaults.max_offset,
        train_batch_size=defaults.train_batch_size,
        test_batch_size=defaults.test_batch_size,
        num_epochs=defaults.num_epochs,
        set_to_none=defaults.set_to_none,
        test_evaluation_interval_epochs=defaults.test_evaluation_interval_epochs,
        snapshot_epochs=snapshot_epochs,
        device=defaults.device,
        torch_matmul_precision=defaults.torch_matmul_precision,
        multiplexer_init_scale=defaults.multiplexer_init_scale,
        base_seed=defaults.base_seed,
        seed_count=defaults.seed_count,
        use_timestamp_dir=defaults.use_timestamp_dir,
        image_size=image_size,
        feature_qubits=defaults.feature_qubits,
        quantum_layers=defaults.quantum_layers,
        brightness_range=defaults.brightness_range,
        fixed_quantum_reduce_readout_to_feature_distribution=(
            defaults.fixed_quantum_reduce_readout_to_feature_distribution
        ),
        sweep_root_directory=defaults.sweep_root_directory,
    )


def iter_figure_5b_training_configs(
    defaults: Figure5bTrainingDefaults | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    resolved_defaults = defaults or DEFAULT_FIGURE_5B_TRAINING_DEFAULTS
    if len(resolved_defaults.scaled_image_size_options) != len(resolved_defaults.image_size_options):
        raise ValueError(
            "Figure 5b size sweep requires scaled_image_size_options and "
            "image_size_options to have the same length."
        )
    configured_pairs = tuple(
        zip(
            resolved_defaults.scaled_image_size_options,
            resolved_defaults.image_size_options,
            strict=True,
        )
    )
    reference_pair = (
        resolved_defaults.reference_scaled_image_size,
        resolved_defaults.reference_image_size,
    )
    reference_pair_count = configured_pairs.count(reference_pair)
    if reference_pair_count != 1:
        raise ValueError(
            "Figure 5b requires exactly one canonical reference pair "
            f"{reference_pair!r}, found {reference_pair_count}."
        )
    result: list[AutoTrainingConfig] = []
    for scaled_image_size, image_size in configured_pairs:
        pair_defaults = _resolve_figure_5b_pair_defaults(
            resolved_defaults,
            scaled_image_size=scaled_image_size,
            image_size=image_size,
        )
        result.append(
            build_article_auto_training_config(
                pair_defaults,
                model_spec=build_image_size_sweep_model_spec(
                    pair_defaults,
                    image_size=image_size,
                ),
                directory_name=build_image_size_sweep_directory_name(
                    pair_defaults,
                    scaled_image_size=scaled_image_size,
                    image_size=image_size,
                ),
                dataset_image_size=image_size,
            )
        )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Figure 5b pairwise size-sweep training artifacts.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: Figure5bTrainingDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_auto_training_manifest_tasks(
        iter_figure_5b_training_configs(defaults),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
