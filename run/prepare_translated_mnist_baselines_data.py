from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_figure_2_directory_name,
    build_figure_2_model_spec,
)
from qcnn.automation import run_auto_training_manifest_tasks

_CLASSICAL_FIGURE_2_MODEL_KINDS: tuple[str, ...] = ("classical_mlp", "classical_cnn")
_QUANTUM_FIGURE_2_MODEL_KINDS: tuple[str, ...] = (
    # "pcsqcnn",
    "pcsqcnn_no_qft",
)


@dataclass(frozen=True)
class Figure2TrainingDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    scaled_image_size_options: tuple[int, ...] = (16, 32)
    max_offset_options: tuple[int, ...] = (8, 0)
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    fixed_feature_qubits: int = 2
    fixed_quantum_layers: int = 3
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False
    fixed_quantum_brightness_range: tuple[float, float] = (0.0, math.pi)


DEFAULT_FIGURE_2_TRAINING_DEFAULTS = Figure2TrainingDefaults()


def iter_figure_2_training_configs(
    defaults: Figure2TrainingDefaults | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    resolved_defaults = defaults or DEFAULT_FIGURE_2_TRAINING_DEFAULTS
    if len(resolved_defaults.scaled_image_size_options) != len(resolved_defaults.max_offset_options):
        raise ValueError(
            "Figure 2 classical baselines require scaled_image_size_options and "
            "max_offset_options to have the same length."
        )

    result: list[AutoTrainingConfig] = []
    for model_kind in _CLASSICAL_FIGURE_2_MODEL_KINDS:
        for scaled_image_size, max_offset in zip(
            resolved_defaults.scaled_image_size_options,
            resolved_defaults.max_offset_options,
            strict=True,
        ):
            run_defaults = replace(
                resolved_defaults,
                scaled_image_size=scaled_image_size,
                max_offset=max_offset,
            )
            result.append(
                build_article_auto_training_config(
                    run_defaults,
                    model_spec=build_figure_2_model_spec(model_kind, defaults=run_defaults),
                    directory_name=f"{model_kind}/{scaled_image_size}on{run_defaults.image_size}",
                )
            )

    result.extend(
        build_article_auto_training_config(
            resolved_defaults,
            model_spec=build_figure_2_model_spec(model_kind, defaults=resolved_defaults),
            directory_name=build_figure_2_directory_name(
                model_kind,
                samples_per_class=resolved_defaults.samples_per_class,
            ),
        )
        for model_kind in _QUANTUM_FIGURE_2_MODEL_KINDS
    )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed Figure 2 training artifacts.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: Figure2TrainingDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_auto_training_manifest_tasks(
        iter_figure_2_training_configs(defaults),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
