from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_architecture_sweep_directory_name,
    build_architecture_sweep_model_spec,
    build_article_auto_training_config,
)
from qcnn.automation import run_auto_training_manifest_tasks


@dataclass(frozen=True)
class Figure5aTrainingDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    feature_qubits_options: tuple[int, ...] = (1, 2, 3)
    quantum_layers_options: tuple[int, ...] = (1, 2, 3, 4, 5)
    brightness_range: tuple[float, float] = (0.0, math.pi)
    sweep_root_directory: str = "pcsqcnn_architecture_sweep"
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False


DEFAULT_FIGURE_5A_TRAINING_DEFAULTS = Figure5aTrainingDefaults()


def resolve_figure_5a_training_defaults(
    defaults: Figure5aTrainingDefaults | None = None,
) -> Figure5aTrainingDefaults:
    resolved_defaults = defaults or DEFAULT_FIGURE_5A_TRAINING_DEFAULTS
    return replace(
        resolved_defaults,
        fixed_quantum_reduce_readout_to_feature_distribution=False,
    )


def iter_figure_5a_training_configs(
    defaults: Figure5aTrainingDefaults | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    resolved_defaults = resolve_figure_5a_training_defaults(defaults)
    result: list[AutoTrainingConfig] = []
    for quantum_layers in resolved_defaults.quantum_layers_options:
        for feature_qubits in resolved_defaults.feature_qubits_options:
            result.append(
                build_article_auto_training_config(
                    resolved_defaults,
                    model_spec=build_architecture_sweep_model_spec(
                        resolved_defaults,
                        feature_qubits=feature_qubits,
                        quantum_layers=quantum_layers,
                    ),
                    directory_name=build_architecture_sweep_directory_name(
                        resolved_defaults,
                        feature_qubits=feature_qubits,
                        quantum_layers=quantum_layers,
                    ),
                )
            )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Figure 5a translated-MNIST PCS-QCNN architecture-sweep artifacts.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: Figure5aTrainingDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved_defaults = resolve_figure_5a_training_defaults(defaults)
    run_auto_training_manifest_tasks(
        iter_figure_5a_training_configs(resolved_defaults),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
