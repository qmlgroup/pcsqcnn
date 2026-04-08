from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_brightness_sweep_directory_name,
    build_brightness_sweep_model_spec,
    generate_brightness_pi_coefficients,
)
from qcnn.automation import run_auto_training_manifest_tasks


@dataclass(frozen=True)
class BrightnessSweepTrainingDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 20
    image_size: int = 32
    scaled_image_size: int | None = 28
    max_offset: int = 2
    fixed_quantum_reduce_readout_to_feature_distribution: bool = False
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 300
    test_evaluation_interval_epochs: int = 150
    seed_count: int = 3
    sweep_root_directory: str = "pcsqcnn_brightness_sweep"
    feature_qubits_options: tuple[int, ...] = (1, 2)
    quantum_layers_options: tuple[int, ...] = (1, 2)
    brightness_start_pi: Fraction = Fraction(0, 1)
    brightness_stop_pi: Fraction = Fraction(2, 1)
    num_interior_points: int = 24


DEFAULT_BRIGHTNESS_SWEEP_TRAINING_DEFAULTS = BrightnessSweepTrainingDefaults()


def iter_brightness_sweep_training_configs(
    defaults: BrightnessSweepTrainingDefaults | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    resolved_defaults = defaults or DEFAULT_BRIGHTNESS_SWEEP_TRAINING_DEFAULTS
    brightness_pi_coefficients = generate_brightness_pi_coefficients(resolved_defaults)
    result: list[AutoTrainingConfig] = []
    for feature_qubits in resolved_defaults.feature_qubits_options:
        for quantum_layers in resolved_defaults.quantum_layers_options:
            for brightness_stop_pi in brightness_pi_coefficients:
                result.append(
                    build_article_auto_training_config(
                        resolved_defaults,
                        model_spec=build_brightness_sweep_model_spec(
                            resolved_defaults,
                            feature_qubits=feature_qubits,
                            quantum_layers=quantum_layers,
                            brightness_stop_pi=brightness_stop_pi,
                        ),
                        directory_name=build_brightness_sweep_directory_name(
                            resolved_defaults,
                            feature_qubits=feature_qubits,
                            quantum_layers=quantum_layers,
                            brightness_stop_pi=brightness_stop_pi,
                        ),
                    )
                )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the PCS-QCNN brightness-sweep training artifacts.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: BrightnessSweepTrainingDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_auto_training_manifest_tasks(
        iter_brightness_sweep_training_configs(defaults),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
