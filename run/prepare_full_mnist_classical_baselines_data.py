from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_figure_2_model_spec,
)
from qcnn.automation import run_auto_training_manifest_tasks


_FULL_MNIST_CLASSICAL_MODEL_KINDS: tuple[str, ...] = ("classical_mlp", "classical_cnn")


@dataclass(frozen=True)
class FullMnistClassicalBaselinesTrainingDefaults(ArticleTrainingDefaults):
    samples_per_class: int | None = None
    learning_rate: float = 5e-3
    image_size: int = 32
    scaled_image_size: int | None = 32
    max_offset: int = 0
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 1000
    seed_count: int = 3
    output_root_directory: str = "full_mnist_classical_baselines"


DEFAULT_FULL_MNIST_CLASSICAL_BASELINES_TRAINING_DEFAULTS = (
    FullMnistClassicalBaselinesTrainingDefaults()
)


def iter_full_mnist_classical_baseline_configs(
    defaults: FullMnistClassicalBaselinesTrainingDefaults | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    resolved_defaults = defaults or DEFAULT_FULL_MNIST_CLASSICAL_BASELINES_TRAINING_DEFAULTS
    run_suffix = f"{resolved_defaults.scaled_image_size}on{resolved_defaults.image_size}"
    return tuple(
        build_article_auto_training_config(
            resolved_defaults,
            model_spec=build_figure_2_model_spec(model_kind, defaults=resolved_defaults),
            directory_name=f"{resolved_defaults.output_root_directory}/{model_kind}/{run_suffix}",
        )
        for model_kind in _FULL_MNIST_CLASSICAL_MODEL_KINDS
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the full-MNIST classical baseline artifacts for the supplementary figure.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: FullMnistClassicalBaselinesTrainingDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_auto_training_manifest_tasks(
        iter_full_mnist_classical_baseline_configs(defaults),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
