"""Train the FRQI sp-QCNN baseline on both translated-MNIST protocols.

The FRQI adaptation of Chinzei et al. (arXiv:2306.07331) uses 12 qubits,
splitting 12 -> (5, 5, 2) -> 12 singletons, depths (10, 10, 1), and an
invariant 8 x 8 x 4 orbit readout: 362 quantum and 2570 head parameters.
Float32 matrix products use highest precision.

The runs are:

- ``sp_qcnn_train8``: the Figure 2 benchmark, offsets ``[-8, 8]`` per axis
  for training and testing;
- ``sp_qcnn_train2``: the translation-generalization split, offsets
  ``[-2, 2]`` for training and ``[-8, 8]`` for testing.

The article reports seeds 0 and 1 (``--seed-count 2``). Everything else
follows the Figure 2 protocol: 1000 training images per
class, digits resized to 16 x 16 on a 32 x 32 canvas, Adam with the hybrid
learning rate, uniform initialization on ``[0, 2 pi)``, 2000 epochs,
train/test batch sizes 256/1600, test evaluation every 10 epochs, and seeds
0, 1, 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path

from qcnn import AutoTrainingConfig
from qcnn.article_training import ArticleTrainingDefaults, build_article_auto_training_config
from qcnn.automation import run_auto_training_manifest_tasks
from qcnn.model_spec import ModelSpec

DEFAULT_SP_QCNN_ROOT_DIRECTORY = "sp_qcnn_translated_mnist"


@dataclass(frozen=True)
class SpQcnnTranslatedMnistDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    test_max_offset: int | None = None
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    torch_matmul_precision: str | None = "highest"
    depth: int = 10
    brightness_range: tuple[float, float] = (0.0, math.pi)
    root_directory: str = DEFAULT_SP_QCNN_ROOT_DIRECTORY


DEFAULT_SP_QCNN_TRANSLATED_MNIST_DEFAULTS = SpQcnnTranslatedMnistDefaults()

# run name -> training offset; offsets below the test offset 8 add a test split
RUNS: dict[str, int] = {"sp_qcnn_train8": 8, "sp_qcnn_train2": 2}
TEST_MAX_OFFSET = 8


def build_sp_qcnn_model_spec(defaults: SpQcnnTranslatedMnistDefaults) -> ModelSpec:
    return ModelSpec(
        module="qcnn.baselines.sp_qcnn",
        class_name="SpQCNN",
        constructor_kwargs={
            "image_size": defaults.image_size,
            "num_classes": defaults.num_classes,
            "depth": defaults.depth,
            "brightness_range": defaults.brightness_range,
            "angle_init_scale": defaults.multiplexer_init_scale,
            "dtype": "float32",
        },
    )


def iter_sp_qcnn_training_configs(
    defaults: SpQcnnTranslatedMnistDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per run; ``runs`` restricts the family by name."""

    resolved_defaults = defaults or DEFAULT_SP_QCNN_TRANSLATED_MNIST_DEFAULTS
    selected = list(RUNS) if runs is None else list(runs)
    unknown = [name for name in selected if name not in RUNS]
    if unknown:
        raise ValueError(f"Unknown sp-QCNN runs {unknown}; available runs are {list(RUNS)}.")

    configs = []
    for name in selected:
        train_offset = RUNS[name]
        # Figure 2 runs keep test_max_offset unset, so their data match the Figure 2 runs exactly.
        run_defaults = replace(
            resolved_defaults,
            max_offset=train_offset,
            test_max_offset=None if train_offset == TEST_MAX_OFFSET else TEST_MAX_OFFSET,
        )
        configs.append(
            build_article_auto_training_config(
                run_defaults,
                model_spec=build_sp_qcnn_model_spec(run_defaults),
                directory_name=str(Path(resolved_defaults.root_directory) / name),
            )
        )
    return tuple(configs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the FRQI sp-QCNN baseline on both translated-MNIST protocols.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild experiment outputs even if the target directory already exists.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Optional subset of runs to train. Concurrent processes must use disjoint subsets.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="Override the number of seeds (default: 3; the article reports seeds 0 and 1).",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: SpQcnnTranslatedMnistDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved = defaults or DEFAULT_SP_QCNN_TRANSLATED_MNIST_DEFAULTS
    if args.seed_count is not None:
        resolved = replace(resolved, seed_count=args.seed_count)
    run_auto_training_manifest_tasks(
        iter_sp_qcnn_training_configs(resolved, runs=args.runs),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
