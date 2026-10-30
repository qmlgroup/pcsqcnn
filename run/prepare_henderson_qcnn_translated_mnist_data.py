"""Train the Henderson-QCNN baselines on the translated-MNIST Figure 2 benchmark.

Every run uses the fixed Figure 2 translated-MNIST protocol of the PCS-QCNN
and RBC-QCNN reference runs: 1000 training images per class, digits resized to
16 x 16 and translated by at most 8 pixels per axis on a 32 x 32 canvas, Adam
with the hybrid learning rate, 2000 epochs, train/test batch sizes 256/1600,
test evaluation every 10 epochs, and seeds 0, 1, 2. The quantum
initialization scale is the article value ``Uniform(0, 2 pi)``.

The default run family contains three ``HendersonQCNN`` architectures:

- ``ch12_ql3``: the depth-matched (``Q = 3``) and parameter-matched main
  baseline with 12 filter channels (40,714 trainable parameters versus 39,434
  for the ``Q = 3``, ``n_f = 2`` PCS-QCNN);
- ``ch1_ql1``: the single trainable quanvolution layer of the PennyLane demo
  geometry (2 x 2 kernel, stride 2) with full readout (41,226 parameters);
- ``ch1_ql3``: the depth-3 single-channel control whose 256-dimensional
  readout and 2,570-parameter head match the PCS-QCNN head exactly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    DEFAULT_HENDERSON_QCNN_TRANSLATED_ROOT_DIRECTORY,
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_henderson_qcnn_directory_name,
    build_henderson_qcnn_model_spec,
    build_henderson_qcnn_run_subdirectory_name,
)
from qcnn.automation import run_auto_training_manifest_tasks
from qcnn.henderson import use_compiled_block_transfer


@dataclass(frozen=True)
class HendersonQcnnTranslatedMnistDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    brightness_range: tuple[float, float] = (0.0, math.pi)
    root_directory: str = DEFAULT_HENDERSON_QCNN_TRANSLATED_ROOT_DIRECTORY
    architectures: tuple[tuple[int, int], ...] = ((12, 3), (1, 1), (1, 3))


DEFAULT_HENDERSON_QCNN_TRANSLATED_MNIST_DEFAULTS = HendersonQcnnTranslatedMnistDefaults()


def iter_henderson_qcnn_training_configs(
    defaults: HendersonQcnnTranslatedMnistDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per requested ``(channels, quantum_layers)`` architecture.

    ``runs`` optionally restricts the family to the given run subdirectory
    names such as ``"ch12_ql3"``; unknown names raise ``ValueError``.
    """

    resolved_defaults = defaults or DEFAULT_HENDERSON_QCNN_TRANSLATED_MNIST_DEFAULTS
    available = {
        build_henderson_qcnn_run_subdirectory_name(channels=channels, quantum_layers=quantum_layers): (
            channels,
            quantum_layers,
        )
        for channels, quantum_layers in resolved_defaults.architectures
    }
    selected = list(available) if runs is None else list(runs)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Unknown Henderson-QCNN runs {unknown}; available runs are {list(available)}.")

    result: list[AutoTrainingConfig] = []
    for name in selected:
        channels, quantum_layers = available[name]
        result.append(
            build_article_auto_training_config(
                resolved_defaults,
                model_spec=build_henderson_qcnn_model_spec(
                    resolved_defaults,
                    image_size=resolved_defaults.image_size,
                    channels=channels,
                    quantum_layers=quantum_layers,
                    brightness_range=resolved_defaults.brightness_range,
                ),
                directory_name=build_henderson_qcnn_directory_name(
                    channels=channels,
                    quantum_layers=quantum_layers,
                    root_directory=resolved_defaults.root_directory,
                ),
            )
        )
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the translated-MNIST Henderson-QCNN baseline artifacts.",
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
        help=(
            "Optional subset of run subdirectories to train, for example 'ch12_ql3'. "
            "Concurrent processes must use disjoint subsets."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Keep the eager block contraction instead of torch.compile on CUDA.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: HendersonQcnnTranslatedMnistDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    configs = iter_henderson_qcnn_training_configs(defaults, runs=args.runs)
    if torch.cuda.is_available() and not args.no_compile:
        use_compiled_block_transfer(True)
    run_auto_training_manifest_tasks(configs, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
