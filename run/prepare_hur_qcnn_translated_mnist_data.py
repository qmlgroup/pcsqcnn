"""Train the Hur-QCNN baseline on the translated-MNIST Figure 2 benchmark.

Every run uses the fixed Figure 2 translated-MNIST protocol of the PCS-QCNN
and RBC-QCNN reference runs: 1000 training images per class, digits resized to
16 x 16 and translated by at most 8 pixels per axis on a 32 x 32 canvas, Adam
with the hybrid learning rate, uniform initialization on ``[0, 2 pi)``, 2000
epochs, train/test batch sizes 256/1600, test evaluation every 10 epochs, and
seeds 0, 1, 2.

The model is the amplitude-encoded QCNN of Hur, Kim, and Park
(arXiv:2108.00661) with the SU(4) convolution and the two-parameter pooling of
their ansatz 9a: the 32 x 32 image is loaded into 10 qubits, one
convolution-pooling layer halves the register to 5 qubits, a final convolution
acts on them, and the 32 outcome probabilities of the 5 measured qubits feed
the linear-softmax head. The default run family contains:

- ``su4_l1``: the paper's single circuit with one convolutional filter per
  layer, as in the paper's Tables I and II: 32 quantum and 330 head
  parameters;
- ``su4_l4``, ``su4_l16``, ``su4_l64``: the same single circuit with ``l``
  filters per layer (the paper's ``l_i``), ``30 l + 2`` quantum parameters
  (122, 482, 1,922) and the same 330-parameter head;
- ``ch8_su4_l1``: eight independent circuits (channels) whose 256 readout
  probabilities and 2,570-parameter head match the PCS-QCNN head, 256
  quantum parameters;
- ``ch112_su4_l1``: 112 channels, 39,434 parameters in total (3,584 quantum,
  35,850 head) as the PCS-QCNN.

Matching the 36,864 quantum parameters of the PCS-QCNN with one circuit would
take about 1,230 filters per layer, roughly 95 hours per seed on a laptop
RTX 3050; ``su4_l64`` takes about 4 hours per seed.

``--learning-rate`` and ``--angle-init-scale`` override the protocol values,
for example with the setting selected by ``tune_hur_qcnn_translated_mnist.py``;
pass ``--root-directory`` so such runs do not replace the protocol runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    DEFAULT_HUR_QCNN_TRANSLATED_ROOT_DIRECTORY,
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_hur_qcnn_directory_name,
    build_hur_qcnn_model_spec,
    build_hur_qcnn_run_subdirectory_name,
)
from qcnn.automation import run_auto_training_manifest_tasks


@dataclass(frozen=True)
class HurQcnnTranslatedMnistDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    root_directory: str = DEFAULT_HUR_QCNN_TRANSLATED_ROOT_DIRECTORY
    # (channels, filters per layer) of every run
    variants: tuple[tuple[int, int], ...] = ((1, 1), (1, 4), (1, 16), (1, 64), (8, 1), (112, 1))


DEFAULT_HUR_QCNN_TRANSLATED_MNIST_DEFAULTS = HurQcnnTranslatedMnistDefaults()


def iter_hur_qcnn_training_configs(
    defaults: HurQcnnTranslatedMnistDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per ``(channels, filters)`` variant.

    ``runs`` optionally restricts the family to the given run subdirectory
    names such as ``"su4_l1"``; unknown names raise ``ValueError``.
    """

    resolved_defaults = defaults or DEFAULT_HUR_QCNN_TRANSLATED_MNIST_DEFAULTS
    available = {
        build_hur_qcnn_run_subdirectory_name(filters=filters, channels=channels): (channels, filters)
        for channels, filters in resolved_defaults.variants
    }
    selected = list(available) if runs is None else list(runs)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Unknown Hur-QCNN runs {unknown}; available runs are {list(available)}.")

    return tuple(
        build_article_auto_training_config(
            resolved_defaults,
            model_spec=build_hur_qcnn_model_spec(
                resolved_defaults,
                image_size=resolved_defaults.image_size,
                filters=available[name][1],
                channels=available[name][0],
            ),
            directory_name=build_hur_qcnn_directory_name(
                filters=available[name][1],
                channels=available[name][0],
                root_directory=resolved_defaults.root_directory,
            ),
        )
        for name in selected
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the translated-MNIST Hur-QCNN baseline artifacts.",
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
            "Optional subset of run subdirectories to train, for example 'ch8_su4_l1'. "
            "Concurrent processes must use disjoint subsets."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the Adam learning rate (default: the shared hybrid learning rate).",
    )
    parser.add_argument(
        "--angle-init-scale",
        type=float,
        default=None,
        help="Override the upper endpoint of the uniform gate-angle initialization (default: 2 pi).",
    )
    parser.add_argument(
        "--root-directory",
        default=None,
        help="Override the artifacts subdirectory that holds the runs, for example 'hur_qcnn_translated_mnist_tuned'.",
    )
    return parser.parse_args(argv)


def apply_overrides(
    defaults: HurQcnnTranslatedMnistDefaults,
    args: argparse.Namespace,
) -> HurQcnnTranslatedMnistDefaults:
    """Apply the optional learning-rate, initialization, and directory overrides."""

    overrides: dict[str, object] = {}
    if args.learning_rate is not None:
        overrides["hybrid_learning_rate"] = args.learning_rate
    if args.angle_init_scale is not None:
        overrides["multiplexer_init_scale"] = args.angle_init_scale
    if args.root_directory is not None:
        overrides["root_directory"] = args.root_directory
    return replace(defaults, **overrides)


def main(
    *,
    defaults: HurQcnnTranslatedMnistDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved = apply_overrides(defaults or DEFAULT_HUR_QCNN_TRANSLATED_MNIST_DEFAULTS, args)
    run_auto_training_manifest_tasks(
        iter_hur_qcnn_training_configs(resolved, runs=args.runs),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
