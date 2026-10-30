"""Train the FRQI QCS-QCNN controls on the translated-MNIST Figure 2 benchmark.

Every run uses the fixed Figure 2 translated-MNIST protocol of the PCS-QCNN
and RBC-QCNN reference runs: 1000 training images per class, digits resized to
16 x 16 and translated by at most 8 pixels per axis on a 32 x 32 canvas, Adam
with the hybrid learning rate, 2000 epochs, train/test batch sizes 256/1600,
test evaluation every 10 epochs, and seeds 0, 1, 2. The QCS-QCNN keeps the
PCS-QCNN encoder, pooling, conditioning, readout, and head (``Q = 3``,
``n_f = 2``) and replaces only the Fourier-multiplexer convolution by
qubit-cyclic-shift-equivariant layers on the FRQI index registers.

The default run family contains two ``QCSQCNN`` variants:

- ``fq2_ql3_sl1``: one sublayer pair per layer; the head (2,570 parameters)
  matches the PCS-QCNN head and the quantum core has 4,608 parameters;
- ``fq2_ql3_sl8``: eight sublayer pairs per layer; the quantum core (36,864
  parameters) and the head both match the PCS-QCNN, 39,434 parameters in
  total.

``--learning-rate`` and ``--generator-init-scale`` override the protocol values,
for example with the setting selected by ``tune_qcsqcnn_translated_mnist.py``;
pass ``--root-directory`` so such runs do not replace the protocol runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    DEFAULT_QCSQCNN_TRANSLATED_ROOT_DIRECTORY,
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_qcsqcnn_directory_name,
    build_qcsqcnn_model_spec,
    build_qcsqcnn_run_subdirectory_name,
)
from qcnn.automation import run_auto_training_manifest_tasks


@dataclass(frozen=True)
class QcsQcnnTranslatedMnistDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    feature_qubits: int = 2
    quantum_layers: int = 3
    brightness_range: tuple[float, float] = (0.0, math.pi)
    root_directory: str = DEFAULT_QCSQCNN_TRANSLATED_ROOT_DIRECTORY
    sublayer_options: tuple[int, ...] = (1, 8)


DEFAULT_QCSQCNN_TRANSLATED_MNIST_DEFAULTS = QcsQcnnTranslatedMnistDefaults()


def iter_qcsqcnn_training_configs(
    defaults: QcsQcnnTranslatedMnistDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per sublayer option.

    ``runs`` optionally restricts the family to the given run subdirectory
    names such as ``"fq2_ql3_sl8"``; unknown names raise ``ValueError``.
    """

    resolved_defaults = defaults or DEFAULT_QCSQCNN_TRANSLATED_MNIST_DEFAULTS
    available = {
        build_qcsqcnn_run_subdirectory_name(
            feature_qubits=resolved_defaults.feature_qubits,
            quantum_layers=resolved_defaults.quantum_layers,
            sublayers=sublayers,
        ): sublayers
        for sublayers in resolved_defaults.sublayer_options
    }
    selected = list(available) if runs is None else list(runs)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Unknown QCS-QCNN runs {unknown}; available runs are {list(available)}.")

    return tuple(
        build_article_auto_training_config(
            resolved_defaults,
            model_spec=build_qcsqcnn_model_spec(
                resolved_defaults,
                image_size=resolved_defaults.image_size,
                feature_qubits=resolved_defaults.feature_qubits,
                quantum_layers=resolved_defaults.quantum_layers,
                sublayers=available[name],
                brightness_range=resolved_defaults.brightness_range,
            ),
            directory_name=build_qcsqcnn_directory_name(
                feature_qubits=resolved_defaults.feature_qubits,
                quantum_layers=resolved_defaults.quantum_layers,
                sublayers=available[name],
                root_directory=resolved_defaults.root_directory,
            ),
        )
        for name in selected
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the translated-MNIST FRQI QCS-QCNN control artifacts.",
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
            "Optional subset of run subdirectories to train, for example 'fq2_ql3_sl8'. "
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
        "--generator-init-scale",
        type=float,
        default=None,
        help="Override the upper endpoint of the uniform Pauli-coefficient initialization (default: 2 pi).",
    )
    parser.add_argument(
        "--root-directory",
        default=None,
        help="Override the artifacts subdirectory that holds the runs, for example 'qcsqcnn_translated_mnist_tuned'.",
    )
    return parser.parse_args(argv)


def apply_overrides(
    defaults: QcsQcnnTranslatedMnistDefaults,
    args: argparse.Namespace,
) -> QcsQcnnTranslatedMnistDefaults:
    """Apply the optional learning-rate, initialization, and directory overrides."""

    overrides: dict[str, object] = {}
    if args.learning_rate is not None:
        overrides["hybrid_learning_rate"] = args.learning_rate
    if args.generator_init_scale is not None:
        overrides["multiplexer_init_scale"] = args.generator_init_scale
    if args.root_directory is not None:
        overrides["root_directory"] = args.root_directory
    return replace(defaults, **overrides)


def main(
    *,
    defaults: QcsQcnnTranslatedMnistDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved = apply_overrides(defaults or DEFAULT_QCSQCNN_TRANSLATED_MNIST_DEFAULTS, args)
    run_auto_training_manifest_tasks(
        iter_qcsqcnn_training_configs(resolved, runs=args.runs),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
