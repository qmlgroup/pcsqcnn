"""Train on small translations and test on the full translated-MNIST benchmark.

The translated-MNIST benchmark of Figure 2 draws training and test offsets
from the same range, so a model can learn every digit position from data and
the test accuracy does not reward translation equivariance. This experiment
separates the two: digits resized to 16 x 16 are placed on a 32 x 32 canvas
with offsets in ``[-2, 2]`` per axis for training and in ``[-8, 8]`` for
testing. The test split is exactly the Figure 2 test split of the same seed.
Offsets ``-2..2`` contain every residue modulo 4, so a model that is exactly
invariant to 4-pixel shifts covers all test positions by symmetry.

Everything else follows the Figure 2 protocol: 1000 training images per class,
Adam with the hybrid learning rate, uniform initialization on ``[0, 2 pi)``,
2000 epochs, train/test batch sizes 256/1600, test evaluation every 10 epochs,
and seeds 0, 1, 2. The default run family contains:

- ``pcsqcnn_fq2_ql3_train2``: the Figure 2 PCS-QCNN (``Q = 3``, ``n_f = 2``,
  full readout, 39,434 parameters);
- ``pcsqcnn_fq2_ql3_reduced_train2``: the same PCS-QCNN with the readout
  summed over positions (``reduce_readout_to_feature_distribution=True``),
  which makes it exactly invariant to 4-pixel shifts (36,914 parameters);
- ``hur_ch8_su4_l1_train2``: the eight-channel Hur-QCNN (2,826 parameters);
- ``pcsqcnn_fq2_ql3_reduced_train8``: the reduced-readout PCS-QCNN trained on
  the full ``[-8, 8]`` offsets, the in-distribution reference for the
  reduced readout (the full-readout PCS-QCNN and the Hur-QCNN have theirs in
  the Figure 2 and Hur-QCNN run families).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import math
from pathlib import Path

from qcnn import AutoTrainingConfig
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_hur_qcnn_model_spec,
    build_pcsqcnn_model_spec,
)
from qcnn.automation import run_auto_training_manifest_tasks
from qcnn.model_spec import ModelSpec

DEFAULT_TRANSLATION_GENERALIZATION_ROOT_DIRECTORY = "translation_generalization"


@dataclass(frozen=True)
class TranslationGeneralizationDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 2
    test_max_offset: int | None = 8
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    feature_qubits: int = 2
    quantum_layers: int = 3
    brightness_range: tuple[float, float] = (0.0, math.pi)
    hur_channels: int = 8
    root_directory: str = DEFAULT_TRANSLATION_GENERALIZATION_ROOT_DIRECTORY


DEFAULT_TRANSLATION_GENERALIZATION_DEFAULTS = TranslationGeneralizationDefaults()


def _pcsqcnn_spec(*, reduced: bool) -> Callable[[TranslationGeneralizationDefaults], ModelSpec]:
    def build(defaults: TranslationGeneralizationDefaults) -> ModelSpec:
        return build_pcsqcnn_model_spec(
            defaults,
            image_size=defaults.image_size,
            feature_qubits=defaults.feature_qubits,
            quantum_layers=defaults.quantum_layers,
            brightness_range=defaults.brightness_range,
            reduce_readout_to_feature_distribution=reduced,
        )

    return build


def _hur_spec(defaults: TranslationGeneralizationDefaults) -> ModelSpec:
    return build_hur_qcnn_model_spec(
        defaults,
        image_size=defaults.image_size,
        filters=1,
        channels=defaults.hur_channels,
    )


# run name -> (model spec builder, training offset; None uses ``max_offset``)
RUNS: dict[str, tuple[Callable[[TranslationGeneralizationDefaults], ModelSpec], int | None]] = {
    "pcsqcnn_fq2_ql3_train2": (_pcsqcnn_spec(reduced=False), None),
    "pcsqcnn_fq2_ql3_reduced_train2": (_pcsqcnn_spec(reduced=True), None),
    "hur_ch8_su4_l1_train2": (_hur_spec, None),
    "pcsqcnn_fq2_ql3_reduced_train8": (_pcsqcnn_spec(reduced=True), 8),
}


def iter_translation_generalization_configs(
    defaults: TranslationGeneralizationDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per run; ``runs`` restricts the family by name."""

    resolved_defaults = defaults or DEFAULT_TRANSLATION_GENERALIZATION_DEFAULTS
    selected = list(RUNS) if runs is None else list(runs)
    unknown = [name for name in selected if name not in RUNS]
    if unknown:
        raise ValueError(f"Unknown translation-generalization runs {unknown}; available runs are {list(RUNS)}.")

    configs = []
    for name in selected:
        build_spec, train_offset = RUNS[name]
        run_defaults = resolved_defaults if train_offset is None else replace(resolved_defaults, max_offset=train_offset)
        configs.append(
            build_article_auto_training_config(
                run_defaults,
                model_spec=build_spec(run_defaults),
                directory_name=str(Path(resolved_defaults.root_directory) / name),
            )
        )
    return tuple(configs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on small translations and test on the full translated-MNIST benchmark.",
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
    return parser.parse_args(argv)


def main(
    *,
    defaults: TranslationGeneralizationDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    run_auto_training_manifest_tasks(
        iter_translation_generalization_configs(defaults, runs=args.runs),
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
