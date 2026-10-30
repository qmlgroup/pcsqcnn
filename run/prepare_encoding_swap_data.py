"""Swap the encodings of the PCS-QCNN and the Hur-QCNN on translated MNIST.

The PCS-QCNN loads images with the FRQI encoding and the Hur-QCNN with the
amplitude encoding, so the Figure 2 comparison of the two mixes the encoding
with the architecture. This experiment trains each architecture with the
encoding of the other:

- ``pcsqcnn_fq2_ql3_amplitude``: the Figure 2 PCS-QCNN (``Q = 3``,
  ``n_f = 2``, full readout) on the amplitude encoding of the Hur-QCNN: the
  normalized pixel values are the amplitudes of the index registers and the
  feature register starts in ``|00>`` (39,434 parameters, as with FRQI);
- ``hur_su4_l1_frqi``: the paper's single Hur-QCNN circuit (one SU(4)
  filter per layer) on the FRQI encoding of the PCS-QCNN, with the colour
  qubit as the last wire of the convolution ring: ``11 -> conv -> pool -> 6
  -> conv -> measure 6`` (32 quantum and 650 head parameters).

Both run on two protocols: the Figure 2 benchmark (offsets ``[-8, 8]`` per
axis for training and testing, suffix ``_train8``) and the
translation-generalization split (offsets ``[-2, 2]`` for training and
``[-8, 8]`` for testing, suffix ``_train2``). ``hur_su4_l1_train2`` is the
amplitude-encoded reference of ``hur_su4_l1_frqi_train2``, and
``pcsqcnn_fq2_ql3_amplitude_reduced_train8`` and ``..._train2`` are the
amplitude-encoded PCS-QCNN with the readout summed over positions (36,914
parameters) on both protocols; the other
references are the Figure 2 runs (``pcsqcnn_architecture_sweep_full_readout/
fq2_ql3``, ``hur_qcnn_translated_mnist/su4_l1``) and
``translation_generalization/pcsqcnn_fq2_ql3_train2``.

Everything else follows the Figure 2 protocol: 1000 training images per
class, digits resized to 16 x 16 on a 32 x 32 canvas, Adam with the hybrid
learning rate, uniform initialization on ``[0, 2 pi)``, 2000 epochs,
train/test batch sizes 256/1600, test evaluation every 10 epochs, and seeds
0, 1, 2.
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

DEFAULT_ENCODING_SWAP_ROOT_DIRECTORY = "encoding_swap"


@dataclass(frozen=True)
class EncodingSwapDefaults(ArticleTrainingDefaults):
    samples_per_class: int = 1000
    image_size: int = 32
    scaled_image_size: int | None = 16
    max_offset: int = 8
    test_max_offset: int | None = None
    train_batch_size: int = 256
    test_batch_size: int = 1600
    num_epochs: int = 2000
    seed_count: int = 3
    feature_qubits: int = 2
    quantum_layers: int = 3
    brightness_range: tuple[float, float] = (0.0, math.pi)
    root_directory: str = DEFAULT_ENCODING_SWAP_ROOT_DIRECTORY


DEFAULT_ENCODING_SWAP_DEFAULTS = EncodingSwapDefaults()


def _pcsqcnn_spec(*, encoding: str, reduced: bool = False) -> Callable[[EncodingSwapDefaults], ModelSpec]:
    def build(defaults: EncodingSwapDefaults) -> ModelSpec:
        return build_pcsqcnn_model_spec(
            defaults,
            image_size=defaults.image_size,
            feature_qubits=defaults.feature_qubits,
            quantum_layers=defaults.quantum_layers,
            brightness_range=defaults.brightness_range,
            reduce_readout_to_feature_distribution=reduced,
            encoding=encoding,
        )

    return build


def _hur_spec(*, encoding: str) -> Callable[[EncodingSwapDefaults], ModelSpec]:
    def build(defaults: EncodingSwapDefaults) -> ModelSpec:
        return build_hur_qcnn_model_spec(
            defaults,
            image_size=defaults.image_size,
            filters=1,
            encoding=encoding,
            brightness_range=defaults.brightness_range,
        )

    return build


# run name -> (model spec builder, training offset); offsets below the test offset 8 add a test split
RUNS: dict[str, tuple[Callable[[EncodingSwapDefaults], ModelSpec], int]] = {
    "pcsqcnn_fq2_ql3_amplitude_train8": (_pcsqcnn_spec(encoding="amplitude"), 8),
    "hur_su4_l1_frqi_train8": (_hur_spec(encoding="frqi"), 8),
    "pcsqcnn_fq2_ql3_amplitude_train2": (_pcsqcnn_spec(encoding="amplitude"), 2),
    "hur_su4_l1_frqi_train2": (_hur_spec(encoding="frqi"), 2),
    "hur_su4_l1_train2": (_hur_spec(encoding="amplitude"), 2),
    "pcsqcnn_fq2_ql3_amplitude_reduced_train8": (_pcsqcnn_spec(encoding="amplitude", reduced=True), 8),
    "pcsqcnn_fq2_ql3_amplitude_reduced_train2": (_pcsqcnn_spec(encoding="amplitude", reduced=True), 2),
}
TEST_MAX_OFFSET = 8

# Warm-started continuations, trained only when named: run name -> (model spec builder, training offset,
# run whose same-seed final checkpoint initializes the weights, epochs of this stage). The checkpoints hold
# weights only, so Adam restarts from fresh moments and the history counts epochs from one again.
WARM_RUNS: dict[str, tuple[Callable[[EncodingSwapDefaults], ModelSpec], int, str, int]] = {
    "hur_su4_l1_frqi_train8_e6000_warm": (_hur_spec(encoding="frqi"), 8, "hur_su4_l1_frqi_train8", 4000),
}


def iter_encoding_swap_configs(
    defaults: EncodingSwapDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
) -> tuple[AutoTrainingConfig, ...]:
    """Return one training config per run; ``runs`` restricts the family by name.

    The default family holds ``RUNS``; the continuations in ``WARM_RUNS`` run only when named.
    """

    resolved_defaults = defaults or DEFAULT_ENCODING_SWAP_DEFAULTS
    available = {**RUNS, **WARM_RUNS}
    selected = list(RUNS) if runs is None else list(runs)
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(f"Unknown encoding-swap runs {unknown}; available runs are {list(available)}.")

    configs = []
    for name in selected:
        build_spec, train_offset, *warm = available[name]
        # Figure 2 runs keep test_max_offset unset, so their data and configs match the Figure 2 runs exactly.
        run_defaults = replace(
            resolved_defaults,
            max_offset=train_offset,
            test_max_offset=None if train_offset == TEST_MAX_OFFSET else TEST_MAX_OFFSET,
        )
        if warm:
            run_defaults = replace(run_defaults, num_epochs=warm[1])
        config = build_article_auto_training_config(
            run_defaults,
            model_spec=build_spec(run_defaults),
            directory_name=str(Path(resolved_defaults.root_directory) / name),
        )
        if warm:
            init_checkpoint = Path(config.output.root) / resolved_defaults.root_directory / warm[0] / "checkpoint_final_seed{seed}.pt"
            config = replace(config, training=replace(config.training, init_checkpoint=str(init_checkpoint)))
        configs.append(config)
    return tuple(configs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PCS-QCNN and the Hur-QCNN with each other's encoding.",
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
        help="Override the number of seeds (default: 3).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Override the first seed (default: 0), for example to add seeds 1 and 2 to a seed-0 run.",
    )
    parser.add_argument(
        "--directory-suffix",
        default="",
        help="Suffix appended to every run directory, for example to train extra seeds apart before merging them.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: EncodingSwapDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved = defaults or DEFAULT_ENCODING_SWAP_DEFAULTS
    if args.seed_count is not None:
        resolved = replace(resolved, seed_count=args.seed_count)
    if args.base_seed is not None:
        resolved = replace(resolved, base_seed=args.base_seed)
    configs = iter_encoding_swap_configs(resolved, runs=args.runs)
    if args.directory_suffix:
        configs = tuple(
            replace(config, output=replace(config.output, directory_name=config.output.directory_name + args.directory_suffix))
            for config in configs
        )
    run_auto_training_manifest_tasks(
        configs,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
