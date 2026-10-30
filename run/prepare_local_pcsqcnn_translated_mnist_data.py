"""Train the local-multiplexer PCS-QCNN on both translated-MNIST protocols.

The model is the Figure 2 PCS-QCNN (``Q = 3``, ``n_f = 2``, FRQI, full
readout) with ``multiplexer_kind="local_3x3"``:
every multiplexer holds two U(4) coins per pooling branch shared across the
Fourier modes, so each quantum convolution is a unitary, translation-
equivariant kernel supported inside ``3 x 3`` on its active periodic grid.
The network holds 288 quantum and 2,570 head parameters (2,858 in total,
against 39,434 for the general multiplexer). Training uses the highest
float32 matmul precision; reduced CUDA precision breaks the unitarity and the
``3 x 3`` support.

The runs are:

- ``pcsqcnn_local_fq2_ql3_train8``: the Figure 2 benchmark, offsets
  ``[-8, 8]`` per axis for training and testing;
- ``pcsqcnn_local_fq2_ql3_train2``: the translation-generalization split,
  offsets ``[-2, 2]`` for training and ``[-8, 8]`` for testing;
- ``pcsqcnn_local_fq2_ql3_reduced_train8`` and ``..._reduced_train2``: the
  same two protocols with the readout summed over positions
  (``reduce_readout_to_feature_distribution=True``), which makes the logits
  exactly invariant to 4-pixel shifts (288 quantum and 50 head parameters);
- ``pcsqcnn_local_fq2_ql3_train8_e4000_warm``: ``pcsqcnn_local_fq2_ql3_train8``
  continued for another 2000 epochs (4000 in total) from the final checkpoint
  of the same seed. The checkpoints hold weights only, so Adam restarts from
  fresh moments (a warm restart, not an exact continuation) and the history
  counts epochs from one again;
- ``pcsqcnn_local_fq2_ql3_train8_e6000_warm``: the ``e4000_warm`` run
  continued for another 2000 epochs (6000 in total) the same way.

``--seed-count`` overrides the number of seeds, for example for single-seed
checks; ``--num-epochs`` overrides the training length, and runs whose length
differs from the protocol's 2000 epochs get the suffix ``_e<epochs>``, for
example ``pcsqcnn_local_fq2_ql3_reduced_train8_e4000``. When a warm restart
and its predecessor are selected together, both epoch and directory suffixes
apply to their checkpoint paths. A standalone warm restart reads its
unselected predecessor from the original protocol directory.

Everything else follows the Figure 2 protocol: 1000 training images per
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
from qcnn.article_training import (
    ArticleTrainingDefaults,
    build_article_auto_training_config,
    build_pcsqcnn_model_spec,
)
from qcnn.automation import run_auto_training_manifest_tasks
from qcnn.model_spec import ModelSpec

DEFAULT_LOCAL_PCSQCNN_ROOT_DIRECTORY = "local_pcsqcnn_translated_mnist"


@dataclass(frozen=True)
class LocalPcsqcnnTranslatedMnistDefaults(ArticleTrainingDefaults):
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
    feature_qubits: int = 2
    quantum_layers: int = 3
    brightness_range: tuple[float, float] = (0.0, math.pi)
    root_directory: str = DEFAULT_LOCAL_PCSQCNN_ROOT_DIRECTORY


DEFAULT_LOCAL_PCSQCNN_TRANSLATED_MNIST_DEFAULTS = LocalPcsqcnnTranslatedMnistDefaults()

# run name -> (training offset, reduced readout, run whose final checkpoints initialize the weights);
# offsets below the test offset 8 add a test split
RUNS: dict[str, tuple[int, bool, str | None]] = {
    "pcsqcnn_local_fq2_ql3_train8": (8, False, None),
    "pcsqcnn_local_fq2_ql3_train2": (2, False, None),
    "pcsqcnn_local_fq2_ql3_reduced_train8": (8, True, None),
    "pcsqcnn_local_fq2_ql3_reduced_train2": (2, True, None),
    "pcsqcnn_local_fq2_ql3_train8_e4000_warm": (8, False, "pcsqcnn_local_fq2_ql3_train8"),
    "pcsqcnn_local_fq2_ql3_train8_e6000_warm": (8, False, "pcsqcnn_local_fq2_ql3_train8_e4000_warm"),
}
TEST_MAX_OFFSET = 8
PROTOCOL_NUM_EPOCHS = 2000


def build_local_pcsqcnn_model_spec(defaults: LocalPcsqcnnTranslatedMnistDefaults, *, reduced: bool = False) -> ModelSpec:
    return build_pcsqcnn_model_spec(
        defaults,
        image_size=defaults.image_size,
        feature_qubits=defaults.feature_qubits,
        quantum_layers=defaults.quantum_layers,
        brightness_range=defaults.brightness_range,
        reduce_readout_to_feature_distribution=reduced,
        multiplexer_kind="local_3x3",
    )


def _directory_name(name: str, defaults: LocalPcsqcnnTranslatedMnistDefaults) -> str:
    if defaults.num_epochs == PROTOCOL_NUM_EPOCHS:
        return name
    return f"{name}_e{defaults.num_epochs}"


def iter_local_pcsqcnn_training_configs(
    defaults: LocalPcsqcnnTranslatedMnistDefaults | None = None,
    *,
    runs: Sequence[str] | None = None,
    directory_suffix: str = "",
) -> tuple[AutoTrainingConfig, ...]:
    """Build a run family, keeping selected warm-start dependencies together.

    A predecessor selected in the same invocation uses its effective output
    path. An unselected predecessor is read from its canonical protocol path,
    allowing a standalone warm restart from an existing baseline.
    """

    resolved_defaults = defaults or DEFAULT_LOCAL_PCSQCNN_TRANSLATED_MNIST_DEFAULTS
    selected = list(RUNS) if runs is None else list(runs)
    unknown = [name for name in selected if name not in RUNS]
    if unknown:
        raise ValueError(f"Unknown local PCS-QCNN runs {unknown}; available runs are {list(RUNS)}.")

    directories = {
        name: str(Path(resolved_defaults.root_directory) / (_directory_name(name, resolved_defaults) + directory_suffix))
        for name in selected
    }
    configs = []
    for name in selected:
        train_offset, reduced, init_from = RUNS[name]
        # Figure 2 runs keep test_max_offset unset, so their data match the Figure 2 runs exactly.
        run_defaults = replace(
            resolved_defaults,
            max_offset=train_offset,
            test_max_offset=None if train_offset == TEST_MAX_OFFSET else TEST_MAX_OFFSET,
        )
        config = build_article_auto_training_config(
            run_defaults,
            model_spec=build_local_pcsqcnn_model_spec(run_defaults, reduced=reduced),
            directory_name=directories[name],
        )
        if init_from is not None:
            source_directory = directories.get(init_from, str(Path(resolved_defaults.root_directory) / init_from))
            init_checkpoint = Path(config.output.root) / source_directory / "checkpoint_final_seed{seed}.pt"
            config = replace(config, training=replace(config.training, init_checkpoint=str(init_checkpoint)))
        configs.append(config)
    return tuple(configs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local-multiplexer PCS-QCNN on both translated-MNIST protocols.",
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
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override the number of training epochs (default: 2000); other lengths get an _e<epochs> suffix.",
    )
    return parser.parse_args(argv)


def main(
    *,
    defaults: LocalPcsqcnnTranslatedMnistDefaults | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    resolved = defaults or DEFAULT_LOCAL_PCSQCNN_TRANSLATED_MNIST_DEFAULTS
    if args.seed_count is not None:
        resolved = replace(resolved, seed_count=args.seed_count)
    if args.base_seed is not None:
        resolved = replace(resolved, base_seed=args.base_seed)
    if args.num_epochs is not None:
        resolved = replace(resolved, num_epochs=args.num_epochs)
    configs = iter_local_pcsqcnn_training_configs(
        resolved, runs=args.runs, directory_suffix=args.directory_suffix,
    )
    run_auto_training_manifest_tasks(
        configs,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
