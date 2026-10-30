"""Successive-halving search for the Hur-QCNN learning rate and initialization.

The search covers the Hur-QCNN baseline with one SU(4) filter per layer
(``su4_l1``; 32 quantum and 330 head parameters) on the translated-MNIST
benchmark of Figure 2, with the same procedure as the QCS-QCNN search in
``tune_qcsqcnn_translated_mnist.py``. Selection never reads the test split:

- the seed-0 translated-MNIST training set (1000 images per class) is split
  into 900 training and 100 validation images per class with
  ``stratified_validation_split``;
- the grid is Adam learning rate ``{1e-3, 3e-3, 1e-2, 3e-2}`` times uniform
  initialization scale ``{0.05, 0.5, 2 pi}`` of the gate angles;
- successive halving trains all 12 configurations to 300 epochs, continues
  the best 4 to 1000 epochs and the best 2 to 2000 epochs, ranking by the mean
  validation accuracy of the last 5 evaluations (every 10 epochs);
- everything else follows the Figure 2 protocol (batch size 256, exact
  probabilities, TF32 matmuls on CUDA).

The search itself lives in ``qcnn.tuning`` and resumes after an interruption.
``search_summary.json`` is written last and marks the search as complete; the
selected setting is then trained on the full training set with
``prepare_hur_qcnn_translated_mnist_data.py --learning-rate ... --angle-init-scale ...``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace

from qcnn.article_training import build_hur_qcnn_model_spec
from qcnn.model_spec import ModelSpec
from qcnn.tuning import SearchConfig, SearchDefaults, run_search


@dataclass(frozen=True)
class HurQcnnTuningDefaults(SearchDefaults):
    filters: int = 1
    output_directory: str = "hur_qcnn_tuning/su4_l1"


DEFAULT_HUR_QCNN_TUNING_DEFAULTS = HurQcnnTuningDefaults()


def build_model_spec(defaults: HurQcnnTuningDefaults, config: SearchConfig) -> ModelSpec:
    return build_hur_qcnn_model_spec(
        replace(defaults, multiplexer_init_scale=config.init_scale),
        image_size=defaults.image_size,
        filters=defaults.filters,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the Hur-QCNN learning rate and initialization scale.")
    parser.add_argument("--rebuild", action="store_true", help="Discard saved search states and start over.")
    return parser.parse_args(argv)


def main(*, defaults: HurQcnnTuningDefaults | None = None, argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resolved = defaults or DEFAULT_HUR_QCNN_TUNING_DEFAULTS
    run_search(
        resolved,
        build_model_spec=lambda config: build_model_spec(resolved, config),
        model_fields={"model": "HurQCNN", "filters": resolved.filters},
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
