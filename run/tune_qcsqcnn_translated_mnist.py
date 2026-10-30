"""Successive-halving search for the FRQI QCS-QCNN learning rate and initialization.

The search covers the fully matched QCS-QCNN (``Q = 3``, ``n_f = 2``,
``K = 8``; 39,434 parameters) on the translated-MNIST benchmark of Figure 2.
Selection never reads the test split:

- the seed-0 translated-MNIST training set (1000 images per class) is split
  into 900 training and 100 validation images per class with
  ``stratified_validation_split``;
- the grid is Adam learning rate ``{1e-3, 3e-3, 1e-2, 3e-2}`` times uniform
  initialization scale ``{0.05, 0.5, 2 pi}`` of the Pauli coefficients;
- successive halving trains all 12 configurations to 300 epochs, continues
  the best 4 to 1000 epochs and the best 2 to 2000 epochs, ranking by the mean
  validation accuracy of the last 5 evaluations (every 10 epochs);
- everything else follows the Figure 2 protocol (batch size 256, exact
  probabilities, TF32 matmuls on CUDA).

The search itself lives in ``qcnn.tuning``. Every configuration is
checkpointed after each rung, so an interrupted search resumes where it
stopped. ``search_summary.json`` is written last and marks the search as
complete; the selected setting is then trained on the full training set with
``prepare_qcsqcnn_translated_mnist_data.py --learning-rate ... --generator-init-scale ...``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

from qcnn.article_training import build_qcsqcnn_model_spec
from qcnn.model_spec import ModelSpec
from qcnn.tuning import (
    SearchConfig,
    SearchDefaults,
    run_search,
    search_grid,
    selection_score,
    successive_halving,
)

__all__ = [
    "QcsQcnnTuningDefaults",
    "SearchConfig",
    "build_model_spec",
    "main",
    "search_grid",
    "selection_score",
    "successive_halving",
]


@dataclass(frozen=True)
class QcsQcnnTuningDefaults(SearchDefaults):
    feature_qubits: int = 2
    quantum_layers: int = 3
    sublayers: int = 8
    brightness_range: tuple[float, float] = (0.0, math.pi)
    output_directory: str = "qcsqcnn_tuning/fq2_ql3_sl8"


DEFAULT_QCSQCNN_TUNING_DEFAULTS = QcsQcnnTuningDefaults()


def build_model_spec(defaults: QcsQcnnTuningDefaults, config: SearchConfig) -> ModelSpec:
    return build_qcsqcnn_model_spec(
        replace(defaults, multiplexer_init_scale=config.init_scale),
        image_size=defaults.image_size,
        feature_qubits=defaults.feature_qubits,
        quantum_layers=defaults.quantum_layers,
        sublayers=defaults.sublayers,
        brightness_range=defaults.brightness_range,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the FRQI QCS-QCNN learning rate and initialization scale.")
    parser.add_argument("--rebuild", action="store_true", help="Discard saved search states and start over.")
    return parser.parse_args(argv)


def main(*, defaults: QcsQcnnTuningDefaults | None = None, argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resolved = defaults or DEFAULT_QCSQCNN_TUNING_DEFAULTS
    run_search(
        resolved,
        build_model_spec=lambda config: build_model_spec(resolved, config),
        model_fields={
            "model": "QCSQCNN",
            "feature_qubits": resolved.feature_qubits,
            "quantum_layers": resolved.quantum_layers,
            "sublayers": resolved.sublayers,
        },
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
