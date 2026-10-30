"""Compare the local-multiplexer PCS-QCNN with the FRQI Hur-QCNN over 6000 epochs.

Both models load images with the FRQI encoding and train on the Figure 2
benchmark (offsets ``[-8, 8]`` for training and testing). The PCS-QCNN
(``Q = 3``, ``n_f = 2``, ``multiplexer_kind="local_3x3"``, full readout,
288 quantum parameters) runs 2000 epochs and is continued by warm restarts
at epochs 2000 and 4000; the Hur-QCNN su4_l1 of the encoding-swap experiment
runs 2000 epochs and is continued by one warm restart at epoch 2000 for
another 4000 epochs. The checkpoints hold weights only, so each restart
starts Adam from fresh moments. Every seed's stages are joined into one
6000-epoch history. The script writes
``local_pcsqcnn_vs_hur_frqi_accuracy.pdf``/``.png`` in the style of the
article's Figure 2 panels: test (solid) and train (dash-dot, dashed) accuracy,
mean over seeds and 50-epoch bins with the pooled 25-75 percentile band, and
a labelled reference line at the final mean test accuracy.

Missing runs are skipped with a warning.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    configure_matplotlib_pdf_fonts,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
OUTPUT_STEM = "local_pcsqcnn_vs_hur_frqi_accuracy"
DEFAULT_EPOCH_GROUP_SIZE = 50
# Figure 2 panel settings of run/plot_translated_mnist_baselines.py.
# The noisy curves cross the final levels inside the axes, so the level labels sit just right of them.
LIMIT_LABEL_X_AXES = 1.01
LIMIT_LABEL_Y_OFFSET_POINTS = 0
LEGEND_LOCATION = "lower right"


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    train_label: str
    color: str
    train_linestyle: str
    # warm-restart stages, oldest first; every seed of the last stage is joined with the same seed of the others
    stages: tuple[str, ...]


MODELS: tuple[Model, ...] = (
    Model(
        "pcs_local",
        "Local PCS-QCNN",
        "Train local PCS-QCNN",
        "C0",
        "-.",
        (
            "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8",
            "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e4000_warm",
            "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e6000_warm",
        ),
    ),
    Model(
        "hur_frqi",
        "HKP ×1, FRQI",
        "Train HKP ×1, FRQI",
        "C1",
        "--",
        (
            "encoding_swap/hur_su4_l1_frqi_train8",
            "encoding_swap/hur_su4_l1_frqi_train8_e6000_warm",
        ),
    ),
)


def _manifest_seeds(directory: Path) -> list[int]:
    return list(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["seeds"])


def _concatenate(first, second):
    """Join two histories; the epochs of ``second`` continue after the last epoch of ``first``."""

    offset = first.train_epoch[-1]
    return SimpleNamespace(
        train_epoch=[*first.train_epoch, *(epoch + offset for epoch in second.train_epoch)],
        test_epoch=[*first.test_epoch, *(epoch + offset for epoch in second.test_epoch)],
        train_metrics={"accuracy": [*first.train_metrics["accuracy"], *second.train_metrics["accuracy"]]},
        test_metrics={"accuracy": [*first.test_metrics["accuracy"], *second.test_metrics["accuracy"]]},
    )


def load_joined_histories(artifacts_root: Path, model: Model) -> list:
    """Return one history per seed of the last stage, joined over all stages."""

    stages = []
    for relative in model.stages:
        directory = artifacts_root / relative
        stages.append(dict(zip(_manifest_seeds(directory), load_saved_training_histories(directory), strict=True)))
    joined = []
    for seed in stages[-1]:
        combined = stages[0][seed]
        for stage in stages[1:]:
            combined = _concatenate(combined, stage[seed])
        joined.append(combined)
    return joined


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _require_matplotlib():
    import matplotlib

    configure_matplotlib_pdf_fonts(matplotlib)
    import matplotlib.pyplot as plt

    return plt


def _summary(histories, split: str, *, epoch_group_size: int, band: bool = True):
    epoch = histories[0].train_epoch if split == "train" else histories[0].test_epoch
    metrics = "test_metrics" if split == "test" else "train_metrics"
    samples = [[float(getattr(history, metrics)["accuracy"][index]) for history in histories] for index in range(len(epoch))]
    return summarize_temporal_samples(
        list(epoch),
        samples,
        epoch_group_size=epoch_group_size,
        lower_percentile=25.0,
        upper_percentile=75.0,
        compute_band=band,
    )


def plot_accuracy(histories: dict, *, epoch_group_size: int):
    """Plot every loaded model history in the style of the article's Figure 2 panels."""

    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=ARTICLE_PANEL_FIGSIZE)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    epochs: list[float] = []
    final_levels: list[float] = []
    for model in MODELS:
        runs = histories.get(model.key)
        if runs is None:
            continue
        final_level = _summary(runs, "test", epoch_group_size=1, band=False).mean[-1]
        test = _summary(runs, "test", epoch_group_size=epoch_group_size)
        train = _summary(runs, "train", epoch_group_size=epoch_group_size)
        plot_temporal_summary(
            ax,
            summary=test,
            color=model.color,
            label=model.label,
            band_alpha=0.30,
            limit_value=final_level,
            show_limit_label=True,
            limit_label_x_axes=LIMIT_LABEL_X_AXES,
            limit_label_y_offset_points=LIMIT_LABEL_Y_OFFSET_POINTS,
        )
        plot_temporal_summary(
            ax,
            summary=train,
            color=model.color,
            label=model.train_label,
            linestyle=model.train_linestyle,
            band_alpha=0.12,
        )
        epochs.extend([*test.epoch, *train.epoch])
        final_levels.append(final_level)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(epochs), max(epochs))
    low, high = ax.get_ylim()
    span = (high - low) or 1.0
    ax.set_ylim(min(low, min(final_levels) - 0.02 * span), max(high, max(final_levels) + 0.05 * span))
    ax.legend(loc=LEGEND_LOCATION, framealpha=0.8)
    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the local-multiplexer PCS-QCNN with the FRQI Hur-QCNN over 6000 epochs.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epoch-group-size", type=int, default=DEFAULT_EPOCH_GROUP_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: dict = {}
    for model in MODELS:
        try:
            histories[model.key] = load_joined_histories(artifacts_root, model)
        except Exception as exc:
            _warn(f"Skipping {model.label}: {exc}")
    if not histories:
        _warn("No runs found; nothing written.")
        return

    figure = plot_accuracy(histories, epoch_group_size=args.epoch_group_size)
    for suffix in (".pdf", ".png"):
        figure.savefig(output_dir / f"{OUTPUT_STEM}{suffix}", dpi=200)
        print(output_dir / f"{OUTPUT_STEM}{suffix}")
    plt.close(figure)


if __name__ == "__main__":
    main()
