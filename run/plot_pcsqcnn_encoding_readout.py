"""Training curves of the PCS-QCNN with both encodings and both readouts.

The PCS-QCNN (``Q = 3``, ``n_f = 2``) is shown with the FRQI-like and the
amplitude encoding, each with the full readout and with the readout summed
over positions, under the two translated-MNIST protocols: training offsets
``[-8, 8]`` (P8) and ``[-2, 2]`` (P2), both tested on the P8 test split.
The runs come from the Figure 2, translation-generalization, and
encoding-swap experiments. The script writes
``pcsqcnn_encoding_readout_accuracy.pdf``/``.png`` in the style of the
article's Figure 2 panels: test (solid) and train (dashed) accuracy, mean
over seeds and 50-epoch bins with the pooled 25-75 percentile band, and a
reference line at each final mean test accuracy with its value at the right
edge.

Missing runs are skipped with a warning.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

from qcnn import (
    configure_matplotlib_pdf_fonts,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
OUTPUT_STEM = "pcsqcnn_encoding_readout_accuracy"
DEFAULT_EPOCH_GROUP_SIZE = 50
FIGSIZE = (12.0, 4.6)
PANEL_TITLES = {8: r"(a) training offsets $[-8,8]^2$", 2: r"(b) training offsets $[-2,2]^2$"}
# Minimum vertical distance between neighbouring final-value labels, as a fraction of the y range.
LABEL_MIN_GAP = 0.045


@dataclass(frozen=True)
class Model:
    label: str
    color: str
    directories: dict[int, str]


# Colours of these models in the shift-radius figure; the amplitude summed readout takes an unused one.
MODELS: tuple[Model, ...] = (
    Model(
        "FRQI, full readout",
        "C0",
        {8: "pcsqcnn_architecture_sweep_full_readout/fq2_ql3", 2: "translation_generalization/pcsqcnn_fq2_ql3_train2"},
    ),
    Model(
        "FRQI, summed readout",
        "C2",
        {
            8: "translation_generalization/pcsqcnn_fq2_ql3_reduced_train8",
            2: "translation_generalization/pcsqcnn_fq2_ql3_reduced_train2",
        },
    ),
    Model(
        "amplitude, full readout",
        "C1",
        {8: "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8", 2: "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train2"},
    ),
    Model(
        "amplitude, summed readout",
        "C5",
        {
            8: "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train8",
            2: "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train2",
        },
    ),
)


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


def spread_label_positions(levels: Sequence[float], *, min_gap: float) -> list[float]:
    """Return label heights near ``levels``, at least ``min_gap`` apart, in the input order."""

    order = sorted(range(len(levels)), key=lambda index: levels[index])
    positions = [float(levels[index]) for index in order]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + min_gap)
    # Centre each pushed group on its levels so no label drifts far from its line.
    shift = (sum(positions) - sum(sorted(levels))) / len(positions) if positions else 0.0
    positions = [position - shift for position in positions]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + min_gap)
    result = [0.0] * len(levels)
    for rank, index in enumerate(order):
        result[index] = positions[rank]
    return result


def plot_accuracy(histories: dict, *, epoch_group_size: int):
    """Plot every loaded ``(model label, offset)`` history, one panel per protocol."""

    plt = _require_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    figure.patch.set_facecolor("white")
    finals: dict[int, list[tuple[float, str]]] = {offset: [] for offset in PANEL_TITLES}
    epochs: list[float] = []
    for ax, offset in zip(axes, PANEL_TITLES, strict=True):
        ax.set_facecolor("white")
        for model in MODELS:
            runs = histories.get((model.label, offset))
            if runs is None:
                continue
            final_level = _summary(runs, "test", epoch_group_size=1, band=False).mean[-1]
            test = _summary(runs, "test", epoch_group_size=epoch_group_size)
            train = _summary(runs, "train", epoch_group_size=epoch_group_size)
            plot_temporal_summary(ax, summary=test, color=model.color, band_alpha=0.30, limit_value=final_level)
            plot_temporal_summary(ax, summary=train, color=model.color, linestyle="--", linewidth=1.0, band_alpha=0.12)
            finals[offset].append((final_level, model.color))
            epochs.extend([*test.epoch, *train.epoch])
        ax.set_title(PANEL_TITLES[offset], fontsize="medium")
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlim(min(epochs), max(epochs))
    low, high = axes[0].get_ylim()
    span = high - low
    for ax, offset in zip(axes, PANEL_TITLES, strict=True):
        levels = [level for level, _ in finals[offset]]
        positions = spread_label_positions(levels, min_gap=LABEL_MIN_GAP * span)
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        for (level, _), position in zip(finals[offset], positions, strict=True):
            ax.annotate(f"{level:.4f}", xy=(1.01, position), xycoords=transform, color="black", ha="left", va="center")
    handles = [Line2D([], [], color=model.color, linewidth=2, label=model.label) for model in MODELS]
    handles += [
        Line2D([], [], color="0.35", linewidth=1.5, label="test (offsets $\\pm8$)"),
        Line2D([], [], color="0.35", linewidth=1.0, linestyle="--", label="train"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize="small")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the PCS-QCNN with both encodings and readouts under P8 and P2.")
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
        for offset, relative in model.directories.items():
            try:
                histories[(model.label, offset)] = load_saved_training_histories(artifacts_root / relative)
            except Exception as exc:
                _warn(f"Skipping {model.label}, training offsets ±{offset}: {exc}")
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
