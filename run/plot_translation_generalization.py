"""Render the translation-generalization experiment.

Models trained on offsets ``[-2, 2]`` (``prepare_translation_generalization_data.py``)
are compared with the same models trained on the benchmark offsets
``[-8, 8]``. The script reads the saved runs and
``translation_generalization/offset_accuracy.pt``
(``evaluate_translation_offsets.py``) and writes:

- ``translation_generalization_radius.pdf``: test accuracy versus shift radius
  ``max(|dy|, |dx|)`` of centred test digits, mean over seeds with the seed
  range as a band; the training offsets of the small-offset runs are shaded;
- ``translation_generalization_heatmaps.pdf``: seed-mean accuracy at every
  offset ``(dy, dx)``, one panel per model, with the ``[-2, 2]^2`` training
  window outlined;
- ``translation_generalization_curves.pdf``: test accuracy on the benchmark
  test split (offsets ``[-8, 8]``) during training;
- ``translation_generalization_encodings.pdf``: test accuracy versus shift
  radius for the PCS-QCNN (FRQI with full or summed readout, amplitude with
  full readout), the FRQI sp-QCNN, and the Hur-QCNN (one circuit on
  amplitude or FRQI, eight circuits on amplitude), trained on ``[-8, 8]``
  (left) and ``[-2, 2]`` (right), from the translation-generalization,
  encoding-swap, and sp-QCNN runs;
- ``translation_generalization_table.md`` and ``.tex``: parameters, final
  test accuracy, and the offset-resolved accuracy inside and outside the
  ``[-2, 2]^2`` window.

Missing runs are skipped with a warning.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import sys

import torch

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    configure_matplotlib_pdf_fonts,
    load_saved_parameter_stats_line,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)
from qcnn.json_artifacts import load_payload, resolve_payload_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
OFFSET_ACCURACY_PATH = "translation_generalization/offset_accuracy.pt"
RADIUS_FILENAME = "translation_generalization_radius.pdf"
HEATMAP_FILENAME = "translation_generalization_heatmaps.pdf"
CURVES_FILENAME = "translation_generalization_curves.pdf"
ENCODINGS_FILENAME = "translation_generalization_encodings.pdf"
TABLE_STEM = "translation_generalization_table"
TRAIN_WINDOW = 2
MAX_RADIUS = 8
DEFAULT_EPOCH_GROUP_SIZE = 50
_PARAMETER_TOTALS_PATTERN = re.compile(r"\bQ(\d+) C(\d+)$")


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    directory: str
    train_offset: int
    color: str

    @property
    def linestyle(self) -> str:
        return "-" if self.train_offset == TRAIN_WINDOW else "--"


MODELS: tuple[Model, ...] = (
    Model("pcs_train2", "PCS-QCNN, train ±2", "translation_generalization/pcsqcnn_fq2_ql3_train2", 2, "C0"),
    Model(
        "pcs_reduced_train2",
        "PCS-QCNN reduced readout, train ±2",
        "translation_generalization/pcsqcnn_fq2_ql3_reduced_train2",
        2,
        "C2",
    ),
    Model("hur8_train2", "Hur-QCNN x8, train ±2", "translation_generalization/hur_ch8_su4_l1_train2", 2, "mediumvioletred"),
    Model("pcs_train8", "PCS-QCNN, train ±8", "pcsqcnn_architecture_sweep_full_readout/fq2_ql3", 8, "C0"),
    Model(
        "pcs_reduced_train8",
        "PCS-QCNN reduced readout, train ±8",
        "translation_generalization/pcsqcnn_fq2_ql3_reduced_train8",
        8,
        "C2",
    ),
    Model("hur8_train8", "Hur-QCNN x8, train ±8", "hur_qcnn_translated_mnist/ch8_su4_l1", 8, "mediumvioletred"),
)


@dataclass(frozen=True)
class Architecture:
    """One model of the encoding/readout comparison; its offset runs are ``{key}_train8`` and ``{key}_train2``."""

    key: str
    label: str
    color: str
    linestyle: str
    marker: str


ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture("pcs", "PCS-QCNN, FRQI, full readout", "C0", "-", "o"),
    Architecture("pcs_reduced", "PCS-QCNN, FRQI, summed readout", "C2", "-", "s"),
    Architecture("pcs_amplitude", "PCS-QCNN, amplitude, full readout", "C1", "-", "D"),
    Architecture("sp_qcnn", "sp-QCNN, FRQI, orbit readout", "C3", "-.", "X"),
    Architecture("hur1", "HKP ×1, amplitude", "C4", "--", "^"),
    Architecture("hur1_frqi", "HKP ×1, FRQI", "C8", "--", "v"),
    Architecture("hur8", "HKP ×8, amplitude", "C6", "--", "<"),
)


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _require_matplotlib():
    import matplotlib

    configure_matplotlib_pdf_fonts(matplotlib)
    import matplotlib.pyplot as plt

    return plt


def radius_accuracy(accuracy: torch.Tensor) -> torch.Tensor:
    """Average per-offset accuracy ``[seeds, 17, 17]`` over rings of equal ``max(|dy|, |dx|)``."""

    size = accuracy.shape[-1]
    center = size // 2
    coordinates = torch.arange(size) - center
    radius = torch.maximum(coordinates.abs()[:, None], coordinates.abs()[None, :])
    return torch.stack([accuracy[:, radius == r].mean(dim=-1) for r in range(center + 1)], dim=-1)


def window_accuracy(accuracy: torch.Tensor, *, inside: bool) -> torch.Tensor:
    """Per-seed mean accuracy inside (or outside) the ``[-2, 2]^2`` training window."""

    size = accuracy.shape[-1]
    coordinates = torch.arange(size) - size // 2
    radius = torch.maximum(coordinates.abs()[:, None], coordinates.abs()[None, :])
    mask = radius <= TRAIN_WINDOW if inside else radius > TRAIN_WINDOW
    return accuracy[:, mask].mean(dim=-1)


def plot_radius(models: Sequence[Model], offset_runs: dict, *, figsize=ARTICLE_PANEL_FIGSIZE):
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(figsize[0] + 2.6, figsize[1]))
    ax.axvspan(-0.5, TRAIN_WINDOW + 0.5, color="0.9", zorder=0, label="training offsets of the ±2 runs")
    for model in models:
        rings = radius_accuracy(offset_runs[model.key]["accuracy"])
        radii = list(range(rings.shape[-1]))
        mean = rings.mean(dim=0).numpy()
        lower, upper = rings.min(dim=0).values.numpy(), rings.max(dim=0).values.numpy()
        ax.fill_between(radii, lower, upper, color=model.color, alpha=0.18, lw=0)
        ax.plot(radii, mean, color=model.color, linestyle=model.linestyle, marker="o", markersize=3, label=model.label)
    ax.set_xlabel(r"Shift radius $\max(|\Delta y|, |\Delta x|)$ [px]")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(-0.5, rings.shape[-1] - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", framealpha=0.85)
    figure.tight_layout()
    return figure


def plot_encoding_panels(offset_runs: dict, *, figsize=ARTICLE_PANEL_FIGSIZE):
    """Two panels of accuracy versus shift radius: training offsets [-8, 8] (left) and [-2, 2] (right)."""

    plt = _require_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1] + 0.9), sharey=True)
    handles = []
    for ax, (suffix, title) in zip(
        axes, (("train8", r"(a) training offsets $[-8,8]^2$"), ("train2", r"(b) training offsets $[-2,2]^2$"))
    ):
        if suffix == "train2":
            ax.axvspan(-0.5, TRAIN_WINDOW + 0.5, color="0.9", zorder=0)
        for architecture in ARCHITECTURES:
            key = f"{architecture.key}_{suffix}"
            if key not in offset_runs:
                continue
            rings = radius_accuracy(offset_runs[key]["accuracy"])
            radii = list(range(rings.shape[-1]))
            ax.fill_between(
                radii,
                rings.min(dim=0).values.numpy(),
                rings.max(dim=0).values.numpy(),
                color=architecture.color,
                alpha=0.15,
                lw=0,
            )
            (line,) = ax.plot(
                radii,
                rings.mean(dim=0).numpy(),
                color=architecture.color,
                linestyle=architecture.linestyle,
                marker=architecture.marker,
                markersize=3.5,
                label=architecture.label,
            )
            if suffix == "train8":
                handles.append(line)
        ax.set_title(title, fontsize="medium")
        ax.set_xlabel(r"Shift radius $\max(|\Delta y|, |\Delta x|)$ [px]")
        ax.set_xlim(-0.5, MAX_RADIUS + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Test accuracy")
    figure.legend(handles=handles, loc="lower center", ncols=2, fontsize="small", frameon=False)
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return figure


def plot_heatmaps(models: Sequence[Model], offset_runs: dict):
    plt = _require_matplotlib()
    from matplotlib.patches import Rectangle

    columns = 3
    rows = -(-len(models) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns + 0.8, 2.9 * rows), squeeze=False)
    image = None
    for ax, model in zip(axes.flat, models):
        accuracy = offset_runs[model.key]["accuracy"].mean(dim=0).numpy()
        image = ax.imshow(accuracy, origin="lower", extent=(-8.5, 8.5, -8.5, 8.5), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.add_patch(
            Rectangle(
                (-TRAIN_WINDOW - 0.5, -TRAIN_WINDOW - 0.5),
                2 * TRAIN_WINDOW + 1,
                2 * TRAIN_WINDOW + 1,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                linestyle="-" if model.train_offset == TRAIN_WINDOW else ":",
            )
        )
        ax.set_title(f"{model.label}\nmean {float(accuracy.mean()):.3f}", fontsize="small")
        ax.set_xlabel(r"$\Delta x$ [px]")
        ax.set_ylabel(r"$\Delta y$ [px]")
    for ax in list(axes.flat)[len(models) :]:
        ax.axis("off")
    figure.tight_layout()
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="Test accuracy")
    return figure


def plot_curves(models: Sequence[Model], histories: dict, *, epoch_group_size: int, figsize=ARTICLE_PANEL_FIGSIZE):
    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=(figsize[0] + 2.6, figsize[1]))
    for model in models:
        runs = histories[model.key]
        epoch = list(runs[0].test_epoch)
        samples = [[float(history.test_metrics["accuracy"][index]) for history in runs] for index in range(len(epoch))]
        summary = summarize_temporal_samples(
            epoch, samples, epoch_group_size=epoch_group_size, lower_percentile=25.0, upper_percentile=75.0
        )
        plot_temporal_summary(ax, summary=summary, color=model.color, label=model.label, linestyle=model.linestyle)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (offsets ±8)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", framealpha=0.85)
    figure.tight_layout()
    return figure


def _percent(values: Sequence[float]) -> str:
    mean = 100.0 * statistics.fmean(values)
    spread = 100.0 * statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} ± {spread:.2f}"


def build_rows(models: Sequence[Model], histories: dict, totals: dict, offset_runs: dict) -> list[dict[str, str]]:
    rows = []
    for model in models:
        final = [float(history.test_metrics["accuracy"][-1]) for history in histories[model.key]]
        quantum, classical = totals[model.key]
        row = {
            "model": model.label,
            "train": f"±{model.train_offset}",
            "parameters": f"{quantum + classical:,}",
            "final": _percent(final),
            "inside": "–",
            "outside": "–",
        }
        if model.key in offset_runs:
            accuracy = offset_runs[model.key]["accuracy"]
            row["inside"] = _percent(window_accuracy(accuracy, inside=True).tolist())
            row["outside"] = _percent(window_accuracy(accuracy, inside=False).tolist())
        rows.append(row)
    return rows


def render_markdown(rows: Sequence[dict[str, str]]) -> str:
    lines = [
        "| Model | Train offsets | Params | Final test acc. (%), offsets ±8 | Acc. inside ±2 window (%) "
        "| Acc. outside ±2 window (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {row['model']} | {row['train']} | {row['parameters']} | {row['final']} | {row['inside']} | {row['outside']} |"
        for row in rows
    ]
    lines.append("")
    lines.append(
        "Final test accuracy is measured on the benchmark test split (offsets uniform in [-8, 8]^2) at the last "
        "epoch. The window columns shift centred test digits to every offset in [-8, 8]^2 and average the "
        "accuracy over offsets with max(|dy|, |dx|) <= 2 (inside, 25 offsets) and > 2 (outside, 264 offsets). "
        "Means and sample standard deviations run over seeds."
    )
    return "\n".join(lines) + "\n"


def render_latex(rows: Sequence[dict[str, str]]) -> str:
    plus_minus = r"\pm"

    def cell(text: str) -> str:
        return "$" + text.replace("±", plus_minus) + "$" if "±" in text else text.replace(",", "{,}")

    def label(text: str) -> str:
        return text.replace("±", "$" + plus_minus + "$")

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Model & Train & Params & Test acc.\ (\%) & Inside $\pm2$ (\%) & Outside $\pm2$ (\%) \\",
        r"\hline",
    ]
    end_of_row = r" \\"
    lines += [
        " & ".join(
            (
                label(row["model"]),
                label(row["train"]),
                cell(row["parameters"]),
                cell(row["final"]),
                cell(row["inside"]),
                cell(row["outside"]),
            )
        )
        + end_of_row
        for row in rows
    ]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the translation-generalization experiment.")
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
    totals: dict = {}
    for model in MODELS:
        directory = artifacts_root / model.directory
        try:
            histories[model.key] = load_saved_training_histories(directory)
            match = _PARAMETER_TOTALS_PATTERN.search(load_saved_parameter_stats_line(directory).strip())
            totals[model.key] = (int(match.group(1)), int(match.group(2)))
        except Exception as exc:
            _warn(f"Skipping {model.label}: could not load {directory}: {exc}")
    available = [model for model in MODELS if model.key in histories]
    offset_path = artifacts_root / OFFSET_ACCURACY_PATH
    offset_path = resolve_payload_path(offset_path)
    offset_runs = load_payload(offset_path)["runs"] if offset_path.is_file() else {}
    if not offset_runs:
        _warn(f"No offset accuracies at {offset_path}; run evaluate_translation_offsets.py first.")
    with_offsets = [model for model in MODELS if model.key in offset_runs]

    written: list[Path] = []
    if with_offsets:
        for filename, figure in (
            (RADIUS_FILENAME, plot_radius(with_offsets, offset_runs)),
            (HEATMAP_FILENAME, plot_heatmaps(with_offsets, offset_runs)),
            (ENCODINGS_FILENAME, plot_encoding_panels(offset_runs)),
        ):
            figure.savefig(output_dir / filename)
            plt.close(figure)
            written.append(output_dir / filename)
    if available:
        figure = plot_curves(available, histories, epoch_group_size=args.epoch_group_size)
        figure.savefig(output_dir / CURVES_FILENAME)
        plt.close(figure)
        written.append(output_dir / CURVES_FILENAME)
        rows = build_rows(available, histories, totals, offset_runs)
        markdown = render_markdown(rows)
        (output_dir / f"{TABLE_STEM}.md").write_text(markdown, encoding="utf-8")
        (output_dir / f"{TABLE_STEM}.tex").write_text(render_latex(rows), encoding="utf-8")
        written += [output_dir / f"{TABLE_STEM}.md", output_dir / f"{TABLE_STEM}.tex"]
        print(markdown)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
