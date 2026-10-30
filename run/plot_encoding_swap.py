"""Render the encoding-swap experiment.

The PCS-QCNN (native FRQI encoding) and the Hur-QCNN (native amplitude
encoding) are compared with the same models trained on the other encoding
(``prepare_encoding_swap_data.py``), on the Figure 2 benchmark (offsets
``[-8, 8]`` for training and testing) and on the translation-generalization
split (offsets ``[-2, 2]`` for training, ``[-8, 8]`` for testing). The native
runs are the existing Figure 2, Hur-QCNN, and translation-generalization runs
plus ``encoding_swap/hur_su4_l1_train2``. The script writes:

- ``encoding_swap_accuracy.pdf``/``.png``: test (solid, mean over seeds and
  50-epoch bins with the pooled 25-75 percentile band) and train (dashed,
  mean) accuracy during training; rows are the protocols, columns the models;
- ``encoding_swap_loss.pdf``/``.png``: the same layout for the cross-entropy
  loss on a logarithmic axis;
- ``encoding_swap_table.md`` and ``.tex``: parameters and final train and
  test accuracy.

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

from qcnn import (
    configure_matplotlib_pdf_fonts,
    load_saved_parameter_stats_line,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
ACCURACY_STEM = "encoding_swap_accuracy"
LOSS_STEM = "encoding_swap_loss"
TABLE_STEM = "encoding_swap_table"
DEFAULT_EPOCH_GROUP_SIZE = 50
FIGSIZE = (10.0, 6.8)

_PARAMETER_TOTALS_PATTERN = re.compile(r"\bQ(\d+) C(\d+)$")

MODEL_LABELS = {"pcs": r"PCS-QCNN ($Q = 3$, $n_f = 2$)", "hur": "Hur-QCNN su4_l1"}
NATIVE_ENCODING = {"pcs": "frqi", "hur": "amplitude"}
ENCODING_LABELS = {"frqi": "FRQI", "amplitude": "amplitude"}
# Slots 1 and 2 of the validated categorical reference palette.
ENCODING_COLORS = {"frqi": "#2a78d6", "amplitude": "#eb6834"}


@dataclass(frozen=True)
class Run:
    model: str
    encoding: str
    train_offset: int
    directory: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model, self.encoding, self.train_offset)

    @property
    def label(self) -> str:
        role = "native" if self.encoding == NATIVE_ENCODING[self.model] else "swapped"
        return f"{MODEL_LABELS[self.model]}, {ENCODING_LABELS[self.encoding]} ({role})"


RUNS: tuple[Run, ...] = (
    Run("pcs", "frqi", 8, "pcsqcnn_architecture_sweep_full_readout/fq2_ql3"),
    Run("pcs", "amplitude", 8, "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8"),
    Run("hur", "amplitude", 8, "hur_qcnn_translated_mnist/su4_l1"),
    Run("hur", "frqi", 8, "encoding_swap/hur_su4_l1_frqi_train8"),
    Run("pcs", "frqi", 2, "translation_generalization/pcsqcnn_fq2_ql3_train2"),
    Run("pcs", "amplitude", 2, "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train2"),
    Run("hur", "amplitude", 2, "encoding_swap/hur_su4_l1_train2"),
    Run("hur", "frqi", 2, "encoding_swap/hur_su4_l1_frqi_train2"),
)
PROTOCOL_LABELS = {8: "train ±8, test ±8", 2: "train ±2, test ±8"}


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _require_matplotlib():
    import matplotlib

    configure_matplotlib_pdf_fonts(matplotlib)
    import matplotlib.pyplot as plt

    return plt


def _summary(epoch, histories, values, *, epoch_group_size: int, band: bool):
    samples = [[float(values(history)[index]) for history in histories] for index in range(len(epoch))]
    return summarize_temporal_samples(
        list(epoch),
        samples,
        epoch_group_size=epoch_group_size,
        lower_percentile=25.0,
        upper_percentile=75.0,
        compute_band=band,
    )


def plot_metric(runs: Sequence[Run], histories: dict, *, metric: str, epoch_group_size: int):
    """Plot ``metric`` (``"accuracy"`` or ``"loss"``) for every run, one panel per protocol and model."""

    plt = _require_matplotlib()
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    for row, offset in enumerate(PROTOCOL_LABELS):
        for column, model in enumerate(MODEL_LABELS):
            ax = axes[row, column]
            for run in runs:
                if (run.model, run.train_offset) != (model, offset):
                    continue
                runs_of = histories[run.key]
                color = ENCODING_COLORS[run.encoding]
                if metric == "accuracy":
                    test_values = lambda history: history.test_metrics["accuracy"]  # noqa: E731
                    train_values = lambda history: history.train_metrics["accuracy"]  # noqa: E731
                else:
                    test_values = lambda history: history.test_loss  # noqa: E731
                    train_values = lambda history: history.train_loss  # noqa: E731
                train = _summary(runs_of[0].train_epoch, runs_of, train_values, epoch_group_size=epoch_group_size, band=False)
                test = _summary(runs_of[0].test_epoch, runs_of, test_values, epoch_group_size=epoch_group_size, band=True)
                plot_temporal_summary(ax, summary=train, color=color, linestyle="--", linewidth=1.0, show_band=False)
                plot_temporal_summary(ax, summary=test, color=color, linewidth=1.6)
            native = ENCODING_LABELS[NATIVE_ENCODING[model]]
            ax.set_title(f"{MODEL_LABELS[model]}, native {native}\n{PROTOCOL_LABELS[offset]}", fontsize="medium")
            ax.grid(True, color="0.88", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel("Epoch")
            if column == 0:
                ax.set_ylabel("Accuracy" if metric == "accuracy" else "Cross-entropy loss")
    if metric == "accuracy":
        axes[0, 0].set_ylim(0.0, 1.0)
    else:
        axes[0, 0].set_yscale("log")
    handles = [
        Line2D([], [], color=ENCODING_COLORS["frqi"], linewidth=2, label="FRQI encoding"),
        Line2D([], [], color=ENCODING_COLORS["amplitude"], linewidth=2, label="amplitude encoding"),
        Line2D([], [], color="0.35", linewidth=1.6, label="test (mean, 25-75% band)"),
        Line2D([], [], color="0.35", linewidth=1.0, linestyle="--", label="train (mean)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize="small")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def _percent(values: Sequence[float]) -> str:
    mean = 100.0 * statistics.fmean(values)
    spread = 100.0 * statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.1f} ± {spread:.1f}"


def build_rows(runs: Sequence[Run], histories: dict, totals: dict) -> list[dict[str, str]]:
    rows = []
    for run in runs:
        quantum, classical = totals[run.key]
        runs_of = histories[run.key]
        rows.append(
            {
                "model": MODEL_LABELS[run.model],
                "encoding": f"{ENCODING_LABELS[run.encoding]} "
                f"({'native' if run.encoding == NATIVE_ENCODING[run.model] else 'swapped'})",
                "protocol": PROTOCOL_LABELS[run.train_offset],
                "parameters": f"{quantum + classical:,}",
                "test": _percent([float(history.test_metrics["accuracy"][-1]) for history in runs_of]),
                "train": _percent([float(history.train_metrics["accuracy"][-1]) for history in runs_of]),
            }
        )
    return rows


def render_markdown(rows: Sequence[dict[str, str]]) -> str:
    lines = [
        "| Model | Encoding | Protocol | Params | Final test acc. (%) | Final train acc. (%) |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {row['model']} | {row['encoding']} | {row['protocol']} | {row['parameters']} | {row['test']} | {row['train']} |"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def render_latex(rows: Sequence[dict[str, str]]) -> str:
    def cell(text: str) -> str:
        return text.replace("±", r"$\pm$").replace(",", "{,}").replace("_", r"\_")

    lines = [
        r"\begin{tabular}{lllrrr}",
        r"\hline",
        r"Model & Encoding & Protocol & Params & Final test acc. (\%) & Final train acc. (\%) \\",
        r"\hline",
    ]
    lines += [
        " & ".join(cell(row[key]) for key in ("model", "encoding", "protocol", "parameters", "test", "train")) + r" \\"
        for row in rows
    ]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the encoding-swap experiment.")
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
    for run in RUNS:
        directory = artifacts_root / run.directory
        try:
            histories[run.key] = load_saved_training_histories(directory)
            match = _PARAMETER_TOTALS_PATTERN.search(load_saved_parameter_stats_line(directory).strip())
            totals[run.key] = (int(match.group(1)), int(match.group(2)))
        except Exception as exc:
            _warn(f"Skipping {run.label}, {PROTOCOL_LABELS[run.train_offset]}: could not load {directory}: {exc}")
    available = [run for run in RUNS if run.key in histories]
    if not available:
        _warn("No runs found; nothing written.")
        return

    written: list[Path] = []
    for stem, metric in ((ACCURACY_STEM, "accuracy"), (LOSS_STEM, "loss")):
        figure = plot_metric(available, histories, metric=metric, epoch_group_size=args.epoch_group_size)
        for suffix in (".pdf", ".png"):
            figure.savefig(output_dir / f"{stem}{suffix}", dpi=200)
            written.append(output_dir / f"{stem}{suffix}")
        plt.close(figure)
    rows = build_rows(available, histories, totals)
    (output_dir / f"{TABLE_STEM}.md").write_text(render_markdown(rows), encoding="utf-8")
    (output_dir / f"{TABLE_STEM}.tex").write_text(render_latex(rows), encoding="utf-8")
    written += [output_dir / f"{TABLE_STEM}.md", output_dir / f"{TABLE_STEM}.tex"]
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
