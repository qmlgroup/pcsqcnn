"""Render the translated-MNIST PCS-QCNN versus QCS-QCNN, Henderson-QCNN, and Hur-QCNN comparison.

The script reads saved training artifacts and writes:

- ``translated_mnist_qcs_comparison.pdf``: the Figure 2(b)-style panel with
  PCS-QCNN, the FRQI QCS-QCNN with matched quantum core and head (protocol and
  tuned learning rate and initialization), and the RBC-QCNN control; all
  share the encoding and 39,434 parameters;
- ``translated_mnist_qcs_variants.pdf``: test accuracy of both QCS-QCNN
  variants against PCS-QCNN and RBC-QCNN;
- ``translated_mnist_henderson_variants.pdf``: test accuracy of the
  Henderson-QCNN (multilayer quanvolution) variants against PCS-QCNN;
- ``translated_mnist_hur_comparison.pdf``: the Figure 2(b)-style panel with
  PCS-QCNN, the Hur-QCNN baseline (paper circuit, 112-channel
  parameter-matched variant, and tuned learning rate and initialization when
  present), and the RBC-QCNN control;
- ``translated_mnist_hur_variants.pdf``: test accuracy of the Hur-QCNN
  variants (1, 8, and 112 channels) against PCS-QCNN and RBC-QCNN;
- ``translated_mnist_hur_filters.pdf``: test accuracy of the single-circuit
  Hur-QCNN with 1, 4, 16, and 64 filters per layer against PCS-QCNN and
  RBC-QCNN;
- ``qcsqcnn_tuning_curves.pdf`` and ``hur_qcnn_tuning_curves.pdf``: validation
  accuracy of every configuration of the QCS-QCNN and Hur-QCNN learning-rate
  and initialization searches, when their summaries exist;
- ``translated_mnist_accuracy_vs_parameters.pdf``: final mean test accuracy
  versus total trainable parameters for the PCS-QCNN architecture sweep and
  every other model;
- ``translated_mnist_qcs_comparison_table.md`` and ``.tex``: parameter counts
  and final test accuracies of every available model.

Missing runs are skipped with a warning so partial artifact trees still render.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import sys
from typing import TYPE_CHECKING

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    configure_matplotlib_pdf_fonts,
    load_saved_parameter_stats_line,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qcnn import TemporalStatisticSummary, TrainingHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_EPOCH_GROUP_SIZE = 50
DEFAULT_TAIL_EVALUATIONS = 10
DEFAULT_LIMIT_LABEL_Y_OFFSET_POINTS = 4
CROWDED_LIMIT_LABEL_Y_OFFSET_POINTS = -10
CROWDED_LIMIT_LABEL_GAP = 0.03
COMPARISON_FILENAME = "translated_mnist_qcs_comparison.pdf"
VARIANTS_FILENAME = "translated_mnist_qcs_variants.pdf"
HENDERSON_FILENAME = "translated_mnist_henderson_variants.pdf"
HUR_FILENAME = "translated_mnist_hur_comparison.pdf"
HUR_VARIANTS_FILENAME = "translated_mnist_hur_variants.pdf"
HUR_FILTERS_FILENAME = "translated_mnist_hur_filters.pdf"
TUNING_FILENAME = "qcsqcnn_tuning_curves.pdf"
TUNING_SUMMARY = "qcsqcnn_tuning/fq2_ql3_sl8/search_summary.json"
HUR_TUNING_FILENAME = "hur_qcnn_tuning_curves.pdf"
HUR_TUNING_SUMMARY = "hur_qcnn_tuning/su4_l1/search_summary.json"
# (search summary relative to the artifacts root, output filename)
TUNING_SEARCHES: tuple[tuple[str, str], ...] = (
    (TUNING_SUMMARY, TUNING_FILENAME),
    (HUR_TUNING_SUMMARY, HUR_TUNING_FILENAME),
)
PARAMETERS_FILENAME = "translated_mnist_accuracy_vs_parameters.pdf"
TABLE_STEM = "translated_mnist_qcs_comparison_table"
_PARAMETER_TOTALS_PATTERN = re.compile(r"\bQ(\d+) C(\d+)$")


@dataclass(frozen=True)
class RunSpec:
    """One saved run family and how it is labelled in figures and tables."""

    key: str
    relative_directory: str
    label: str
    color: str
    train_linestyle: str | tuple[int, tuple[int, ...]]
    family: str


PCS_QCNN = RunSpec("pcsqcnn", "pcsqcnn_architecture_sweep_full_readout/fq2_ql3", "PCS-QCNN", "C0", "-.", "pcs")
QCS_QCNN = RunSpec("qcsqcnn", "qcsqcnn_translated_mnist/fq2_ql3_sl8", "QCS-QCNN", "C2", ":", "qcs")
QCS_TUNED = RunSpec(
    "qcsqcnn_tuned",
    "qcsqcnn_translated_mnist_tuned/fq2_ql3_sl8",
    "QCS-QCNN (tuned)",
    "C9",
    (0, (1, 1)),
    "qcs",
)
QCS_HEAD_MATCHED = RunSpec(
    "qcsqcnn_head_matched",
    "qcsqcnn_translated_mnist/fq2_ql3_sl1",
    "QCS-QCNN K=1 (head-matched)",
    "C8",
    ":",
    "qcs",
)
HENDERSON_QCNN = RunSpec(
    "henderson",
    "henderson_qcnn_translated_mnist/ch12_ql3",
    "Henderson-QCNN",
    "C3",
    ":",
    "henderson",
)
HENDERSON_QUANVOLUTION = RunSpec(
    "henderson_quanvolution",
    "henderson_qcnn_translated_mnist/ch1_ql1",
    "Henderson-QCNN Q=1, 1 ch (quanvolution)",
    "C4",
    ":",
    "henderson",
)
HENDERSON_HEAD_MATCHED = RunSpec(
    "henderson_head_matched",
    "henderson_qcnn_translated_mnist/ch1_ql3",
    "Henderson-QCNN Q=3, 1 ch (head-matched)",
    "C5",
    ":",
    "henderson",
)
HUR_QCNN = RunSpec("hur", "hur_qcnn_translated_mnist/su4_l1", "Hur-QCNN", "C6", ":", "hur")
HUR_TUNED = RunSpec(
    "hur_tuned",
    "hur_qcnn_translated_mnist_tuned/su4_l1",
    "Hur-QCNN (tuned)",
    "C7",
    (0, (1, 1)),
    "hur",
)
HUR_HEAD_MATCHED = RunSpec(
    "hur_head_matched",
    "hur_qcnn_translated_mnist/ch8_su4_l1",
    "Hur-QCNN x8 (head-matched)",
    "mediumvioletred",
    ":",
    "hur",
)
HUR_FILTERS = tuple(
    RunSpec(
        f"hur_l{filters}",
        f"hur_qcnn_translated_mnist/su4_l{filters}",
        f"Hur-QCNN l={filters}",
        color,
        ":",
        "hur",
    )
    for filters, color in ((4, "goldenrod"), (16, "saddlebrown"), (64, "dimgray"))
)
HUR_PARAMETER_MATCHED = RunSpec(
    "hur_parameter_matched",
    "hur_qcnn_translated_mnist/ch112_su4_l1",
    "Hur-QCNN x112 (parameter-matched)",
    "darkcyan",
    (0, (3, 1, 1, 1)),
    "hur",
)
RBC_QCNN = RunSpec("rbcqcnn", "pcsqcnn_no_qft_spc1000", "RBC-QCNN", "C1", "--", "rbc")
CLASSICAL_CNN = RunSpec("cnn", "classical_cnn/16on32", "Classical CNN", "0.25", "-.", "classical")
CLASSICAL_MLP = RunSpec("mlp", "classical_mlp/16on32", "Classical MLP", "0.55", "--", "classical")

# (filename, runs, show train curves, runs that must exist for the panel to render, legend location)
PANELS: tuple[tuple[str, tuple[RunSpec, ...], bool, tuple[RunSpec, ...], str], ...] = (
    (COMPARISON_FILENAME, (PCS_QCNN, QCS_QCNN, QCS_TUNED, RBC_QCNN), True, (QCS_QCNN, QCS_TUNED), "upper center"),
    (
        VARIANTS_FILENAME,
        (PCS_QCNN, QCS_QCNN, QCS_TUNED, QCS_HEAD_MATCHED, RBC_QCNN),
        False,
        (QCS_QCNN, QCS_TUNED, QCS_HEAD_MATCHED),
        "upper center",
    ),
    (
        HENDERSON_FILENAME,
        (PCS_QCNN, HENDERSON_QCNN, HENDERSON_QUANVOLUTION, HENDERSON_HEAD_MATCHED),
        False,
        (HENDERSON_QCNN, HENDERSON_QUANVOLUTION, HENDERSON_HEAD_MATCHED),
        "lower right",
    ),
    (
        HUR_FILENAME,
        (PCS_QCNN, HUR_QCNN, HUR_PARAMETER_MATCHED, HUR_TUNED, RBC_QCNN),
        True,
        (HUR_QCNN, HUR_PARAMETER_MATCHED, HUR_TUNED),
        "upper center",
    ),
    (
        HUR_FILTERS_FILENAME,
        (PCS_QCNN, HUR_QCNN, *HUR_FILTERS, RBC_QCNN),
        False,
        HUR_FILTERS,
        "upper center",
    ),
    (
        HUR_VARIANTS_FILENAME,
        (PCS_QCNN, HUR_QCNN, HUR_HEAD_MATCHED, HUR_PARAMETER_MATCHED, RBC_QCNN),
        False,
        (HUR_HEAD_MATCHED, HUR_PARAMETER_MATCHED),
        "upper center",
    ),
)
TABLE_RUNS: tuple[RunSpec, ...] = (
    CLASSICAL_CNN,
    CLASSICAL_MLP,
    PCS_QCNN,
    QCS_QCNN,
    QCS_TUNED,
    QCS_HEAD_MATCHED,
    HENDERSON_QCNN,
    HENDERSON_QUANVOLUTION,
    HENDERSON_HEAD_MATCHED,
    HUR_QCNN,
    *HUR_FILTERS,
    HUR_HEAD_MATCHED,
    HUR_PARAMETER_MATCHED,
    HUR_TUNED,
    RBC_QCNN,
)
PCS_SWEEP_ROOT = "pcsqcnn_architecture_sweep_full_readout"
PCS_SWEEP_RUN_PATTERN = re.compile(r"^fq(\d+)_ql(\d+)$")


@dataclass(frozen=True)
class LoadedRun:
    spec: RunSpec
    directory: Path
    histories: Sequence["TrainingHistory"]
    quantum_parameters: int
    classical_parameters: int

    @property
    def total_parameters(self) -> int:
        return self.quantum_parameters + self.classical_parameters

    def final_accuracies(self, split: str = "test") -> list[float]:
        return [
            float((history.test_metrics if split == "test" else history.train_metrics)["accuracy"][-1])
            for history in self.histories
        ]

    def tail_test_accuracies(self, evaluations: int = DEFAULT_TAIL_EVALUATIONS) -> list[float]:
        """Per-seed mean test accuracy over the last ``evaluations`` test evaluations."""

        return [
            statistics.fmean(float(value) for value in history.test_metrics["accuracy"][-evaluations:])
            for history in self.histories
        ]


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_qcsqcnn_comparison.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra."
        ) from exc
    return plt


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def parse_parameter_totals(parameter_stats_line: str) -> tuple[int, int]:
    """Return ``(quantum, classical)`` totals from a saved ``parameter_stats_line``."""

    match = _PARAMETER_TOTALS_PATTERN.search(parameter_stats_line.strip())
    if match is None:
        raise ValueError(f"Could not parse parameter totals from {parameter_stats_line!r}.")
    return int(match.group(1)), int(match.group(2))


def load_run(artifacts_root: Path, spec: RunSpec) -> LoadedRun | None:
    """Load one run family, or warn and return ``None`` when it is unavailable."""

    directory = artifacts_root / spec.relative_directory
    try:
        histories = load_saved_training_histories(directory)
        quantum, classical = parse_parameter_totals(load_saved_parameter_stats_line(directory))
    except Exception as exc:
        _warn(f"Skipping {spec.label}: could not load {directory}: {exc}")
        return None
    return LoadedRun(
        spec=spec,
        directory=directory,
        histories=histories,
        quantum_parameters=quantum,
        classical_parameters=classical,
    )


def load_pcs_architecture_sweep(artifacts_root: Path) -> list[tuple[int, int, LoadedRun]]:
    """Load every ``fq{n_f}_ql{Q}`` PCS-QCNN sweep run as ``(n_f, Q, run)`` triples."""

    sweep_root = artifacts_root / PCS_SWEEP_ROOT
    if not sweep_root.is_dir():
        _warn(f"Skipping the PCS-QCNN architecture sweep: {sweep_root} does not exist.")
        return []
    result: list[tuple[int, int, LoadedRun]] = []
    for directory in sorted(sweep_root.iterdir()):
        match = PCS_SWEEP_RUN_PATTERN.match(directory.name)
        if match is None:
            continue
        spec = RunSpec(
            key=f"pcsqcnn_{directory.name}",
            relative_directory=f"{PCS_SWEEP_ROOT}/{directory.name}",
            label=f"PCS-QCNN Q={match.group(2)}, n_f={match.group(1)}",
            color=PCS_QCNN.color,
            train_linestyle=PCS_QCNN.train_linestyle,
            family="pcs",
        )
        loaded = load_run(artifacts_root, spec)
        if loaded is not None:
            result.append((int(match.group(1)), int(match.group(2)), loaded))
    return result


def _accuracy_samples(
    histories: Sequence["TrainingHistory"],
    *,
    split: str,
) -> tuple[list[int], list[list[float]]]:
    reference = histories[0]
    epoch = list(reference.train_epoch if split == "train" else reference.test_epoch)
    samples: list[list[float]] = [[] for _ in epoch]
    for history in histories:
        history_epoch = list(history.train_epoch if split == "train" else history.test_epoch)
        if history_epoch != epoch:
            raise ValueError("All histories must share the same epoch grid.")
        metrics = history.train_metrics if split == "train" else history.test_metrics
        for index, value in enumerate(metrics["accuracy"]):
            samples[index].append(float(value))
    return epoch, samples


def summarize_accuracy(
    histories: Sequence["TrainingHistory"],
    *,
    split: str,
    epoch_group_size: int,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    compute_band: bool = True,
) -> "TemporalStatisticSummary":
    """Group per-seed accuracy traces into the article's mean and percentile band."""

    epoch, samples = _accuracy_samples(histories, split=split)
    return summarize_temporal_samples(
        epoch,
        samples,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        compute_band=compute_band,
    )


def _final_mean_test_accuracy(run: LoadedRun) -> float:
    return statistics.fmean(run.final_accuracies("test"))


def _limit_label_offsets(runs: Sequence[LoadedRun]) -> dict[str, int]:
    """Place final-value labels above their guide line unless a higher label is too close."""

    offsets: dict[str, int] = {}
    placed: list[float] = []
    for run in sorted(runs, key=_final_mean_test_accuracy, reverse=True):
        value = _final_mean_test_accuracy(run)
        crowded = any(abs(value - other) < CROWDED_LIMIT_LABEL_GAP for other in placed)
        offsets[run.spec.key] = (
            CROWDED_LIMIT_LABEL_Y_OFFSET_POINTS if crowded else DEFAULT_LIMIT_LABEL_Y_OFFSET_POINTS
        )
        placed.append(value)
    return offsets


def plot_training_panel(
    runs: Sequence[LoadedRun],
    *,
    show_train: bool = True,
    epoch_group_size: int = DEFAULT_EPOCH_GROUP_SIZE,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
    legend_location: str = "lower right",
) -> "Figure":
    """Render a Figure 2(b)-style accuracy-versus-epoch panel for the given runs."""

    plt = _require_matplotlib()
    if not runs:
        raise ValueError("plot_training_panel requires at least one run.")

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    label_offsets = _limit_label_offsets(runs)
    epochs: list[float] = []
    test_handles = []
    train_handles = []
    for run in runs:
        test_summary = summarize_accuracy(
            run.histories,
            split="test",
            epoch_group_size=epoch_group_size,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        test_handle = plot_temporal_summary(
            ax,
            summary=test_summary,
            color=run.spec.color,
            label=run.spec.label,
            band_alpha=0.30,
            limit_value=_final_mean_test_accuracy(run),
            show_limit_label=True,
            limit_label_x_axes=0.82,
            limit_label_y_offset_points=label_offsets[run.spec.key],
        )
        test_handles.append(test_handle)
        epochs.extend(test_summary.epoch)
        if show_train:
            train_summary = summarize_accuracy(
                run.histories,
                split="train",
                epoch_group_size=epoch_group_size,
                compute_band=False,
            )
            train_handles.append(
                plot_temporal_summary(
                    ax,
                    summary=train_summary,
                    color=run.spec.color,
                    label=f"Train {run.spec.label}",
                    linestyle=run.spec.train_linestyle,
                    show_band=False,
                )
            )
            epochs.extend(train_summary.epoch)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(epochs), max(epochs))
    if train_handles and len(test_handles) <= 3:
        # One column per model: the test entry above its train entry.
        handles = [handle for pair in zip(test_handles, train_handles) for handle in pair]
        columns = len(test_handles)
    else:
        # Test curves fill the first column and train curves the second.
        handles = test_handles + train_handles
        columns = 2 if (train_handles or legend_location == "upper center") else 1
    if legend_location == "upper center":
        # Reserve headroom above the curves, growing with the legend height, so the
        # legend never covers data or the final-value labels.
        rows = math.ceil(len(handles) / columns)
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + max(0.22, 0.085 * rows + 0.04) * (upper - lower))
    ax.legend(handles=handles, loc=legend_location, framealpha=0.8, ncols=columns, fontsize="small")
    figure.tight_layout()
    return figure


def plot_accuracy_versus_parameters(
    *,
    pcs_sweep: Sequence[tuple[int, int, LoadedRun]],
    other_runs: Sequence[LoadedRun],
    figsize: tuple[float, float] = (ARTICLE_PANEL_FIGSIZE[0] + 3.0, ARTICLE_PANEL_FIGSIZE[1]),
) -> "Figure":
    """Plot final mean test accuracy (with seed min-max bars) against parameter count."""

    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def draw(run: LoadedRun, *, marker: str, color: str, label: str | None, markersize: float = 6.0) -> None:
        accuracies = run.final_accuracies("test")
        mean = statistics.fmean(accuracies)
        ax.errorbar(
            [run.total_parameters],
            [mean],
            yerr=[[mean - min(accuracies)], [max(accuracies) - mean]],
            fmt=marker,
            color=color,
            markersize=markersize,
            capsize=2.5,
            linewidth=1.0,
            label=label,
        )

    if pcs_sweep:
        for index, (feature_qubits, quantum_layers, run) in enumerate(pcs_sweep):
            draw(
                run,
                marker="o",
                color=PCS_QCNN.color,
                label="PCS-QCNN sweep (labels: Q, $n_f$)" if index == 0 else None,
                markersize=4.5,
            )
            ax.annotate(
                f"{quantum_layers},{feature_qubits}",
                (run.total_parameters, _final_mean_test_accuracy(run)),
                xytext=(4, -3),
                textcoords="offset points",
                fontsize=6,
                color=PCS_QCNN.color,
            )
    markers = {"qcs": "s", "henderson": "P", "hur": "v", "rbc": "^", "classical": "D", "pcs": "o"}
    for run in other_runs:
        draw(run, marker=markers.get(run.spec.family, "x"), color=run.spec.color, label=run.spec.label)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final mean test accuracy")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.85, fontsize="small")
    figure.tight_layout()
    return figure


def plot_tuning_curves(summary: dict, *, figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE) -> "Figure":
    """Plot validation accuracy of every search configuration with the rung boundaries."""

    plt = _require_matplotlib()
    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")
    rates = sorted(set(summary["learning_rates"]))
    scales = sorted(set(summary["init_scales"]))
    colors = {rate: f"C{index}" for index, rate in enumerate(rates)}
    styles = dict(zip(scales, ("-", "--", ":", "-.")))
    selected = summary["selected"]["name"]
    window = int(summary.get("selection_window", 5))
    for name, history in summary["histories"].items():
        parsed_rate, parsed_scale = _parse_config_name(name)
        # Names round the values (2 pi -> 6.283), so snap them back onto the grid.
        rate = min(rates, key=lambda value: abs(value - parsed_rate))
        scale = min(scales, key=lambda value: abs(value - parsed_scale))
        accuracy = history["validation_accuracy"]
        # Trailing mean over the selection window: the quantity the search ranks on.
        smoothed = [statistics.fmean(accuracy[max(0, index - window + 1) : index + 1]) for index in range(len(accuracy))]
        ax.plot(history["epoch"], accuracy, color=colors[rate], linewidth=0.6, alpha=0.15)
        ax.plot(
            history["epoch"],
            smoothed,
            color=colors[rate],
            linestyle=styles[scale],
            linewidth=2.2 if name == selected else 1.1,
            alpha=1.0 if name == selected else 0.85,
        )
    for epochs in summary["rung_epochs"][:-1]:
        ax.axvline(epochs, color="0.6", linewidth=0.8, linestyle="--")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=colors[rate], label=f"lr {rate:g}") for rate in rates]
    handles += [Line2D([], [], color="0.3", linestyle=styles[scale], label=f"init U(0, {scale:.3g})") for scale in scales]
    handles.append(Line2D([], [], color="0.3", linewidth=2.2, label=f"selected: {selected}"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small", framealpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation accuracy ({window}-evaluation mean)")
    figure.set_size_inches(figsize[0] + 2.5, figsize[1])
    figure.tight_layout()
    return figure


def _parse_config_name(name: str) -> tuple[float, float]:
    match = re.fullmatch(r"lr([0-9.e+-]+)_init([0-9.e+-]+)", name)
    if match is None:
        raise ValueError(f"Unexpected search configuration name {name!r}.")
    return float(match.group(1)), float(match.group(2))


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def build_summary_rows(runs: Sequence[LoadedRun]) -> list[dict[str, str]]:
    """Return one plain row per run with parameter counts and final accuracies."""

    rows: list[dict[str, str]] = []
    for run in runs:
        test = run.final_accuracies("test")
        train = run.final_accuracies("train")
        tail = run.tail_test_accuracies()
        spread = statistics.stdev(test) if len(test) > 1 else 0.0
        tail_spread = statistics.stdev(tail) if len(tail) > 1 else 0.0
        rows.append(
            {
                "model": run.spec.label,
                "quantum": f"{run.quantum_parameters:,}",
                "classical": f"{run.classical_parameters:,}",
                "total": f"{run.total_parameters:,}",
                "test_mean": _format_percent(statistics.fmean(test)),
                "test_std": _format_percent(spread),
                "test_seeds": ", ".join(_format_percent(value) for value in test),
                "tail_mean": _format_percent(statistics.fmean(tail)),
                "tail_std": _format_percent(tail_spread),
                "train_mean": _format_percent(statistics.fmean(train)),
                "directory": run.spec.relative_directory,
            }
        )
    return rows


def render_markdown_table(rows: Sequence[dict[str, str]]) -> str:
    lines = [
        "| Model | Quantum params | Classical params | Total params | Final test acc. (%) | Seeds (%) "
        "| Last-10-eval. test acc. (%) | Final train acc. (%) | Run directory |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['quantum']} | {row['classical']} | {row['total']} | "
            f"{row['test_mean']} ± {row['test_std']} | {row['test_seeds']} | "
            f"{row['tail_mean']} ± {row['tail_std']} | {row['train_mean']} | `{row['directory']}` |"
        )
    lines.append("")
    lines.append(
        "Final accuracies are taken at the last training epoch, as in the article. The last-10-evaluation "
        f"column averages each seed's test accuracy over its last {DEFAULT_TAIL_EVALUATIONS} test evaluations "
        "(every 10 epochs) before averaging over seeds. Means and sample standard deviations run over the "
        "saved seeds."
    )
    return "\n".join(lines) + "\n"


def render_latex_table(rows: Sequence[dict[str, str]]) -> str:
    def escape(text: str) -> str:
        return text.replace("_", r"\_").replace("&", r"\&")

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Model & Quantum params & Classical params & Total params & Accuracy (\%) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{escape(row['model'])} & {row['quantum'].replace(',', '{,}')} & "
            f"{row['classical'].replace(',', '{,}')} & {row['total'].replace(',', '{,}')} & "
            f"${row['test_mean']} \\pm {row['test_std']}$ \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the translated-MNIST PCS-QCNN versus QCS-QCNN comparison.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the saved PCS-QCNN, QCS-QCNN, Hur-QCNN, RBC-QCNN, and classical runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PDFs and summary tables are written.",
    )
    parser.add_argument(
        "--epoch-group-size",
        type=int,
        default=DEFAULT_EPOCH_GROUP_SIZE,
        help="Epoch-width grouping used for the temporal panels, as in Figure 2(b).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = {spec.key: load_run(artifacts_root, spec) for spec in TABLE_RUNS}
    written: list[Path] = []

    for filename, specs, show_train, required, legend_location in PANELS:
        if not any(loaded[spec.key] is not None for spec in required):
            _warn(f"Skipping {filename}: none of {[spec.label for spec in required]} is available.")
            continue
        runs = [loaded[spec.key] for spec in specs if loaded[spec.key] is not None]
        figure = plot_training_panel(
            runs,
            show_train=show_train,
            epoch_group_size=args.epoch_group_size,
            legend_location=legend_location,
        )
        figure.savefig(output_dir / filename)
        plt.close(figure)
        written.append(output_dir / filename)

    for relative_summary, filename in TUNING_SEARCHES:
        summary_path = artifacts_root / relative_summary
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        figure = plot_tuning_curves(summary)
        figure.savefig(output_dir / filename)
        plt.close(figure)
        written.append(output_dir / filename)

    other_runs = [run for run in loaded.values() if run is not None and run.spec.key != PCS_QCNN.key]
    pcs_sweep = load_pcs_architecture_sweep(artifacts_root)
    if pcs_sweep or other_runs:
        figure = plot_accuracy_versus_parameters(pcs_sweep=pcs_sweep, other_runs=other_runs)
        figure.savefig(output_dir / PARAMETERS_FILENAME)
        plt.close(figure)
        written.append(output_dir / PARAMETERS_FILENAME)

    rows = build_summary_rows([run for run in loaded.values() if run is not None])
    if rows:
        markdown = render_markdown_table(rows)
        (output_dir / f"{TABLE_STEM}.md").write_text(markdown, encoding="utf-8")
        (output_dir / f"{TABLE_STEM}.tex").write_text(render_latex_table(rows), encoding="utf-8")
        written.extend([output_dir / f"{TABLE_STEM}.md", output_dir / f"{TABLE_STEM}.tex"])
        print(markdown)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
