from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence
import warnings

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    TemporalStatisticSummary,
    configure_matplotlib_pdf_fonts,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qcnn import TrainingHistory

LineStyle = str | tuple[int, tuple[int, ...]]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_SWEEP_DIRECTORY_NAME = "pcsqcnn_architecture_sweep_full_readout"
DEFAULT_OUTPUT_FILENAME = "pcsqcnn_architecture_sweep_full_readout.pdf"
DEFAULT_EPOCH_GROUP_SIZE = 50
LAYER_LEGEND_LOCATION = "lower right"
FEATURE_LEGEND_LOCATION = "lower center"
LAYER_COLOR_CYCLE: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5")
FEATURE_LINESTYLE_CYCLE: tuple[LineStyle, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
)
RUN_DIRECTORY_PATTERN = re.compile(r"^fq(?P<feature_qubits>\d+)_ql(?P<quantum_layers>\d+)$")


@dataclass(frozen=True, order=True)
class ArchitectureKey:
    quantum_layers: int
    feature_qubits: int

    @property
    def label(self) -> str:
        return rf"Q={self.quantum_layers}, $n_f={self.feature_qubits}$"


@dataclass(frozen=True)
class ArchitectureSweepSeries:
    architecture: ArchitectureKey
    summary: TemporalStatisticSummary

    @property
    def label(self) -> str:
        return self.architecture.label


@dataclass(frozen=True)
class ArchitecturePlotStyle:
    color: str
    linestyle: LineStyle


@dataclass(frozen=True)
class LegendEntry:
    label: str
    color: str
    linestyle: LineStyle


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_pcsqcnn_architecture_sweep.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure 5a."
        ) from exc
    return plt


def build_sweep_directory_name() -> str:
    return DEFAULT_SWEEP_DIRECTORY_NAME


def build_output_filename() -> str:
    return DEFAULT_OUTPUT_FILENAME


def parse_architecture_run_directory_name(name: str) -> ArchitectureKey:
    match = RUN_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "Architecture-sweep run directory name must match "
            "'fq{feature_qubits}_ql{quantum_layers}', "
            f"got {name!r}."
        )

    return ArchitectureKey(
        quantum_layers=int(match.group("quantum_layers")),
        feature_qubits=int(match.group("feature_qubits")),
    )


def detect_observed_layers(series_by_architecture: Sequence[ArchitectureSweepSeries]) -> list[int]:
    return sorted({series.architecture.quantum_layers for series in series_by_architecture})


def detect_observed_feature_qubits(series_by_architecture: Sequence[ArchitectureSweepSeries]) -> list[int]:
    return sorted({series.architecture.feature_qubits for series in series_by_architecture})


def _resolve_reserved_style_value(
    value: int,
    *,
    palette: Sequence[object],
    axis_name: str,
) -> object:
    if value < 1:
        raise ValueError(f"{axis_name} values must be positive integers, got {value}.")
    if value > len(palette):
        raise ValueError(
            f"Figure 5a {axis_name} palette only defines {len(palette)} styles, got {value}."
        )
    return palette[value - 1]


def resolve_layer_color(quantum_layers: int) -> str:
    return str(
        _resolve_reserved_style_value(
            quantum_layers,
            palette=LAYER_COLOR_CYCLE,
            axis_name="layer",
        )
    )


def resolve_feature_linestyle(feature_qubits: int) -> LineStyle:
    return _resolve_reserved_style_value(
        feature_qubits,
        palette=FEATURE_LINESTYLE_CYCLE,
        axis_name="feature",
    )


def resolve_architecture_style(architecture: ArchitectureKey) -> ArchitecturePlotStyle:
    return ArchitecturePlotStyle(
        color=resolve_layer_color(architecture.quantum_layers),
        linestyle=resolve_feature_linestyle(architecture.feature_qubits),
    )


def build_layer_legend_entries(observed_layers: Sequence[int]) -> list[LegendEntry]:
    return [
        LegendEntry(label=f"Q={layer}", color=resolve_layer_color(layer), linestyle="-")
        for layer in sorted(set(observed_layers))
    ]


def build_feature_legend_entries(observed_feature_qubits: Sequence[int]) -> list[LegendEntry]:
    return [
        LegendEntry(
            label=rf"$n_f={feature_qubits}$",
            color="black",
            linestyle=resolve_feature_linestyle(feature_qubits),
        )
        for feature_qubits in sorted(set(observed_feature_qubits))
    ]


def _test_accuracy_series(history: TrainingHistory) -> tuple[list[int], list[float]]:
    accuracy = history.test_metrics.get("accuracy")
    if accuracy is None:
        raise ValueError("history is missing test_metrics['accuracy'].")
    if len(accuracy) != len(history.test_epoch):
        raise ValueError(
            "test_metrics['accuracy'] length does not match epoch length: "
            f"{len(accuracy)} != {len(history.test_epoch)}."
        )
    return list(history.test_epoch), list(accuracy)


def _test_accuracy_samples_by_epoch(
    histories: Sequence[TrainingHistory],
) -> tuple[list[int], list[list[float]]]:
    if not histories:
        raise ValueError("histories must contain at least one TrainingHistory.")

    reference_epoch, reference_accuracy = _test_accuracy_series(histories[0])
    if not reference_epoch:
        raise ValueError("Training histories must contain at least one epoch.")
    samples_by_epoch: list[list[float]] = [[float(value)] for value in reference_accuracy]

    for history in histories[1:]:
        epoch, accuracy = _test_accuracy_series(history)
        if epoch != reference_epoch:
            raise ValueError("All histories must share the same epoch grid.")
        for epoch_index, value in enumerate(accuracy):
            samples_by_epoch[epoch_index].append(float(value))

    return reference_epoch, samples_by_epoch


def summarize_architecture_sweep(
    *,
    sweep_root: str | Path,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> list[ArchitectureSweepSeries]:
    resolved_root = Path(sweep_root)
    if not resolved_root.is_dir():
        raise ValueError(f"Architecture sweep root directory does not exist: {resolved_root}")

    series_by_architecture: list[ArchitectureSweepSeries] = []
    for run_directory in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
        try:
            architecture = parse_architecture_run_directory_name(run_directory.name)
            histories = load_saved_training_histories(run_directory)
            epoch, samples_by_epoch = _test_accuracy_samples_by_epoch(histories)
            summary = summarize_temporal_samples(
                epoch,
                samples_by_epoch,
                epoch_start=epoch_start,
                epoch_end=epoch_end,
                epoch_group_size=epoch_group_size,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
        except Exception as exc:
            warnings.warn(
                f"Skipping architecture sweep run {run_directory.name!r}: {exc}",
                stacklevel=2,
            )
            continue

        series_by_architecture.append(
            ArchitectureSweepSeries(
                architecture=architecture,
                summary=summary,
            )
        )

    if not series_by_architecture:
        raise ValueError(f"No valid architecture sweep data is available in {resolved_root}.")

    return sorted(series_by_architecture, key=lambda series: series.architecture)


def plot_article_figure_5a(
    *,
    sweep_root: str | Path,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = DEFAULT_EPOCH_GROUP_SIZE,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> Figure:
    plt = _require_matplotlib()
    from matplotlib.lines import Line2D

    series_by_architecture = summarize_architecture_sweep(
        sweep_root=sweep_root,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    observed_layers = detect_observed_layers(series_by_architecture)
    observed_feature_qubits = detect_observed_feature_qubits(series_by_architecture)

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_epoch_values: list[float] = []
    for series in series_by_architecture:
        style = resolve_architecture_style(series.architecture)
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=style.color,
            linewidth=1.5,
            linestyle=style.linestyle,
            band_alpha=0.20,
            label="_nolegend_",
        )
        all_epoch_values.extend(series.summary.epoch)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(all_epoch_values), max(all_epoch_values))

    layer_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=entry.color, linewidth=1.5, linestyle=entry.linestyle, label=entry.label)
            for entry in build_layer_legend_entries(observed_layers)
        ],
        loc=LAYER_LEGEND_LOCATION,
        framealpha=0.8,
    )
    ax.add_artist(layer_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color=entry.color, linewidth=1.5, linestyle=entry.linestyle, label=entry.label)
            for entry in build_feature_legend_entries(observed_feature_qubits)
        ],
        loc=FEATURE_LEGEND_LOCATION,
        framealpha=0.8,
    )

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 5a from saved PCS-QCNN architecture-sweep artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the Figure 5a architecture-sweep root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pcsqcnn_architecture_sweep_full_readout.pdf will be written.",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=None,
        help="Optional inclusive starting epoch for the plotted history window.",
    )
    parser.add_argument(
        "--epoch-end",
        type=int,
        default=None,
        help="Optional inclusive ending epoch for the plotted history window.",
    )
    parser.add_argument(
        "--epoch-group-size",
        type=int,
        default=DEFAULT_EPOCH_GROUP_SIZE,
        help="Epoch-span width used to pool plotted points into one displayed point.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=25.0,
        help="Lower percentile used for shaded temporal bands.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=75.0,
        help="Upper percentile used for shaded temporal bands.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plt = _require_matplotlib()

    sweep_root = args.artifacts_root.expanduser().resolve() / build_sweep_directory_name()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_article_figure_5a(
        sweep_root=sweep_root,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        epoch_group_size=args.epoch_group_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    output_path = output_dir / build_output_filename()
    figure.savefig(output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
