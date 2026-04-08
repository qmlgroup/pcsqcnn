from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence
import warnings

from qcnn import ARTICLE_PANEL_FIGSIZE, load_saved_training_histories

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qcnn import TrainingHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_SWEEP_DIRECTORY_NAME = "pcsqcnn_brightness_sweep"
DEFAULT_EPOCHS: tuple[int, ...] = (150, 300)
RUN_DIRECTORY_PATTERN = re.compile(
    r"^fq(?P<feature_qubits>\d+)_ql(?P<quantum_layers>\d+)_u(?P<numerator>\d+)by(?P<denominator>\d+)pi$"
)


@dataclass(frozen=True, order=True)
class ArchitectureKey:
    feature_qubits: int
    quantum_layers: int

    @property
    def label(self) -> str:
        return rf"Q={self.quantum_layers}, $n_f={self.feature_qubits}$"


@dataclass(frozen=True)
class SweepEpochSeries:
    architecture: ArchitectureKey
    brightness_pi: list[float]
    mean: list[float]
    minimum: list[float]
    maximum: list[float]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_brightness_sweep.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render sweep figures."
        ) from exc
    return plt


def parse_sweep_run_directory_name(name: str) -> tuple[ArchitectureKey, Fraction]:
    match = RUN_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "Sweep run directory name must match "
            "'fq{feature_qubits}_ql{quantum_layers}_u{numerator}by{denominator}pi', "
            f"got {name!r}."
        )

    feature_qubits = int(match.group("feature_qubits"))
    quantum_layers = int(match.group("quantum_layers"))
    brightness_pi = Fraction(int(match.group("numerator")), int(match.group("denominator")))
    return ArchitectureKey(feature_qubits=feature_qubits, quantum_layers=quantum_layers), brightness_pi


def _test_accuracy_at_epoch(history: TrainingHistory, *, epoch: int) -> float:
    try:
        epoch_index = list(history.test_epoch).index(epoch)
    except ValueError as exc:
        raise ValueError(f"history does not contain epoch {epoch}.") from exc

    accuracy = history.test_metrics.get("accuracy")
    if accuracy is None:
        raise ValueError("history is missing test_metrics['accuracy'].")
    if len(accuracy) != len(history.test_epoch):
        raise ValueError(
            "test_metrics['accuracy'] length does not match epoch length: "
            f"{len(accuracy)} != {len(history.test_epoch)}."
        )
    return float(accuracy[epoch_index])


def summarize_brightness_sweep_epoch(
    *,
    sweep_root: str | Path,
    epoch: int,
) -> list[SweepEpochSeries]:
    resolved_root = Path(sweep_root)
    if not resolved_root.is_dir():
        raise ValueError(f"Sweep root directory does not exist: {resolved_root}")

    grouped_points: dict[ArchitectureKey, list[tuple[Fraction, float, float, float]]] = defaultdict(list)
    for run_directory in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
        try:
            architecture, brightness_pi = parse_sweep_run_directory_name(run_directory.name)
            histories = load_saved_training_histories(run_directory)
            accuracy_values = [_test_accuracy_at_epoch(history, epoch=epoch) for history in histories]
        except Exception as exc:
            warnings.warn(
                f"Skipping sweep run {run_directory.name!r} for epoch {epoch}: {exc}",
                stacklevel=2,
            )
            continue

        grouped_points[architecture].append(
            (
                brightness_pi,
                sum(accuracy_values) / len(accuracy_values),
                min(accuracy_values),
                max(accuracy_values),
            )
        )

    if not grouped_points:
        raise ValueError(f"No valid sweep data is available for epoch {epoch} in {resolved_root}.")

    series_by_architecture: list[SweepEpochSeries] = []
    for architecture in sorted(grouped_points):
        ordered_points = sorted(grouped_points[architecture], key=lambda point: point[0])
        brightness_pi, mean, minimum, maximum = zip(*ordered_points, strict=True)
        series_by_architecture.append(
            SweepEpochSeries(
                architecture=architecture,
                brightness_pi=[float(value) for value in brightness_pi],
                mean=list(mean),
                minimum=list(minimum),
                maximum=list(maximum),
            )
        )

    return series_by_architecture


def plot_brightness_sweep_epoch(
    *,
    sweep_root: str | Path,
    epoch: int,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> Figure:
    plt = _require_matplotlib()
    series_by_architecture = summarize_brightness_sweep_epoch(sweep_root=sweep_root, epoch=epoch)

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_x_values: list[float] = []
    for color_index, series in enumerate(series_by_architecture):
        color = f"C{color_index}"
        ax.fill_between(
            series.brightness_pi,
            series.minimum,
            series.maximum,
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        ax.plot(
            series.brightness_pi,
            series.mean,
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=series.architecture.label,
        )
        all_x_values.extend(series.brightness_pi)

    ax.set_title(f"Epoch {epoch}")
    ax.set_xlabel("Max brightness / pi")
    ax.set_ylabel("Accuracy")
    if len(all_x_values) == 1:
        ax.set_xlim(all_x_values[0] - 0.05, all_x_values[0] + 0.05)
    else:
        ax.set_xlim(min(all_x_values), max(all_x_values))
    ax.legend(loc="best", framealpha=0.8)

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-epoch brightness sweep figures from saved PCS-QCNN sweep artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_brightness_sweep run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where brightness_sweep_epochXXX.pdf files will be written.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPOCHS),
        help="Epochs to render as separate brightness sweep figures.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plt = _require_matplotlib()

    sweep_root = args.artifacts_root.expanduser().resolve() / DEFAULT_SWEEP_DIRECTORY_NAME
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in args.epochs:
        figure = plot_brightness_sweep_epoch(sweep_root=sweep_root, epoch=epoch)
        output_path = output_dir / f"brightness_sweep_epoch{epoch:03d}.pdf"
        figure.savefig(output_path)
        plt.close(figure)
        print(output_path)


if __name__ == "__main__":
    main()
