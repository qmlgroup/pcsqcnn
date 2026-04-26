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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_SWEEP_DIRECTORY_NAME = "pcsqcnn_image_size_sweep"
DEFAULT_EPOCH_GROUP_SIZE = 50
RUN_DIRECTORY_PATTERN = re.compile(r"^(?P<scaled_image_size>\d+)on(?P<image_size>\d+)$")


@dataclass(frozen=True, order=True)
class ImageSizeSweepSeries:
    scaled_image_size: int
    image_size: int
    summary: TemporalStatisticSummary

    @property
    def label(self) -> str:
        return f"{self.scaled_image_size}({self.image_size})"


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_pcsqcnn_image_size_sweep.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure 5b."
        ) from exc
    return plt


def parse_image_size_run_directory_name(name: str) -> tuple[int, int]:
    match = RUN_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            "Image-size run directory name must match "
            "'{scaled_image_size}on{image_size}', "
            f"got {name!r}."
        )
    return int(match.group("scaled_image_size")), int(match.group("image_size"))


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


def summarize_image_size_sweep(
    *,
    sweep_root: str | Path,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> list[ImageSizeSweepSeries]:
    resolved_root = Path(sweep_root)
    if not resolved_root.is_dir():
        raise ValueError(f"Image-size sweep root directory does not exist: {resolved_root}")

    series_by_image_size: list[ImageSizeSweepSeries] = []
    for run_directory in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
        try:
            scaled_image_size, image_size = parse_image_size_run_directory_name(run_directory.name)
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
                f"Skipping image-size sweep run {run_directory.name!r}: {exc}",
                stacklevel=2,
            )
            continue

        series_by_image_size.append(
            ImageSizeSweepSeries(
                scaled_image_size=scaled_image_size,
                image_size=image_size,
                summary=summary,
            )
        )

    if not series_by_image_size:
        raise ValueError(f"No valid image-size sweep data is available in {resolved_root}.")

    return sorted(series_by_image_size)


def plot_article_figure_5b(
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
    series_by_image_size = summarize_image_size_sweep(
        sweep_root=sweep_root,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_epoch_values: list[float] = []
    for color_index, series in enumerate(series_by_image_size):
        color = f"C{color_index}"
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=color,
            linewidth=1.5,
            linestyle="-",
            band_alpha=0.25,
            label=series.label,
        )
        all_epoch_values.extend(series.summary.epoch)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(all_epoch_values), max(all_epoch_values))
    ax.legend(loc="lower right", framealpha=0.8)

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 5b from saved PCS-QCNN pairwise size-sweep artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_image_size_sweep run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pcsqcnn_image_size_sweep.pdf will be written.",
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

    sweep_root = args.artifacts_root.expanduser().resolve() / DEFAULT_SWEEP_DIRECTORY_NAME
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_article_figure_5b(
        sweep_root=sweep_root,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        epoch_group_size=args.epoch_group_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    output_path = output_dir / "pcsqcnn_image_size_sweep.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
