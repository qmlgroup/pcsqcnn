from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    TemporalStatisticSummary,
    plot_temporal_summary,
    summarize_temporal_samples,
)
from qcnn.article_training import build_canonical_reference_run_directory_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_REFERENCE_DIRECTORY_NAME = build_canonical_reference_run_directory_name()
DEFAULT_PAYLOAD_FILENAME = "finite_shot_accuracy.pt"
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = (10, 100, 200, 300, 400, 500, 600, 700, 800)
DEFAULT_SHOT_BUDGETS: tuple[int | None, ...] = (128, 256, 512, 1024, 2048, None)
DEFAULT_LABEL_BLOCK_LEFT_X = 0.055
DEFAULT_LABEL_BLOCK_BOTTOM_Y = 0.075
DEFAULT_LABEL_BLOCK_ROW_SPACING = 0.055
DEFAULT_LABEL_UNDERLINE_OFFSET_POINTS = 1.0
DEFAULT_LABEL_UNDERLINE_MIN_RIGHT_PADDING = 0.012
DEFAULT_LABEL_UNDERLINE_STAGGER_STEP = 0.025
DEFAULT_LABEL_ARROW_TAIL_GAP_AXES = 0.015
DEFAULT_LABEL_UNDERLINE_LINEWIDTH = 1.0
DEFAULT_LABEL_POINTER_LINEWIDTH = 0.8


@dataclass(frozen=True)
class FiniteShotAccuracySeries:
    shot_budget: int | None
    summary: TemporalStatisticSummary

    @property
    def label(self) -> str:
        if self.shot_budget is None:
            return "Inf"
        return f"{self.shot_budget} shots"


@dataclass(frozen=True)
class _ShotLabelBlockRow:
    label: str
    row_y_axes: float
    text_left_x_axes: float
    text_right_x_axes: float
    underline_y_axes: float
    underline_end_x_axes: float


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_finite_shot_accuracy.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure 6."
        ) from exc
    return plt


def _render_shot_label(shot_budget: int | None) -> str:
    if shot_budget is None:
        return "exact"
    return f"{shot_budget} shots"


def _axes_y_distance_for_points(ax, points: float) -> float:
    return points * ax.figure.dpi / 72.0 / ax.bbox.height


def _axes_to_data(ax, *, x_axes: float, y_axes: float) -> tuple[float, float]:
    display_x, display_y = ax.transAxes.transform((x_axes, y_axes))
    data_x, data_y = ax.transData.inverted().transform((display_x, display_y))
    return float(data_x), float(data_y)


def _data_to_axes(ax, *, x_data: float, y_data: float) -> tuple[float, float]:
    display_x, display_y = ax.transData.transform((x_data, y_data))
    axes_x, axes_y = ax.transAxes.inverted().transform((display_x, display_y))
    return float(axes_x), float(axes_y)


def _interpolate_series_value_at_x(
    summary: TemporalStatisticSummary,
    x_value: float,
) -> float:
    if not summary.epoch or not summary.mean:
        raise ValueError("summary must contain at least one epoch and one mean value.")
    if len(summary.epoch) != len(summary.mean):
        raise ValueError("summary epoch and mean lengths must match.")

    epoch = [float(value) for value in summary.epoch]
    mean = [float(value) for value in summary.mean]
    if x_value <= epoch[0]:
        return mean[0]
    if x_value >= epoch[-1]:
        return mean[-1]

    for right_index in range(1, len(epoch)):
        if x_value > epoch[right_index]:
            continue
        left_index = right_index - 1
        left_epoch = epoch[left_index]
        right_epoch = epoch[right_index]
        left_value = mean[left_index]
        right_value = mean[right_index]
        if right_epoch == left_epoch:
            return right_value
        weight = (x_value - left_epoch) / (right_epoch - left_epoch)
        return left_value + weight * (right_value - left_value)

    return mean[-1]


def _compute_staggered_label_block_layout(
    ax,
    *,
    labels: Sequence[str],
    left_x: float = DEFAULT_LABEL_BLOCK_LEFT_X,
    bottom_y: float = DEFAULT_LABEL_BLOCK_BOTTOM_Y,
    row_spacing: float = DEFAULT_LABEL_BLOCK_ROW_SPACING,
    underline_offset_points: float = DEFAULT_LABEL_UNDERLINE_OFFSET_POINTS,
    underline_min_right_padding: float = DEFAULT_LABEL_UNDERLINE_MIN_RIGHT_PADDING,
    underline_stagger_step: float = DEFAULT_LABEL_UNDERLINE_STAGGER_STEP,
) -> list[_ShotLabelBlockRow]:
    texts = [
        ax.text(
            left_x,
            bottom_y + row_index * row_spacing,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="black",
        )
        for row_index, label in enumerate(labels)
    ]

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    underline_offset_axes = _axes_y_distance_for_points(ax, underline_offset_points)
    bbox_axes_by_text = [
        text.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()) for text in texts
    ]
    max_text_right_x_axes = max(float(bbox_axes.x1) for bbox_axes in bbox_axes_by_text)

    rows: list[_ShotLabelBlockRow] = []
    for row_index, (text, bbox_axes) in enumerate(zip(texts, bbox_axes_by_text, strict=True)):
        text_left_x_axes = float(bbox_axes.x0)
        text_right_x_axes = float(bbox_axes.x1)
        underline_y_axes = float(bbox_axes.y0) - underline_offset_axes
        rows.append(
            _ShotLabelBlockRow(
                label=text.get_text(),
                row_y_axes=float(text.get_position()[1]),
                text_left_x_axes=text_left_x_axes,
                text_right_x_axes=text_right_x_axes,
                underline_y_axes=underline_y_axes,
                underline_end_x_axes=(
                    max_text_right_x_axes
                    + underline_min_right_padding
                    + (len(texts) - 1 - row_index) * underline_stagger_step
                ),
            )
        )

    return rows


def _draw_staggered_label_block(
    ax,
    *,
    series_by_budget: Sequence[FiniteShotAccuracySeries],
) -> None:
    from matplotlib.patches import FancyArrowPatch

    layout_rows = _compute_staggered_label_block_layout(
        ax,
        labels=[_render_shot_label(series.shot_budget) for series in series_by_budget],
    )

    for color_index, (series, row) in enumerate(zip(series_by_budget, layout_rows, strict=True)):
        color = f"C{color_index}"
        ax.plot(
            [row.text_left_x_axes, row.underline_end_x_axes],
            [row.underline_y_axes, row.underline_y_axes],
            transform=ax.transAxes,
            color=color,
            linewidth=DEFAULT_LABEL_UNDERLINE_LINEWIDTH,
            solid_capstyle="round",
        )

        leader_x_data, _ = _axes_to_data(
            ax,
            x_axes=row.underline_end_x_axes,
            y_axes=row.underline_y_axes,
        )
        leader_target_y_data = _interpolate_series_value_at_x(series.summary, leader_x_data)
        _, leader_target_y_axes = _data_to_axes(
            ax,
            x_data=leader_x_data,
            y_data=leader_target_y_data,
        )
        arrow_tail_y_axes = max(
            row.underline_y_axes,
            leader_target_y_axes - DEFAULT_LABEL_ARROW_TAIL_GAP_AXES,
        )

        if arrow_tail_y_axes > row.underline_y_axes:
            ax.plot(
                [row.underline_end_x_axes, row.underline_end_x_axes],
                [row.underline_y_axes, arrow_tail_y_axes],
                transform=ax.transAxes,
                color=color,
                linewidth=DEFAULT_LABEL_POINTER_LINEWIDTH,
                solid_capstyle="round",
            )

        ax.add_patch(
            FancyArrowPatch(
                (row.underline_end_x_axes, arrow_tail_y_axes),
                (row.underline_end_x_axes, leader_target_y_axes),
                transform=ax.transAxes,
                arrowstyle="->",
                mutation_scale=10.0,
                linewidth=DEFAULT_LABEL_POINTER_LINEWIDTH,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )


def load_finite_shot_accuracy_payload(payload_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("finite-shot accuracy payload must deserialize to a mapping.")
    return dict(payload)


def summarize_finite_shot_accuracy_payload(
    payload_path: str | Path,
    *,
    expected_epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
    shot_budgets: Sequence[int | None] = DEFAULT_SHOT_BUDGETS,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
) -> list[FiniteShotAccuracySeries]:
    payload = load_finite_shot_accuracy_payload(payload_path)
    payload_epochs = payload.get("epochs")
    evaluations = payload.get("evaluations")
    if list(payload_epochs or []) != list(expected_epochs):
        raise ValueError(
            f"Expected epoch grid {list(expected_epochs)}, got {payload_epochs!r}."
        )
    if not isinstance(evaluations, list):
        raise ValueError("finite-shot accuracy payload must contain an 'evaluations' list.")

    evaluation_map: dict[tuple[int, int | None], float] = {}
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each evaluation entry must be a mapping.")
        epoch = evaluation.get("epoch")
        shot_budget = evaluation.get("shot_budget")
        accuracy = evaluation.get("accuracy")
        if not isinstance(epoch, int):
            raise ValueError("Each evaluation entry must contain an integer 'epoch'.")
        if shot_budget is not None and not isinstance(shot_budget, int):
            raise ValueError("Each evaluation entry must contain an integer or None 'shot_budget'.")
        if not isinstance(accuracy, (float, int)):
            raise ValueError("Each evaluation entry must contain a numeric 'accuracy'.")
        key = (epoch, shot_budget)
        if key in evaluation_map:
            raise ValueError(f"Duplicate finite-shot evaluation for epoch={epoch}, shot_budget={shot_budget!r}.")
        evaluation_map[key] = float(accuracy)

    series_by_budget: list[FiniteShotAccuracySeries] = []
    for shot_budget in shot_budgets:
        samples_by_epoch: list[list[float]] = []
        for epoch in expected_epochs:
            key = (epoch, shot_budget)
            if key not in evaluation_map:
                raise ValueError(
                    f"Missing finite-shot evaluation for epoch={epoch}, shot_budget={shot_budget!r}."
                )
            samples_by_epoch.append([evaluation_map[key]])
        series_by_budget.append(
            FiniteShotAccuracySeries(
                shot_budget=shot_budget,
                summary=summarize_temporal_samples(
                    expected_epochs,
                    samples_by_epoch,
                    epoch_start=epoch_start,
                    epoch_end=epoch_end,
                    epoch_group_size=epoch_group_size,
                    compute_band=False,
                ),
            )
        )
    return series_by_budget


def plot_article_figure_6(
    *,
    payload_path: str | Path,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()
    series_by_budget = summarize_finite_shot_accuracy_payload(
        payload_path,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
    )

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_epoch_values: list[float] = []
    for color_index, series in enumerate(series_by_budget):
        color = f"C{color_index}"
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=color,
            linewidth=1.5,
            linestyle="-",
            marker="o",
            markersize=3.0,
            label=series.label,
            show_band=False,
        )
        all_epoch_values.extend(series.summary.epoch)

    ax.set_xlabel("Epoch checkpoint")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(all_epoch_values), max(all_epoch_values))
    _draw_staggered_label_block(ax, series_by_budget=series_by_budget)

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 6 from saved finite-shot reference-run reevaluations.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_image_size_sweep artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where finite_shot_accuracy.pdf will be written.",
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
        default=1,
        help="Epoch-span width used to pool plotted points into one displayed point.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plt = _require_matplotlib()

    payload_path = (
        args.artifacts_root.expanduser().resolve()
        / DEFAULT_REFERENCE_DIRECTORY_NAME
        / DEFAULT_PAYLOAD_FILENAME
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_article_figure_6(
        payload_path=payload_path,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        epoch_group_size=args.epoch_group_size,
    )
    output_path = output_dir / "finite_shot_accuracy.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
