import importlib.util
from pathlib import Path
import sys
import matplotlib
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcnn import TemporalStatisticSummary

PLOT_FIGURE_6_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_finite_shot_accuracy.py"
PLOT_FIGURE_6_SPEC = importlib.util.spec_from_file_location("plot_finite_shot_accuracy_module", PLOT_FIGURE_6_PATH)
if PLOT_FIGURE_6_SPEC is None or PLOT_FIGURE_6_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_finite_shot_accuracy.py from {PLOT_FIGURE_6_PATH}.")
plot_figure_6 = importlib.util.module_from_spec(PLOT_FIGURE_6_SPEC)
sys.modules[PLOT_FIGURE_6_SPEC.name] = plot_figure_6
PLOT_FIGURE_6_SPEC.loader.exec_module(plot_figure_6)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert plot_figure_6.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"


def write_fake_payload(payload_path: Path) -> None:
    payload = {
        "seed": 0,
        "epochs": list(plot_figure_6.DEFAULT_REFERENCE_EPOCHS),
        "shot_budgets": list(plot_figure_6.DEFAULT_SHOT_BUDGETS),
        "targets": torch.tensor([0, 1, 2], dtype=torch.long),
        "evaluations": [],
    }
    for budget_index, shot_budget in enumerate(plot_figure_6.DEFAULT_SHOT_BUDGETS):
        for epoch_index, epoch in enumerate(plot_figure_6.DEFAULT_REFERENCE_EPOCHS):
            payload["evaluations"].append(
                {
                    "epoch": epoch,
                    "shot_budget": shot_budget,
                    "loss": 1.0 - 0.01 * epoch_index,
                    "accuracy": 0.4 + 0.05 * budget_index + 0.01 * epoch_index,
                    "predictions": torch.tensor([0, 1, 2], dtype=torch.long),
                }
            )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)
def test_summarize_finite_shot_accuracy_payload_builds_expected_series(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_6.DEFAULT_REFERENCE_DIRECTORY_NAME / "finite_shot_accuracy.pt"
    )
    write_fake_payload(payload_path)

    series_by_budget = plot_figure_6.summarize_finite_shot_accuracy_payload(payload_path)

    assert len(series_by_budget) == 6
    assert [series.label for series in series_by_budget] == [
        "128 shots",
        "256 shots",
        "512 shots",
        "1024 shots",
        "2048 shots",
        "Inf",
    ]
    assert series_by_budget[0].summary.epoch == list(plot_figure_6.DEFAULT_REFERENCE_EPOCHS)
    assert series_by_budget[0].summary.lower is None
    assert series_by_budget[0].summary.upper is None
    assert series_by_budget[-1].summary.mean[-1] == pytest.approx(0.73)


def test_summarize_finite_shot_accuracy_payload_supports_epoch_grouping(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_6.DEFAULT_REFERENCE_DIRECTORY_NAME / "finite_shot_accuracy.pt"
    )
    write_fake_payload(payload_path)

    series_by_budget = plot_figure_6.summarize_finite_shot_accuracy_payload(
        payload_path,
        epoch_end=100,
        epoch_group_size=100,
    )

    assert series_by_budget[0].summary.epoch == pytest.approx([55.0])
    assert series_by_budget[0].summary.mean == pytest.approx([0.405])


def test_plot_article_figure_6_uses_inline_labels_instead_of_legend(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_6.DEFAULT_REFERENCE_DIRECTORY_NAME / "finite_shot_accuracy.pt"
    )
    write_fake_payload(payload_path)

    figure = plot_figure_6.plot_article_figure_6(payload_path=payload_path)

    legend = figure.axes[0].get_legend()
    assert legend is None
    text_labels = [text.get_text() for text in figure.axes[0].texts]
    assert text_labels == [
        "128 shots",
        "256 shots",
        "512 shots",
        "1024 shots",
        "2048 shots",
        "exact",
    ]

    plt.close(figure)


def test_compute_staggered_label_block_layout_uses_even_rows_and_fixed_stagger() -> None:
    figure, ax = plt.subplots()

    rows = plot_figure_6._compute_staggered_label_block_layout(
        ax,
        labels=["128 shots", "256 shots", "512 shots"],
    )

    row_positions = [row.row_y_axes for row in rows]
    assert row_positions == pytest.approx(
        [
            plot_figure_6.DEFAULT_LABEL_BLOCK_BOTTOM_Y,
            plot_figure_6.DEFAULT_LABEL_BLOCK_BOTTOM_Y + plot_figure_6.DEFAULT_LABEL_BLOCK_ROW_SPACING,
            plot_figure_6.DEFAULT_LABEL_BLOCK_BOTTOM_Y + 2 * plot_figure_6.DEFAULT_LABEL_BLOCK_ROW_SPACING,
        ]
    )
    underline_end_positions = [row.underline_end_x_axes for row in rows]
    max_text_right_x = max(row.text_right_x_axes for row in rows)
    assert underline_end_positions[0] - max_text_right_x == pytest.approx(
        plot_figure_6.DEFAULT_LABEL_UNDERLINE_MIN_RIGHT_PADDING
        + 2 * plot_figure_6.DEFAULT_LABEL_UNDERLINE_STAGGER_STEP
    )
    assert underline_end_positions[1] - underline_end_positions[0] == pytest.approx(
        -plot_figure_6.DEFAULT_LABEL_UNDERLINE_STAGGER_STEP
    )
    assert underline_end_positions[2] - underline_end_positions[1] == pytest.approx(
        -plot_figure_6.DEFAULT_LABEL_UNDERLINE_STAGGER_STEP
    )
    assert underline_end_positions[2] - max_text_right_x == pytest.approx(
        plot_figure_6.DEFAULT_LABEL_UNDERLINE_MIN_RIGHT_PADDING
    )
    assert underline_end_positions[0] > underline_end_positions[1] > underline_end_positions[2]
    assert all(underline_end_x > max_text_right_x for underline_end_x in underline_end_positions)

    plt.close(figure)


def test_interpolate_series_value_at_x_matches_linear_interpolation() -> None:
    summary = TemporalStatisticSummary(
        epoch=[100.0, 200.0, 300.0],
        mean=[0.2, 0.5, 0.8],
        lower=None,
        upper=None,
    )

    assert plot_figure_6._interpolate_series_value_at_x(summary, 150.0) == pytest.approx(0.35)
    assert plot_figure_6._interpolate_series_value_at_x(summary, 250.0) == pytest.approx(0.65)
    assert plot_figure_6._interpolate_series_value_at_x(summary, 50.0) == pytest.approx(0.2)
    assert plot_figure_6._interpolate_series_value_at_x(summary, 350.0) == pytest.approx(0.8)


def test_parse_args_defaults_to_project_artifacts_and_output_dirs() -> None:
    args = plot_figure_6.parse_args([])

    assert args.artifacts_root == plot_figure_6.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_6.DEFAULT_OUTPUT_DIR


def test_parse_args_accepts_artifacts_and_output_dir_overrides() -> None:
    args = plot_figure_6.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")


def test_parse_args_accepts_epoch_window_and_grouping_overrides() -> None:
    args = plot_figure_6.parse_args(
        [
            "--epoch-start",
            "100",
            "--epoch-end",
            "500",
            "--epoch-group-size",
            "3",
        ]
    )

    assert args.epoch_start == 100
    assert args.epoch_end == 500
    assert args.epoch_group_size == 3
