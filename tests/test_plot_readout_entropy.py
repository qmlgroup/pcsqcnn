import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

import pytest
import torch

from .script_loading import load_run_script

plot_figure_s2b = load_run_script(
    "plot_readout_entropy_module",
    "plot_readout_entropy.py",
)


def write_fake_payload(
    payload_path: Path,
    *,
    epochs: tuple[int, ...] = (100, 200),
    shot_budgets: tuple[int, ...] = (128, 256),
) -> None:
    if epochs == (100, 200) and shot_budgets == (128, 256):
        evaluations = [
            {
                "epoch": 100,
                "shot_budget": 128,
                "entropy": torch.tensor([0.1, 0.2, 0.4], dtype=torch.float32),
            },
            {
                "epoch": 200,
                "shot_budget": 128,
                "entropy": torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32),
            },
            {
                "epoch": 100,
                "shot_budget": 256,
                "entropy": torch.tensor([0.6, 0.8, 1.0], dtype=torch.float32),
            },
            {
                "epoch": 200,
                "shot_budget": 256,
                "entropy": torch.tensor([0.2, 0.6, 1.0], dtype=torch.float32),
            },
        ]
    else:
        evaluations = []
        for shot_budget_index, shot_budget in enumerate(shot_budgets):
            for epoch_index, epoch in enumerate(epochs):
                start = 0.1 * (shot_budget_index + 1) + 0.05 * epoch_index
                evaluations.append(
                    {
                        "epoch": epoch,
                        "shot_budget": shot_budget,
                        "entropy": torch.tensor(
                            [start, start + 0.1, start + 0.3],
                            dtype=torch.float32,
                        ),
                    }
                )
    payload = {
        "seed": 7,
        "epochs": list(epochs),
        "shot_budgets": list(shot_budgets),
        "evaluations": evaluations,
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert plot_figure_s2b.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"


def test_summarize_readout_entropy_payload_builds_mean_and_interquartile_series(
    tmp_path: Path,
) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s2b.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_entropy.pt"
    )
    write_fake_payload(payload_path)

    series_list = plot_figure_s2b.summarize_readout_entropy_payload(
        payload_path,
        expected_epochs=(100, 200),
        shot_budgets=(128, 256),
    )

    assert [series.label for series in series_list] == ["128 shots", "256 shots"]
    assert series_list[0].summary.mean == pytest.approx([0.23333333, 0.5])
    assert series_list[0].summary.lower == pytest.approx([0.15, 0.4])
    assert series_list[0].summary.upper == pytest.approx([0.3, 0.6])
    assert series_list[1].summary.mean == pytest.approx([0.8, 0.6])
    assert series_list[1].summary.lower == pytest.approx([0.7, 0.4])
    assert series_list[1].summary.upper == pytest.approx([0.9, 0.8])


def test_plot_article_figure_s2b_uses_shared_temporal_plot_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s2b.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_entropy.pt"
    )
    write_fake_payload(
        payload_path,
        epochs=plot_figure_s2b.DEFAULT_REFERENCE_EPOCHS,
        shot_budgets=plot_figure_s2b.DEFAULT_SHOT_BUDGETS,
    )
    expected_series = plot_figure_s2b.summarize_readout_entropy_payload(
        payload_path,
        expected_epochs=plot_figure_s2b.DEFAULT_REFERENCE_EPOCHS,
        shot_budgets=plot_figure_s2b.DEFAULT_SHOT_BUDGETS,
    )
    captured_calls: list[dict[str, object]] = []

    def fake_plot_temporal_summary(ax, **kwargs):
        captured_calls.append(
            {
                "summary": kwargs["summary"],
                "color": kwargs["color"],
                "label": kwargs.get("label"),
                "linewidth": kwargs.get("linewidth", 1.5),
                "linestyle": kwargs.get("linestyle", "-"),
                "marker": kwargs.get("marker"),
                "markersize": kwargs.get("markersize"),
                "band_alpha": kwargs.get("band_alpha", 0.16),
                "show_band": kwargs.get("show_band", True),
            }
        )
        (line,) = ax.plot(
            kwargs["summary"].epoch,
            kwargs["summary"].mean,
            color=kwargs["color"],
            label=kwargs.get("label"),
        )
        return line

    monkeypatch.setattr(plot_figure_s2b, "plot_temporal_summary", fake_plot_temporal_summary)

    figure = plot_figure_s2b.plot_article_figure_s2b(payload_path=payload_path)

    assert len(captured_calls) == len(expected_series)
    for series_index, series in enumerate(expected_series):
        assert captured_calls[series_index]["summary"].epoch == series.summary.epoch
        assert captured_calls[series_index]["summary"].mean == pytest.approx(series.summary.mean)
        assert captured_calls[series_index]["summary"].lower == pytest.approx(series.summary.lower)
        assert captured_calls[series_index]["summary"].upper == pytest.approx(series.summary.upper)
        assert captured_calls[series_index]["color"] == f"C{series_index % 10}"
        assert captured_calls[series_index]["label"] == series.label
        assert captured_calls[series_index]["linewidth"] == 1.5
        assert captured_calls[series_index]["linestyle"] == "-"
        assert captured_calls[series_index]["marker"] == "o"
        assert captured_calls[series_index]["markersize"] == 3.0
        assert captured_calls[series_index]["band_alpha"] == 0.16
        assert captured_calls[series_index]["show_band"] is True

    plt.close(figure)


def test_parse_args_accepts_artifacts_and_output_dir_overrides() -> None:
    args = plot_figure_s2b.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")


def test_parse_args_accepts_temporal_summary_overrides() -> None:
    args = plot_figure_s2b.parse_args(
        [
            "--epoch-start",
            "100",
            "--epoch-end",
            "500",
            "--epoch-group-size",
            "2",
            "--lower-percentile",
            "10",
            "--upper-percentile",
            "90",
        ]
    )

    assert args.epoch_start == 100
    assert args.epoch_end == 500
    assert args.epoch_group_size == 2
    assert args.lower_percentile == 10.0
    assert args.upper_percentile == 90.0
