import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PLOT_FIGURE_7_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_finite_shot_loss_histograms.py"
PLOT_FIGURE_7_SPEC = importlib.util.spec_from_file_location("plot_finite_shot_loss_histograms_module", PLOT_FIGURE_7_PATH)
if PLOT_FIGURE_7_SPEC is None or PLOT_FIGURE_7_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_finite_shot_loss_histograms.py from {PLOT_FIGURE_7_PATH}.")
plot_figure_7 = importlib.util.module_from_spec(PLOT_FIGURE_7_SPEC)
sys.modules[PLOT_FIGURE_7_SPEC.name] = plot_figure_7
PLOT_FIGURE_7_SPEC.loader.exec_module(plot_figure_7)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert plot_figure_7.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"


def write_fake_payload(payload_path: Path) -> None:
    payload = {
        "seed": 0,
        "epochs": list(plot_figure_7.DEFAULT_REFERENCE_EPOCHS),
        "shot_budgets": list(plot_figure_7.DEFAULT_SHOT_BUDGETS),
        "repetitions": 5,
        "batch_size": 250,
        "evaluations": [],
    }
    for epoch_index, epoch in enumerate(plot_figure_7.DEFAULT_REFERENCE_EPOCHS):
        for budget_index, shot_budget in enumerate(plot_figure_7.DEFAULT_SHOT_BUDGETS):
            base = 0.2 + 0.1 * epoch_index + 0.03 * budget_index
            batch_sizes = torch.tensor([250, 250, 100], dtype=torch.int64)
            if shot_budget is None:
                batch_mean_loss = torch.tensor(
                    [[base, base + 0.02, base + 0.04]],
                    dtype=torch.float32,
                )
                num_draws = 1
            else:
                batch_mean_loss = torch.tensor(
                    [
                        [base, base + 0.02, base + 0.04],
                        [base + 0.01, base + 0.03, base + 0.05],
                        [base + 0.02, base + 0.04, base + 0.06],
                        [base + 0.03, base + 0.05, base + 0.07],
                        [base + 0.04, base + 0.06, base + 0.08],
                    ],
                    dtype=torch.float32,
                )
                num_draws = 5
            payload["evaluations"].append(
                {
                    "epoch": epoch,
                    "shot_budget": shot_budget,
                    "num_draws": num_draws,
                    "batch_sizes": batch_sizes,
                    "batch_loss_sum": batch_mean_loss * batch_sizes.to(dtype=torch.float32),
                    "batch_correct_count": torch.tensor(
                        (
                            [[210, 211, 85]]
                            if shot_budget is None
                            else [
                                [200, 201, 80],
                                [201, 202, 81],
                                [202, 203, 82],
                                [203, 204, 83],
                                [204, 205, 84],
                            ]
                        ),
                        dtype=torch.int64,
                    ),
                }
            )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)
def test_summarize_finite_shot_loss_sampling_payload_builds_panels(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_figure_7.DEFAULT_REFERENCE_DIRECTORY_NAME
        / "finite_shot_loss_sampling.pt"
    )
    write_fake_payload(payload_path)

    panels = plot_figure_7.summarize_finite_shot_loss_sampling_payload(payload_path)

    assert [panel.epoch for panel in panels] == [100, 800]
    assert [series.label for series in panels[0].series_by_budget] == [
        "128 shots",
        "256 shots",
        "512 shots",
        "1024 shots",
        "Inf",
    ]
    expected_x_cap = max(series.weighted_mean_loss for series in panels[0].series_by_budget)
    assert panels[0].x_cap == pytest.approx(expected_x_cap)
    bin_widths = torch.tensor(panels[0].bins[1:] - panels[0].bins[:-1], dtype=torch.float64)
    assert torch.allclose(bin_widths, torch.full_like(bin_widths, bin_widths[0]))
    assert float(bin_widths[0]) == pytest.approx(panels[0].x_cap / plot_figure_7.DEFAULT_BASE_BIN_COUNT)
    assert len(panels[0].bins) >= plot_figure_7.DEFAULT_BASE_BIN_COUNT + 1
    max_value = max(max(series.batch_mean_loss) for series in panels[0].series_by_budget)
    assert panels[0].histogram_upper_edge >= max_value
    assert panels[0].histogram_upper_edge < max_value + float(bin_widths[0])
    assert len(panels[0].series_by_budget[0].batch_mean_loss) == 15
    assert len(panels[0].series_by_budget[-1].batch_mean_loss) == 3


def test_build_histogram_layers_returns_histogram_counts(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_figure_7.DEFAULT_REFERENCE_DIRECTORY_NAME
        / "finite_shot_loss_sampling.pt"
    )
    write_fake_payload(payload_path)
    panels = plot_figure_7.summarize_finite_shot_loss_sampling_payload(payload_path)

    layers = plot_figure_7.build_histogram_layers(panels[0])

    assert len(layers) == 5
    assert all(layer.relative_frequency.shape == layers[0].relative_frequency.shape for layer in layers)
    expected_shot_order = [128, 256, 512, 1024, None]
    assert [layer.series.shot_budget for layer in layers] == expected_shot_order
    assert [plot_figure_7.shot_budget_sort_key(layer.series.shot_budget) for layer in layers] == [
        (0, 128),
        (0, 256),
        (0, 512),
        (0, 1024),
        (1, 0),
    ]
    expected_bin_centers = ((panels[0].bins[:-1] + panels[0].bins[1:]) * 0.5).tolist()
    for layer in layers:
        expected_counts, _ = np.histogram(np.asarray(layer.series.batch_mean_loss, dtype=float), bins=panels[0].bins)
        expected_relative_frequency = expected_counts.astype(float) / float(expected_counts.sum())
        assert layer.bin_centers.tolist() == pytest.approx(expected_bin_centers)
        assert layer.relative_frequency.tolist() == pytest.approx(expected_relative_frequency.tolist())
        assert float(layer.relative_frequency.sum()) == pytest.approx(1.0)


def test_summarize_finite_shot_loss_sampling_payload_extends_bins_with_uniform_width(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_figure_7.DEFAULT_REFERENCE_DIRECTORY_NAME
        / "finite_shot_loss_sampling.pt"
    )
    write_fake_payload(payload_path)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    payload["evaluations"][0]["batch_loss_sum"][0, 0] = 5.0 * payload["evaluations"][0]["batch_sizes"][0].to(dtype=torch.float32)
    torch.save(payload, payload_path)

    panels = plot_figure_7.summarize_finite_shot_loss_sampling_payload(payload_path)

    bin_widths = panels[0].bins[1:] - panels[0].bins[:-1]
    assert bin_widths.tolist() == pytest.approx([float(bin_widths[0])] * len(bin_widths))
    assert panels[0].histogram_upper_edge >= 5.0
    assert panels[0].histogram_upper_edge < 5.0 + bin_widths[0]


def test_summarize_finite_shot_loss_sampling_payload_accepts_custom_base_bin_count(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_figure_7.DEFAULT_REFERENCE_DIRECTORY_NAME
        / "finite_shot_loss_sampling.pt"
    )
    write_fake_payload(payload_path)
    custom_base_bin_count = max(1, plot_figure_7.DEFAULT_BASE_BIN_COUNT // 2)

    default_panels = plot_figure_7.summarize_finite_shot_loss_sampling_payload(payload_path)
    custom_panels = plot_figure_7.summarize_finite_shot_loss_sampling_payload(
        payload_path,
        base_bin_count=custom_base_bin_count,
    )

    default_bin_width = default_panels[0].bins[1] - default_panels[0].bins[0]
    custom_bin_width = custom_panels[0].bins[1] - custom_panels[0].bins[0]
    assert default_bin_width == pytest.approx(default_panels[0].x_cap / plot_figure_7.DEFAULT_BASE_BIN_COUNT)
    assert custom_bin_width == pytest.approx(custom_panels[0].x_cap / float(custom_base_bin_count))
    assert custom_bin_width == pytest.approx(
        default_bin_width * (plot_figure_7.DEFAULT_BASE_BIN_COUNT / float(custom_base_bin_count))
    )
    assert len(default_panels[0].bins) >= plot_figure_7.DEFAULT_BASE_BIN_COUNT + 1
    assert len(custom_panels[0].bins) >= custom_base_bin_count + 1


def test_parse_args_defaults_to_project_artifacts_output_dir_and_num_bins() -> None:
    args = plot_figure_7.parse_args([])

    assert args.artifacts_root == plot_figure_7.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_7.DEFAULT_OUTPUT_DIR
    assert args.num_bins == plot_figure_7.DEFAULT_BASE_BIN_COUNT


def test_parse_args_accepts_artifacts_output_dir_and_num_bins_overrides() -> None:
    default_args = plot_figure_7.parse_args([])
    custom_num_bins = max(1, default_args.num_bins // 2)
    args = plot_figure_7.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
            "--num-bins",
            str(custom_num_bins),
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")
    assert args.num_bins == custom_num_bins
    assert args.num_bins != default_args.num_bins
