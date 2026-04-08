import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

PLOT_FIGURE_S3_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_readout_landscape.py"
PLOT_FIGURE_S3_SPEC = importlib.util.spec_from_file_location("plot_readout_landscape_module", PLOT_FIGURE_S3_PATH)
if PLOT_FIGURE_S3_SPEC is None or PLOT_FIGURE_S3_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_readout_landscape.py from {PLOT_FIGURE_S3_PATH}.")
plot_figure_s3 = importlib.util.module_from_spec(PLOT_FIGURE_S3_SPEC)
sys.modules[PLOT_FIGURE_S3_SPEC.name] = plot_figure_s3
PLOT_FIGURE_S3_SPEC.loader.exec_module(plot_figure_s3)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert plot_figure_s3.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"
    assert plot_figure_s3.DEFAULT_MIN_VALID_FRACTION == pytest.approx(0.10)


def write_fake_payload(payload_path: Path) -> None:
    pc1_sigma = torch.linspace(-3.0, 3.0, 5)
    pc2_sigma = torch.linspace(-3.0, 3.0, 5)
    epoch10_valid_count = torch.tensor(
        [
            [1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 1],
        ],
        dtype=torch.int64,
    )
    epoch800_valid_count = torch.tensor(
        [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 1, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=torch.int64,
    )
    payload = {
        "seed": 7,
        "epochs": [10, 100, 800],
        "pc1_sigma": pc1_sigma,
        "pc2_sigma": pc2_sigma,
        "sigma_shot_budget": 1024,
        "basis_mode": "sample_local_pca_exact_readout_shot_noise_covariance",
        "total_samples": 20,
        "evaluations": [
            {
                "epoch": 10,
                "mean_loss": torch.tensor(
                    [
                        [0.01, 0.2, 0.3, 0.4, 0.5],
                        [0.2, 0.3, 0.4, 0.5, 0.6],
                        [0.3, 0.4, 0.5, 0.6, 0.7],
                        [0.4, 0.5, 0.6, 0.7, 0.8],
                        [0.5, 0.6, 0.7, 0.8, 0.95],
                    ],
                    dtype=torch.float32,
                ),
                "valid_count": epoch10_valid_count,
                "valid_fraction": epoch10_valid_count.to(dtype=torch.float32) / 20.0,
                "eligible_sample_count": 18,
            },
            {
                "epoch": 100,
                "mean_loss": torch.tensor(
                    [
                        [0.12, 0.18, 0.24, 0.30, 0.36],
                        [0.18, 0.24, 0.30, 0.36, 0.42],
                        [0.24, 0.30, 0.36, 0.42, 0.48],
                        [0.30, 0.36, 0.42, 0.48, 0.54],
                        [0.36, 0.42, 0.48, 0.54, 0.60],
                    ],
                    dtype=torch.float32,
                ),
                "valid_count": torch.full((5, 5), 2, dtype=torch.int64),
                "valid_fraction": torch.full((5, 5), 0.10, dtype=torch.float32),
                "eligible_sample_count": 19,
            },
            {
                "epoch": 800,
                "mean_loss": torch.tensor(
                    [
                        [0.9, 0.8, 0.7, 0.6, 0.5],
                        [0.8, 0.7, 0.6, 0.5, 0.4],
                        [0.7, 0.6, 0.5, 0.4, 0.3],
                        [0.6, 0.5, 0.4, 0.3, 0.2],
                        [0.5, 0.4, 0.3, 0.2, 0.1],
                    ],
                    dtype=torch.float32,
                ),
                "valid_count": epoch800_valid_count,
                "valid_fraction": epoch800_valid_count.to(dtype=torch.float32) / 20.0,
                "eligible_sample_count": 17,
            },
        ],
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, payload_path)


def test_summarize_readout_landscape_payload_builds_panels(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s3.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_landscape.pt"
    )
    write_fake_payload(payload_path)

    panels = plot_figure_s3.summarize_readout_landscape_payload(payload_path)

    assert [panel.epoch for panel in panels] == [10, 100, 800]
    assert panels[0].mean_loss.shape == (5, 5)
    assert panels[0].valid_count.shape == (5, 5)
    assert panels[0].valid_fraction[0, 0].item() == pytest.approx(0.05)
    assert panels[0].eligible_sample_count == 18
    assert panels[1].eligible_sample_count == 19
    assert panels[2].eligible_sample_count == 17


def test_compute_panel_color_scale_uses_only_valid_cells(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s3.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_landscape.pt"
    )
    write_fake_payload(payload_path)
    panels = plot_figure_s3.summarize_readout_landscape_payload(payload_path)

    assert plot_figure_s3._compute_panel_color_scale(panels[0]) == pytest.approx((0.2, 0.8))
    assert plot_figure_s3._compute_panel_color_scale(panels[1]) == pytest.approx((0.12, 0.60))
    assert plot_figure_s3._compute_panel_color_scale(panels[2]) == pytest.approx((0.1, 0.9))


def test_compute_panel_color_scale_rejects_panels_without_valid_cells(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s3.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_landscape.pt"
    )
    write_fake_payload(payload_path)
    panels = plot_figure_s3.summarize_readout_landscape_payload(payload_path)
    invalid_panel = plot_figure_s3.ReadoutLandscapePanel(
        epoch=panels[0].epoch,
        pc1_sigma=panels[0].pc1_sigma,
        pc2_sigma=panels[0].pc2_sigma,
        mean_loss=panels[0].mean_loss,
        valid_count=torch.zeros_like(panels[0].valid_count),
        valid_fraction=torch.zeros_like(panels[0].valid_fraction),
        total_samples=panels[0].total_samples,
        eligible_sample_count=panels[0].eligible_sample_count,
    )

    with pytest.raises(ValueError, match="epoch=10"):
        plot_figure_s3._compute_panel_color_scale(invalid_panel)


def test_plot_article_figure_s3_panel_masks_low_valid_cells_and_sets_new_axis_labels(tmp_path: Path) -> None:
    payload_path = (
        tmp_path / "artifacts" / plot_figure_s3.DEFAULT_REFERENCE_DIRECTORY_NAME / "readout_landscape.pt"
    )
    write_fake_payload(payload_path)
    panels = plot_figure_s3.summarize_readout_landscape_payload(payload_path)

    figure = plot_figure_s3.plot_article_figure_s3_panel(panel=panels[0])

    assert figure.axes[0].get_xlabel() == "Local shot-noise axis 1 (sigma)"
    assert figure.axes[0].get_ylabel() == "Local shot-noise axis 2 (sigma)"
    masked = figure.axes[0].images[0].get_array()
    assert figure.axes[0].images[0].norm.vmin == pytest.approx(0.2)
    assert figure.axes[0].images[0].norm.vmax == pytest.approx(0.8)
    assert bool(masked.mask[0, 0]) is True
    assert bool(masked.mask[4, 4]) is True
    assert bool(masked.mask[2, 2]) is False

    plt.close(figure)


def test_parse_args_defaults_to_project_artifacts_and_output_dirs() -> None:
    args = plot_figure_s3.parse_args([])

    assert args.artifacts_root == plot_figure_s3.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_s3.DEFAULT_OUTPUT_DIR


def test_parse_args_accepts_artifacts_and_output_dir_overrides() -> None:
    args = plot_figure_s3.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")
