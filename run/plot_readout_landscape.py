from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qcnn import ARTICLE_PANEL_FIGSIZE
from qcnn.article_training import build_canonical_reference_run_directory_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_REFERENCE_DIRECTORY_NAME = build_canonical_reference_run_directory_name()
DEFAULT_PAYLOAD_FILENAME = "readout_landscape.pt"
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = (10, 100, 800)
DEFAULT_PANEL_FILENAMES: dict[int, str] = {
    10: "readout_landscape_epoch10.pdf",
    100: "readout_landscape_epoch100.pdf",
    800: "readout_landscape_epoch800.pdf",
}

# Heatmap cells with fewer than this fraction of valid test samples are masked
# and therefore rendered as white. The per-cell validity counts are computed
# upstream by `run/evaluate_readout_landscape.py`.
DEFAULT_MIN_VALID_FRACTION = 0.10


@dataclass(frozen=True)
class ReadoutLandscapePanel:
    epoch: int
    pc1_sigma: torch.Tensor
    pc2_sigma: torch.Tensor
    mean_loss: torch.Tensor
    valid_count: torch.Tensor
    valid_fraction: torch.Tensor
    total_samples: int
    eligible_sample_count: int


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_readout_landscape.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure S3."
        ) from exc
    return plt


def load_readout_landscape_payload(payload_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("readout landscape payload must deserialize to a mapping.")
    return dict(payload)


def summarize_readout_landscape_payload(
    payload_path: str | Path,
    *,
    expected_epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
) -> list[ReadoutLandscapePanel]:
    payload = load_readout_landscape_payload(payload_path)
    payload_epochs = payload.get("epochs")
    pc1_sigma = payload.get("pc1_sigma")
    pc2_sigma = payload.get("pc2_sigma")
    total_samples = payload.get("total_samples")
    evaluations = payload.get("evaluations")
    if list(payload_epochs or []) != list(expected_epochs):
        raise ValueError(f"Expected epoch grid {list(expected_epochs)}, got {payload_epochs!r}.")
    if not isinstance(pc1_sigma, torch.Tensor) or pc1_sigma.ndim != 1:
        raise ValueError("readout landscape payload must contain a 1D tensor 'pc1_sigma'.")
    if not isinstance(pc2_sigma, torch.Tensor) or pc2_sigma.ndim != 1:
        raise ValueError("readout landscape payload must contain a 1D tensor 'pc2_sigma'.")
    if not isinstance(total_samples, int) or total_samples <= 0:
        raise ValueError("readout landscape payload must contain a positive integer 'total_samples'.")
    if not isinstance(evaluations, list):
        raise ValueError("readout landscape payload must contain an 'evaluations' list.")

    evaluation_map: dict[int, ReadoutLandscapePanel] = {}
    expected_shape = (pc2_sigma.numel(), pc1_sigma.numel())
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each evaluation entry must be a mapping.")
        epoch = evaluation.get("epoch")
        mean_loss = evaluation.get("mean_loss")
        valid_count = evaluation.get("valid_count")
        valid_fraction = evaluation.get("valid_fraction")
        eligible_sample_count = evaluation.get("eligible_sample_count")
        if not isinstance(epoch, int):
            raise ValueError("Each evaluation entry must contain an integer 'epoch'.")
        if not isinstance(mean_loss, torch.Tensor) or mean_loss.shape != expected_shape:
            raise ValueError(f"Each evaluation entry must contain a tensor 'mean_loss' with shape {expected_shape}.")
        if not isinstance(valid_count, torch.Tensor) or valid_count.shape != expected_shape:
            raise ValueError(f"Each evaluation entry must contain a tensor 'valid_count' with shape {expected_shape}.")
        if not isinstance(valid_fraction, torch.Tensor) or valid_fraction.shape != expected_shape:
            raise ValueError(f"Each evaluation entry must contain a tensor 'valid_fraction' with shape {expected_shape}.")
        if not isinstance(eligible_sample_count, int) or eligible_sample_count < 0 or eligible_sample_count > total_samples:
            raise ValueError(
                f"Each evaluation entry must contain 'eligible_sample_count' between 0 and total_samples for epoch={epoch}."
            )
        if epoch in evaluation_map:
            raise ValueError(f"Duplicate readout-landscape evaluation for epoch={epoch}.")
        evaluation_map[epoch] = ReadoutLandscapePanel(
            epoch=epoch,
            pc1_sigma=pc1_sigma.clone(),
            pc2_sigma=pc2_sigma.clone(),
            mean_loss=mean_loss.clone(),
            valid_count=valid_count.clone(),
            valid_fraction=valid_fraction.clone(),
            total_samples=total_samples,
            eligible_sample_count=eligible_sample_count,
        )

    panels: list[ReadoutLandscapePanel] = []
    for epoch in expected_epochs:
        if epoch not in evaluation_map:
            raise ValueError(f"Missing readout-landscape evaluation for epoch={epoch}.")
        panels.append(evaluation_map[epoch])
    return panels


def _compute_panel_color_scale(panel: ReadoutLandscapePanel) -> tuple[float, float]:
    valid_mask = panel.valid_fraction >= DEFAULT_MIN_VALID_FRACTION
    if not valid_mask.any():
        raise ValueError(
            f"Readout-landscape panel for epoch={panel.epoch} does not contain any heatmap cells "
            "meeting the valid-sample threshold. Recompute it with "
            "`uv run run/evaluate_readout_landscape.py --rebuild`."
        )
    valid_values = panel.mean_loss[valid_mask]
    vmin = float(valid_values.min().item())
    vmax = float(valid_values.max().item())
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax


def plot_article_figure_s3_panel(
    *,
    panel: ReadoutLandscapePanel,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()
    vmin, vmax = _compute_panel_color_scale(panel)

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    masked_mean_loss = np.ma.masked_where(
        panel.valid_fraction.numpy() < DEFAULT_MIN_VALID_FRACTION,
        panel.mean_loss.numpy(),
    )
    image = ax.imshow(
        masked_mean_loss,
        origin="lower",
        extent=(
            float(panel.pc1_sigma[0].item()),
            float(panel.pc1_sigma[-1].item()),
            float(panel.pc2_sigma[0].item()),
            float(panel.pc2_sigma[-1].item()),
        ),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Local shot-noise axis 1 (sigma)")
    ax.set_ylabel("Local shot-noise axis 2 (sigma)")
    colorbar = figure.colorbar(image, ax=ax)
    colorbar.set_label("Mean cross-entropy (nats/sample)")

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure S3 from saved readout-landscape payloads.",
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
        help=(
            "Directory where readout_landscape_epoch10.pdf and "
            "readout_landscape_epoch100.pdf and "
            "readout_landscape_epoch800.pdf will be written."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()

    payload_path = (
        args.artifacts_root.expanduser().resolve()
        / DEFAULT_REFERENCE_DIRECTORY_NAME
        / DEFAULT_PAYLOAD_FILENAME
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = summarize_readout_landscape_payload(payload_path)
    output_paths: list[Path] = []
    for panel in panels:
        figure = plot_article_figure_s3_panel(panel=panel)
        try:
            output_name = DEFAULT_PANEL_FILENAMES[panel.epoch]
        except KeyError as exc:
            raise ValueError(f"No default output filename is defined for epoch {panel.epoch}.") from exc
        output_path = output_dir / output_name
        figure.savefig(output_path)
        plt.close(figure)
        output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
