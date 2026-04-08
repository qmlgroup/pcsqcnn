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
DEFAULT_PAYLOAD_FILENAME = "finite_shot_loss_sampling.pt"
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = (100, 800)
DEFAULT_SHOT_BUDGETS: tuple[int | None, ...] = (128, 256, 512, 1024, None)
DEFAULT_PANEL_FILENAMES: dict[int, str] = {
    100: "finite_shot_loss_histogram_epoch100.pdf",
    800: "finite_shot_loss_histogram_epoch800.pdf",
}
DEFAULT_BASE_BIN_COUNT = 30
FILL_ZORDER_BASE = 10.0
CONTOUR_ZORDER_BASE = 30.0
MEAN_LINE_ZORDER_BASE = 40.0
TEXT_ZORDER_BASE = 50.0
MEAN_LINE_WIDTH = 0.9
SHOT_LABEL_Y_FRACTION = 0.94
SHOT_LABEL_X_SHIFT_BIN_WIDTHS = 0.35


@dataclass(frozen=True)
class FiniteShotLossSamplingSeries:
    shot_budget: int | None
    batch_mean_loss: list[float]
    weighted_mean_loss: float

    @property
    def label(self) -> str:
        if self.shot_budget is None:
            return "Inf"
        return f"{self.shot_budget} shots"


@dataclass(frozen=True)
class FiniteShotLossSamplingPanel:
    epoch: int
    series_by_budget: list[FiniteShotLossSamplingSeries]
    x_cap: float
    histogram_upper_edge: float
    bins: np.ndarray


@dataclass(frozen=True)
class FiniteShotLossSamplingHistogramLayer:
    series_index: int
    series: FiniteShotLossSamplingSeries
    bin_centers: np.ndarray
    relative_frequency: np.ndarray


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_finite_shot_loss_histograms.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure 7."
        ) from exc
    return plt


def _lighten_matplotlib_color(color: str, *, blend: float = 0.78) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(color)
    return (
        blend + (1.0 - blend) * red,
        blend + (1.0 - blend) * green,
        blend + (1.0 - blend) * blue,
    )


def shot_budget_sort_key(shot_budget: int | None) -> tuple[int, int]:
    if shot_budget is None:
        return (1, 0)
    return (0, shot_budget)


def load_finite_shot_loss_sampling_payload(payload_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("finite-shot loss-sampling payload must deserialize to a mapping.")
    return dict(payload)


def summarize_finite_shot_loss_sampling_payload(
    payload_path: str | Path,
    *,
    expected_epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
    shot_budgets: Sequence[int | None] = DEFAULT_SHOT_BUDGETS,
    base_bin_count: int = DEFAULT_BASE_BIN_COUNT,
) -> list[FiniteShotLossSamplingPanel]:
    if base_bin_count <= 0:
        raise ValueError(f"base_bin_count must be positive, got {base_bin_count}.")
    payload = load_finite_shot_loss_sampling_payload(payload_path)
    payload_epochs = payload.get("epochs")
    evaluations = payload.get("evaluations")
    repetitions = payload.get("repetitions")
    batch_size = payload.get("batch_size")
    if list(payload_epochs or []) != list(expected_epochs):
        raise ValueError(
            f"Expected epoch grid {list(expected_epochs)}, got {payload_epochs!r}."
        )
    if not isinstance(evaluations, list):
        raise ValueError("finite-shot loss-sampling payload must contain an 'evaluations' list.")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError(f"finite-shot loss-sampling payload must contain a positive integer 'repetitions', got {repetitions!r}.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"finite-shot loss-sampling payload must contain a positive integer 'batch_size', got {batch_size!r}.")

    evaluation_map: dict[tuple[int, int | None], FiniteShotLossSamplingSeries] = {}
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each evaluation entry must be a mapping.")
        epoch = evaluation.get("epoch")
        shot_budget = evaluation.get("shot_budget")
        num_draws = evaluation.get("num_draws", repetitions)
        batch_sizes = evaluation.get("batch_sizes")
        batch_loss_sum = evaluation.get("batch_loss_sum")
        batch_correct_count = evaluation.get("batch_correct_count")
        if not isinstance(epoch, int):
            raise ValueError("Each evaluation entry must contain an integer 'epoch'.")
        if shot_budget is not None and not isinstance(shot_budget, int):
            raise ValueError("Each evaluation entry must contain an integer 'shot_budget' or None.")
        if not isinstance(num_draws, int) or num_draws <= 0:
            raise ValueError("Each evaluation entry must contain a positive integer 'num_draws'.")
        if not isinstance(batch_sizes, torch.Tensor) or batch_sizes.ndim != 1:
            raise ValueError("Each evaluation entry must contain a 1D tensor 'batch_sizes'.")
        if not isinstance(batch_loss_sum, torch.Tensor) or batch_loss_sum.ndim != 2:
            raise ValueError("Each evaluation entry must contain a 2D tensor 'batch_loss_sum'.")
        if not isinstance(batch_correct_count, torch.Tensor) or batch_correct_count.ndim != 2:
            raise ValueError("Each evaluation entry must contain a 2D tensor 'batch_correct_count'.")
        if batch_loss_sum.shape != batch_correct_count.shape:
            raise ValueError("batch_loss_sum and batch_correct_count must have the same shape.")
        if batch_loss_sum.shape[0] != num_draws:
            raise ValueError("Each evaluation entry must contain one batch aggregate row per recorded draw.")
        if batch_loss_sum.shape[1] != batch_sizes.numel():
            raise ValueError("batch_sizes length must match the number of batch aggregate columns.")
        key = (epoch, shot_budget)
        if key in evaluation_map:
            raise ValueError(f"Duplicate finite-shot loss-sampling entry for epoch={epoch}, shot_budget={shot_budget}.")
        batch_sizes_float = batch_sizes.to(dtype=batch_loss_sum.dtype)
        batch_mean_loss = (batch_loss_sum / batch_sizes_float.unsqueeze(0)).reshape(-1)
        weighted_mean_loss = float(batch_loss_sum.sum().item() / batch_sizes_float.sum().item() / num_draws)
        evaluation_map[key] = FiniteShotLossSamplingSeries(
            shot_budget=shot_budget,
            batch_mean_loss=batch_mean_loss.tolist(),
            weighted_mean_loss=weighted_mean_loss,
        )

    panels: list[FiniteShotLossSamplingPanel] = []
    for epoch in expected_epochs:
        series_by_budget: list[FiniteShotLossSamplingSeries] = []
        for shot_budget in shot_budgets:
            key = (epoch, shot_budget)
            if key not in evaluation_map:
                raise ValueError(f"Missing finite-shot loss-sampling entry for epoch={epoch}, shot_budget={shot_budget}.")
            series_by_budget.append(evaluation_map[key])

        x_cap = max(series.weighted_mean_loss for series in series_by_budget)
        max_value = max(max(series.batch_mean_loss) for series in series_by_budget)
        if x_cap <= 0.0:
            x_cap = max(max_value, 1.0)
        bin_width = x_cap / float(base_bin_count)
        if bin_width <= 0.0:
            bin_width = 1.0 / float(base_bin_count)
        histogram_upper_edge = max(x_cap, max_value)
        num_bins = max(base_bin_count, int(np.ceil(histogram_upper_edge / bin_width)))
        histogram_upper_edge = num_bins * bin_width
        bins = np.linspace(0.0, histogram_upper_edge, num_bins + 1)
        panels.append(
            FiniteShotLossSamplingPanel(
                epoch=epoch,
                series_by_budget=series_by_budget,
                x_cap=x_cap,
                histogram_upper_edge=histogram_upper_edge,
                bins=bins,
            )
        )

    return panels


def build_histogram_layers(panel: FiniteShotLossSamplingPanel) -> list[FiniteShotLossSamplingHistogramLayer]:
    layers: list[FiniteShotLossSamplingHistogramLayer] = []
    bin_centers = 0.5 * (panel.bins[:-1] + panel.bins[1:])
    for index, series in enumerate(panel.series_by_budget):
        counts = np.histogram(np.asarray(series.batch_mean_loss, dtype=float), bins=panel.bins)[0].astype(float)
        total_count = float(counts.sum())
        relative_frequency = counts / total_count if total_count > 0.0 else np.zeros_like(counts)
        layers.append(
            FiniteShotLossSamplingHistogramLayer(
                series_index=index,
                series=series,
                bin_centers=bin_centers,
                relative_frequency=relative_frequency,
            )
        )
    return layers


def plot_article_figure_7_panel(
    *,
    panel: FiniteShotLossSamplingPanel,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    histogram_layers = build_histogram_layers(panel)
    fill_layers = sorted(histogram_layers, key=lambda layer: shot_budget_sort_key(layer.series.shot_budget))
    for fill_index, layer in enumerate(fill_layers):
        color = f"C{layer.series_index}"
        fill_x = np.concatenate(
            [
                np.array([panel.bins[0]], dtype=float),
                layer.bin_centers,
                np.array([panel.bins[-1]], dtype=float),
            ]
        )
        fill_y = np.concatenate(
            [
                np.array([0.0], dtype=float),
                layer.relative_frequency,
                np.array([0.0], dtype=float),
            ]
        )
        ax.fill_between(
            fill_x,
            np.zeros_like(fill_y),
            fill_y,
            color=_lighten_matplotlib_color(color),
            linewidth=0.0,
            zorder=FILL_ZORDER_BASE + fill_index,
        )

    step_handles = []
    mean_positions: list[tuple[float, str, str, int]] = []
    for contour_index, layer in enumerate(histogram_layers):
        color = f"C{layer.series_index}"
        series = layer.series
        (step_line,) = ax.plot(
            layer.bin_centers,
            layer.relative_frequency,
            color=color,
            linewidth=1.5,
            label=series.label,
            zorder=CONTOUR_ZORDER_BASE + contour_index,
        )
        step_handles.append(step_line)
        mean_position = float(series.weighted_mean_loss)
        ax.axvline(
            mean_position,
            color=color,
            linestyle="-",
            linewidth=MEAN_LINE_WIDTH,
            zorder=MEAN_LINE_ZORDER_BASE + contour_index,
        )
        mean_positions.append(
            (
                mean_position,
                "Inf" if series.shot_budget is None else str(series.shot_budget),
                color,
                contour_index,
            )
        )

    ax.set_xlabel("Batch-mean test cross-entropy (nats/sample)")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0.0, panel.histogram_upper_edge)
    ax.set_ylim(bottom=0.0)
    ax.legend(handles=step_handles, loc="upper right", framealpha=0.8)

    y_max = ax.get_ylim()[1]
    x_shift = (panel.bins[1] - panel.bins[0]) * SHOT_LABEL_X_SHIFT_BIN_WIDTHS
    y_position = y_max * SHOT_LABEL_Y_FRACTION
    for mean_position, shot_label, color, index in mean_positions:
        ax.text(
            mean_position + x_shift,
            y_position,
            shot_label,
            color="black",
            rotation=90,
            va="top",
            ha="left",
            zorder=TEXT_ZORDER_BASE + index,
        )

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 7 from saved repeated finite-shot loss samples.",
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
            "Directory where finite_shot_loss_histogram_epoch100.pdf and "
            "finite_shot_loss_histogram_epoch800.pdf will be written."
        ),
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_BASE_BIN_COUNT,
        help="Base number of histogram bins on [0, x_cap] for Figure 7.",
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

    panels = summarize_finite_shot_loss_sampling_payload(
        payload_path,
        base_bin_count=args.num_bins,
    )
    output_paths: list[Path] = []
    for panel in panels:
        figure = plot_article_figure_7_panel(panel=panel)
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
