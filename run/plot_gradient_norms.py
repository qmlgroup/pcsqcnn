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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_INPUT_DIRECTORY_NAME = "depth_scaling_gradient_norms"
DEFAULT_PAYLOAD_FILENAME = "gradient_norms.pt"


@dataclass(frozen=True)
class GradientNormSeries:
    label: str
    summary: TemporalStatisticSummary
    color: str
    linestyle: str


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_gradient_norms.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure S2a."
        ) from exc
    return plt


def load_gradient_norms_payload(payload_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("gradient-norm payload must deserialize to a mapping.")
    return dict(payload)


def _summarize_gradient_norms_payload_mapping(
    payload: Mapping[str, Any],
    *,
    expected_depths: Sequence[int] | None = None,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> list[GradientNormSeries]:
    payload_depths = payload.get("depths")
    evaluations = payload.get("evaluations")

    resolved_depths = list(payload_depths or []) if expected_depths is None else list(expected_depths)
    if list(payload_depths or []) != resolved_depths:
        raise ValueError(f"Expected depth grid {resolved_depths}, got {payload_depths!r}.")
    if not isinstance(evaluations, list):
        raise ValueError("gradient-norm payload must contain an 'evaluations' list.")

    total_norm_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    first_norm_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    last_norm_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    total_rms_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    first_rms_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    last_rms_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}

    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each gradient-norm evaluation entry must be a mapping.")
        depth = evaluation.get("depth")
        full_quantum_gradient_norm = evaluation.get("full_quantum_gradient_norm")
        first_quantum_layer_gradient_norm = evaluation.get("first_quantum_layer_gradient_norm")
        last_quantum_layer_gradient_norm = evaluation.get("last_quantum_layer_gradient_norm")
        full_quantum_gradient_rms = evaluation.get("full_quantum_gradient_rms")
        first_quantum_layer_gradient_rms = evaluation.get("first_quantum_layer_gradient_rms")
        last_quantum_layer_gradient_rms = evaluation.get("last_quantum_layer_gradient_rms")

        if not isinstance(depth, int):
            raise ValueError("Each gradient-norm evaluation entry must contain an integer 'depth'.")
        if depth not in total_norm_by_depth:
            raise ValueError(f"Unexpected depth {depth!r} in gradient payload.")
        if (
            not isinstance(full_quantum_gradient_norm, torch.Tensor)
            or full_quantum_gradient_norm.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'full_quantum_gradient_norm' tensor."
            )
        if (
            not isinstance(first_quantum_layer_gradient_norm, torch.Tensor)
            or first_quantum_layer_gradient_norm.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'first_quantum_layer_gradient_norm' tensor."
            )
        if (
            not isinstance(last_quantum_layer_gradient_norm, torch.Tensor)
            or last_quantum_layer_gradient_norm.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'last_quantum_layer_gradient_norm' tensor."
            )
        if (
            not isinstance(full_quantum_gradient_rms, torch.Tensor)
            or full_quantum_gradient_rms.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'full_quantum_gradient_rms' tensor."
            )
        if (
            not isinstance(first_quantum_layer_gradient_rms, torch.Tensor)
            or first_quantum_layer_gradient_rms.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'first_quantum_layer_gradient_rms' tensor."
            )
        if (
            not isinstance(last_quantum_layer_gradient_rms, torch.Tensor)
            or last_quantum_layer_gradient_rms.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'last_quantum_layer_gradient_rms' tensor."
            )

        total_norm_by_depth[depth].append(float(full_quantum_gradient_norm.item()))
        first_norm_by_depth[depth].append(float(first_quantum_layer_gradient_norm.item()))
        last_norm_by_depth[depth].append(float(last_quantum_layer_gradient_norm.item()))
        total_rms_by_depth[depth].append(float(full_quantum_gradient_rms.item()))
        first_rms_by_depth[depth].append(float(first_quantum_layer_gradient_rms.item()))
        last_rms_by_depth[depth].append(float(last_quantum_layer_gradient_rms.item()))

    total_norm_samples_by_depth = [total_norm_by_depth[depth] for depth in resolved_depths]
    first_norm_samples_by_depth = [first_norm_by_depth[depth] for depth in resolved_depths]
    last_norm_samples_by_depth = [last_norm_by_depth[depth] for depth in resolved_depths]
    total_rms_samples_by_depth = [total_rms_by_depth[depth] for depth in resolved_depths]
    first_rms_samples_by_depth = [first_rms_by_depth[depth] for depth in resolved_depths]
    last_rms_samples_by_depth = [last_rms_by_depth[depth] for depth in resolved_depths]
    if any(not samples for samples in total_norm_samples_by_depth):
        raise ValueError("Every depth must have at least one total-gradient sample.")
    if any(not samples for samples in first_norm_samples_by_depth):
        raise ValueError("Every depth must have at least one first-layer sample.")
    if any(not samples for samples in last_norm_samples_by_depth):
        raise ValueError("Every depth must have at least one last-layer sample.")
    if any(not samples for samples in total_rms_samples_by_depth):
        raise ValueError("Every depth must have at least one total-gradient RMS sample.")
    if any(not samples for samples in first_rms_samples_by_depth):
        raise ValueError("Every depth must have at least one first-layer RMS sample.")
    if any(not samples for samples in last_rms_samples_by_depth):
        raise ValueError("Every depth must have at least one last-layer RMS sample.")

    return [
        GradientNormSeries(
            label="All quantum parameters (empirical loss)",
            summary=summarize_temporal_samples(
                resolved_depths,
                total_norm_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C0",
            linestyle="-",
        ),
        GradientNormSeries(
            label="First quantum layer (empirical loss)",
            summary=summarize_temporal_samples(
                resolved_depths,
                first_norm_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C1",
            linestyle="-",
        ),
        GradientNormSeries(
            label="Last quantum layer (empirical loss)",
            summary=summarize_temporal_samples(
                resolved_depths,
                last_norm_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C2",
            linestyle="-",
        ),
        GradientNormSeries(
            label="All quantum parameters (per-sample RMS)",
            summary=summarize_temporal_samples(
                resolved_depths,
                total_rms_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C0",
            linestyle="--",
        ),
        GradientNormSeries(
            label="First quantum layer (per-sample RMS)",
            summary=summarize_temporal_samples(
                resolved_depths,
                first_rms_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C1",
            linestyle="--",
        ),
        GradientNormSeries(
            label="Last quantum layer (per-sample RMS)",
            summary=summarize_temporal_samples(
                resolved_depths,
                last_rms_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
            color="C2",
            linestyle="--",
        ),
    ]


def summarize_gradient_norms_payload(
    payload_path: str | Path,
    *,
    expected_depths: Sequence[int] | None = None,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> list[GradientNormSeries]:
    payload = load_gradient_norms_payload(payload_path)
    return _summarize_gradient_norms_payload_mapping(
        payload,
        expected_depths=expected_depths,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )


def plot_article_figure_s2a(
    *,
    payload_path: str | Path,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
    log_y: bool = True,
) -> "Figure":
    plt = _require_matplotlib()
    from matplotlib.lines import Line2D

    payload = load_gradient_norms_payload(payload_path)
    series_list = _summarize_gradient_norms_payload_mapping(payload, expected_depths=None)

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for series in series_list:
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=series.color,
            linewidth=1.75,
            linestyle=series.linestyle,
            marker="o",
            markersize=3.5,
            label="_nolegend_",
            band_alpha=0.16,
        )

    ax.set_xlabel("Quantum depth $Q$")
    ax.set_ylabel("Initialization gradient norm")
    plotted_depths = series_list[0].summary.epoch
    ax.set_xlim(min(plotted_depths), max(plotted_depths))
    ax.set_yscale("log" if log_y else "linear")

    color_legend_handles = [
        Line2D(
            [0],
            [0],
            color="C0",
            marker="o",
            linewidth=1.75,
            markersize=3.5,
            label="All quantum parameters",
        ),
        Line2D(
            [0],
            [0],
            color="C1",
            marker="o",
            linewidth=1.75,
            markersize=3.5,
            label="First quantum layer",
        ),
        Line2D(
            [0],
            [0],
            color="C2",
            marker="o",
            linewidth=1.75,
            markersize=3.5,
            label="Last quantum layer",
        ),
    ]
    linestyle_legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=1.75,
            linestyle="-",
            label="Empirical-loss gradient",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            linewidth=1.75,
            linestyle="--",
            label="Per-sample RMS gradient",
        ),
    ]
    color_legend = ax.legend(
        handles=color_legend_handles,
        loc="upper right",
        framealpha=0.85,
    )
    ax.add_artist(color_legend)
    ax.legend(
        handles=linestyle_legend_handles,
        loc="lower left",
        framealpha=0.85,
    )

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure S2a from the depth-scaling gradient diagnostics payload.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the depth-scaling gradient-diagnostic payload.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where gradient_norms.pdf will be written.",
    )
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the y-axis on a log scale. Use --no-log-y to force a linear scale.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()

    payload_path = (
        args.artifacts_root.expanduser().resolve()
        / DEFAULT_INPUT_DIRECTORY_NAME
        / DEFAULT_PAYLOAD_FILENAME
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figure = plot_article_figure_s2a(payload_path=payload_path, log_y=args.log_y)
    output_path = output_dir / "gradient_norms.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
