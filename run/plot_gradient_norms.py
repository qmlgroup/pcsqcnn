from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
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
DEFAULT_THEOREM_EPSILON = 0.0
DEFAULT_THEOREM_INDEX_DIMENSIONS = 2
DEFAULT_THEOREM_NUM_CLASSES = 10
THEOREM_LOWER_BOUND_LABEL = r"Theorem lower bound ($\varepsilon=0$)"


@dataclass(frozen=True)
class GradientNormSeries:
    label: str
    summary: TemporalStatisticSummary


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

    total_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}
    last_by_depth: dict[int, list[float]] = {depth: [] for depth in resolved_depths}

    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each gradient-norm evaluation entry must be a mapping.")
        depth = evaluation.get("depth")
        full_quantum_gradient_rms = evaluation.get("full_quantum_gradient_rms")
        last_quantum_layer_gradient_rms = evaluation.get("last_quantum_layer_gradient_rms")

        if not isinstance(depth, int):
            raise ValueError("Each gradient-norm evaluation entry must contain an integer 'depth'.")
        if depth not in total_by_depth:
            raise ValueError(f"Unexpected depth {depth!r} in gradient payload.")
        if (
            not isinstance(full_quantum_gradient_rms, torch.Tensor)
            or full_quantum_gradient_rms.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'full_quantum_gradient_rms' tensor."
            )
        if (
            not isinstance(last_quantum_layer_gradient_rms, torch.Tensor)
            or last_quantum_layer_gradient_rms.ndim != 0
        ):
            raise ValueError(
                "Each gradient-norm evaluation entry must contain a scalar "
                "'last_quantum_layer_gradient_rms' tensor."
            )

        total_by_depth[depth].append(float(full_quantum_gradient_rms.item()))
        last_by_depth[depth].append(float(last_quantum_layer_gradient_rms.item()))

    total_samples_by_depth = [total_by_depth[depth] for depth in resolved_depths]
    last_samples_by_depth = [last_by_depth[depth] for depth in resolved_depths]
    if any(not samples for samples in total_samples_by_depth):
        raise ValueError("Every depth must have at least one total-gradient sample.")
    if any(not samples for samples in last_samples_by_depth):
        raise ValueError("Every depth must have at least one last-layer sample.")

    return [
        GradientNormSeries(
            label="All quantum parameters",
            summary=summarize_temporal_samples(
                resolved_depths,
                total_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
        ),
        GradientNormSeries(
            label="Last quantum layer",
            summary=summarize_temporal_samples(
                resolved_depths,
                last_samples_by_depth,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            ),
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


def resolve_default_linear_weight_variance(*, fan_in: int) -> float:
    if fan_in < 1:
        raise ValueError(f"fan_in must be positive, got {fan_in}.")
    # torch.nn.Linear defaults to U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
    # whose variance equals 1 / (3 * fan_in).
    return 1.0 / (3.0 * float(fan_in))


def compute_theorem_rms_lower_bound(
    payload: Mapping[str, Any],
    *,
    epsilon: float = DEFAULT_THEOREM_EPSILON,
    index_dimensions: int = DEFAULT_THEOREM_INDEX_DIMENSIONS,
    num_classes: int = DEFAULT_THEOREM_NUM_CLASSES,
) -> float:
    post_pooling_index_qubits = payload.get("post_pooling_index_qubits")
    feature_qubits = payload.get("feature_qubits")

    if not isinstance(post_pooling_index_qubits, int) or post_pooling_index_qubits < 0:
        raise ValueError("gradient-norm payload must contain a non-negative integer 'post_pooling_index_qubits'.")
    if not isinstance(feature_qubits, int) or feature_qubits < 1:
        raise ValueError("gradient-norm payload must contain a positive integer 'feature_qubits'.")
    if index_dimensions < 1:
        raise ValueError(f"index_dimensions must be positive, got {index_dimensions}.")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}.")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}.")

    d_idx = 2 ** (index_dimensions * post_pooling_index_qubits)
    d_f = 2 ** feature_qubits
    d_out = d_idx * d_f
    sigma_w2 = resolve_default_linear_weight_variance(fan_in=d_out)
    local_term = d_f / (2.0 * (d_f + 1.0) ** 2) - epsilon * (d_f + 1.0 / (2.0 * (d_f + 1.0)))
    if local_term <= 0.0:
        raise ValueError(
            "The chosen theorem epsilon does not yield a positive lower bound "
            f"for D_f={d_f}: local_term={local_term}."
        )

    squared_lower_bound = (
        (sigma_w2 / d_out)
        * (1.0 - 1.0 / num_classes)
        * (1.0 / (d_idx**2 * 2**index_dimensions))
        * local_term
    )
    return math.sqrt(squared_lower_bound)


def plot_article_figure_s2a(
    *,
    payload_path: str | Path,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
    log_y: bool = True,
) -> "Figure":
    plt = _require_matplotlib()
    payload = load_gradient_norms_payload(payload_path)
    series_list = _summarize_gradient_norms_payload_mapping(payload, expected_depths=None)
    theorem_rms_lower_bound = compute_theorem_rms_lower_bound(payload)

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for series_index, series in enumerate(series_list):
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=f"C{series_index % 10}",
            linewidth=1.75,
            linestyle="-",
            marker="o",
            markersize=3.5,
            label=series.label,
            band_alpha=0.16,
        )

    ax.axhline(
        theorem_rms_lower_bound,
        color="black",
        linewidth=1.25,
        linestyle="--",
        alpha=0.9,
        label=THEOREM_LOWER_BOUND_LABEL,
    )

    ax.set_xlabel("Quantum depth $Q$")
    ax.set_ylabel("Initialization RMS gradient norm")
    plotted_depths = series_list[0].summary.epoch
    ax.set_xlim(min(plotted_depths), max(plotted_depths))
    ax.set_yscale("log" if log_y else "linear")
    ax.legend(loc="center right", framealpha=0.85)

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
