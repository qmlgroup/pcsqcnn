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
from qcnn.article_training import (
    DEFAULT_S2_ENTROPY_SHOT_BUDGETS,
    DEFAULT_S2_REFERENCE_EPOCHS,
    build_canonical_reference_run_directory_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_REFERENCE_DIRECTORY_NAME = build_canonical_reference_run_directory_name()
DEFAULT_PAYLOAD_FILENAME = "readout_entropy.pt"
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = DEFAULT_S2_REFERENCE_EPOCHS
DEFAULT_SHOT_BUDGETS: tuple[int, ...] = DEFAULT_S2_ENTROPY_SHOT_BUDGETS


@dataclass(frozen=True)
class ReadoutEntropySeries:
    shot_budget: int
    summary: TemporalStatisticSummary

    @property
    def label(self) -> str:
        return f"{self.shot_budget} shots"


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_readout_entropy.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure S2b."
        ) from exc
    return plt


def load_readout_entropy_payload(payload_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("readout-entropy payload must deserialize to a mapping.")
    return dict(payload)


def summarize_readout_entropy_payload(
    payload_path: str | Path,
    *,
    expected_epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
    shot_budgets: Sequence[int] = DEFAULT_SHOT_BUDGETS,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
) -> list[ReadoutEntropySeries]:
    payload = load_readout_entropy_payload(payload_path)
    payload_epochs = payload.get("epochs")
    payload_shot_budgets = payload.get("shot_budgets")
    evaluations = payload.get("evaluations")

    if list(payload_epochs or []) != list(expected_epochs):
        raise ValueError(f"Expected epoch grid {list(expected_epochs)}, got {payload_epochs!r}.")
    if list(payload_shot_budgets or []) != list(shot_budgets):
        raise ValueError(f"Expected shot budgets {list(shot_budgets)}, got {payload_shot_budgets!r}.")
    if not isinstance(evaluations, list):
        raise ValueError("readout-entropy payload must contain an 'evaluations' list.")

    evaluation_map: dict[tuple[int, int], torch.Tensor] = {}
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping):
            raise ValueError("Each readout-entropy evaluation entry must be a mapping.")
        epoch = evaluation.get("epoch")
        shot_budget = evaluation.get("shot_budget")
        entropy = evaluation.get("entropy")
        if not isinstance(epoch, int):
            raise ValueError("Each readout-entropy evaluation entry must contain an integer 'epoch'.")
        if not isinstance(shot_budget, int):
            raise ValueError("Each readout-entropy evaluation entry must contain an integer 'shot_budget'.")
        if not isinstance(entropy, torch.Tensor) or entropy.ndim != 1:
            raise ValueError("Each readout-entropy evaluation entry must contain a 1D 'entropy' tensor.")
        key = (epoch, shot_budget)
        if key in evaluation_map:
            raise ValueError(
                f"Duplicate readout-entropy evaluation for epoch={epoch}, shot_budget={shot_budget}."
            )
        evaluation_map[key] = entropy.clone()

    series_list: list[ReadoutEntropySeries] = []
    for shot_budget in shot_budgets:
        samples_by_epoch: list[list[float]] = []
        for epoch in expected_epochs:
            entropy = evaluation_map.get((epoch, shot_budget))
            if entropy is None:
                raise ValueError(
                    f"Missing readout-entropy evaluation for epoch={epoch}, shot_budget={shot_budget}."
                )
            samples_by_epoch.append([float(value) for value in entropy.tolist()])
        series_list.append(
            ReadoutEntropySeries(
                shot_budget=shot_budget,
                summary=summarize_temporal_samples(
                    expected_epochs,
                    samples_by_epoch,
                    epoch_start=epoch_start,
                    epoch_end=epoch_end,
                    epoch_group_size=epoch_group_size,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                ),
            )
        )
    return series_list


def plot_article_figure_s2b(
    *,
    payload_path: str | Path,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()
    series_list = summarize_readout_entropy_payload(
        payload_path,
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
    for series_index, series in enumerate(series_list):
        color = f"C{series_index % 10}"
        plot_temporal_summary(
            ax,
            summary=series.summary,
            color=color,
            label=series.label,
            linewidth=1.5,
            linestyle="-",
            band_alpha=0.16,
            marker="o",
            markersize=3.0,
        )
        all_epoch_values.extend(series.summary.epoch)

    ax.set_xlabel("Epoch checkpoint")
    ax.set_ylabel("Shannon entropy (nats)")
    ax.set_xlim(min(all_epoch_values), max(all_epoch_values))
    ax.legend(loc="upper right", framealpha=0.8)

    figure.tight_layout()
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure S2b from saved readout-entropy diagnostics.",
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
        help="Directory where readout_entropy.pdf will be written.",
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

    figure = plot_article_figure_s2b(
        payload_path=payload_path,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end,
        epoch_group_size=args.epoch_group_size,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    output_path = output_dir / "readout_entropy.pdf"
    figure.savefig(output_path)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    main()
