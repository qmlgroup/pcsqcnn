"""Reusable helpers for article figures built from saved training artifacts.

This module intentionally stays generic: it loads saved training histories,
aggregates epoch-level statistics, and provides plotting helpers for temporal
article figures. Figure-specific rendering belongs in the script that produces
the corresponding figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from qcnn.model import TrainingHistory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

ARTICLE_PANEL_FIGSIZE: tuple[float, float] = (6,4)


def figure_2_fixed_run_directory_name(base_name: str, *, samples_per_class: int) -> str:
    """Return the Figure 2 fixed-run directory name for one sample-count setting."""

    if samples_per_class <= 0:
        raise ValueError(f"samples_per_class must be positive, got {samples_per_class}.")
    return f"{base_name}_spc{samples_per_class}"


def _load_run_manifest(run_directory: str | Path) -> tuple[Path, Mapping[str, object]]:
    resolved_directory = Path(run_directory)
    manifest_path = resolved_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run directory is missing manifest.json: {resolved_directory}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json must deserialize to a mapping.")
    return resolved_directory, manifest


def load_saved_run_seeds(run_directory: str | Path) -> list[int]:
    """Load the saved seed order recorded in a run manifest."""

    resolved_directory, manifest = _load_run_manifest(run_directory)
    seed_order = manifest.get("seeds")
    if not isinstance(seed_order, list) or not all(isinstance(seed, int) for seed in seed_order):
        raise ValueError("manifest.json must contain a 'seeds' list of integers.")
    if len(set(seed_order)) != len(seed_order):
        raise ValueError("manifest.json 'seeds' must not contain duplicates.")
    if not seed_order:
        raise ValueError(f"manifest.json must contain at least one saved seed: {resolved_directory}")
    return list(seed_order)


def resolve_saved_run_seed(
    run_directory: str | Path,
    *,
    seed: int | None = None,
) -> int:
    """Resolve one concrete saved seed from a run manifest."""

    available_seeds = load_saved_run_seeds(run_directory)
    if seed is None:
        return available_seeds[0]
    if seed not in available_seeds:
        raise ValueError(
            f"Seed {seed} is not present in manifest.json for {Path(run_directory)}. "
            f"Available seeds: {available_seeds}."
        )
    return seed


def _load_manifest_seed_result_paths(run_directory: str | Path) -> list[tuple[int, Path]]:
    resolved_directory, manifest = _load_run_manifest(run_directory)

    seed_order = load_saved_run_seeds(resolved_directory)

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("manifest.json must contain a 'runs' list.")

    result_paths_by_seed: dict[int, Path] = {}
    for run_entry in runs:
        if not isinstance(run_entry, Mapping):
            raise ValueError("Each manifest run entry must be a mapping.")

        seed = run_entry.get("seed")
        result_name = run_entry.get("result")
        if not isinstance(seed, int):
            raise ValueError("Each manifest run entry must contain an integer 'seed'.")
        if not isinstance(result_name, str) or not result_name:
            raise ValueError("Each manifest run entry must contain a non-empty 'result' path.")
        if seed in result_paths_by_seed:
            raise ValueError(f"manifest.json contains duplicate run entries for seed {seed}.")
        result_paths_by_seed[seed] = resolved_directory / result_name

    ordered_paths: list[tuple[int, Path]] = []
    for seed in seed_order:
        result_path = result_paths_by_seed.get(seed)
        if result_path is None:
            raise ValueError(f"manifest.json is missing a run entry for seed {seed}.")
        if not result_path.is_file():
            raise ValueError(f"Manifest result file is missing for seed {seed}: {result_path}")
        ordered_paths.append((seed, result_path))
    return ordered_paths


def _load_saved_result_payload(result_path: Path, *, expected_seed: int) -> Mapping[str, object]:
    payload = torch.load(result_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run result must deserialize to a mapping: {result_path}")

    saved_seed = payload.get("seed")
    if saved_seed != expected_seed:
        raise ValueError(
            f"Run result seed mismatch for {result_path}: expected {expected_seed}, got {saved_seed!r}."
        )
    return payload


@dataclass(frozen=True)
class MetricHistorySummary:
    """Per-epoch aggregate statistics for one metric across multiple runs."""

    epoch: list[int]
    mean: list[float]
    std: list[float]


@dataclass(frozen=True)
class TemporalStatisticSummary:
    """Per-point temporal statistics after optional epoch pooling."""

    epoch: list[float]
    mean: list[float]
    lower: list[float] | None
    upper: list[float] | None


def load_saved_training_histories(run_directory: str | Path) -> list[TrainingHistory]:
    """Load training histories for all seeds recorded in ``manifest.json``.

    Args:
        run_directory: Directory containing one saved automated-training run.

    Returns:
        Training histories ordered by the manifest seed list.
    """

    histories: list[TrainingHistory] = []
    for seed, result_path in _load_manifest_seed_result_paths(run_directory):
        payload = _load_saved_result_payload(result_path, expected_seed=seed)
        training_history = payload.get("training_history")
        if not isinstance(training_history, TrainingHistory):
            raise ValueError(f"Run result is missing a valid training_history: {result_path}")

        histories.append(training_history)

    return histories


def load_saved_parameter_stats_line(run_directory: str | Path) -> str:
    """Load one saved per-model parameter summary line for a run directory."""

    observed_lines: list[str] = []
    for seed, result_path in _load_manifest_seed_result_paths(run_directory):
        payload = _load_saved_result_payload(result_path, expected_seed=seed)
        parameter_stats_line = payload.get("parameter_stats_line")
        if parameter_stats_line is None:
            raise ValueError(f"Run result is missing parameter_stats_line: {result_path}")
        if not isinstance(parameter_stats_line, str):
            raise ValueError(
                f"Run result parameter_stats_line must be a string when present: {result_path}"
            )
        observed_lines.append(parameter_stats_line)

    if not observed_lines:
        raise ValueError(f"Run directory does not contain any saved parameter_stats_line values: {run_directory}")
    if len(set(observed_lines)) != 1:
        raise ValueError("Run result parameter_stats_line values do not match across seeds.")
    return observed_lines[0]


def summarize_temporal_samples(
    epoch: Sequence[int | float],
    samples_by_epoch: Sequence[Sequence[float]],
    *,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = 1,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    compute_band: bool = True,
) -> TemporalStatisticSummary:
    """Summarize scalar temporal samples with optional epoch-width grouping."""

    if epoch_start is not None and epoch_end is not None and epoch_start > epoch_end:
        raise ValueError(f"epoch_start must be <= epoch_end, got {epoch_start} > {epoch_end}.")
    if not epoch:
        raise ValueError("epoch grid must contain at least one point.")
    if len(samples_by_epoch) != len(epoch):
        raise ValueError(
            "samples_by_epoch length must match epoch length: "
            f"{len(samples_by_epoch)} != {len(epoch)}."
        )
    if epoch_group_size < 1:
        raise ValueError(f"epoch_group_size must be at least 1, got {epoch_group_size}.")
    if not 0.0 <= lower_percentile <= upper_percentile <= 100.0:
        raise ValueError(
            "Percentiles must satisfy 0 <= lower_percentile <= upper_percentile <= 100."
        )

    normalized_epoch = [float(value) for value in epoch]
    for previous_epoch, current_epoch in zip(normalized_epoch, normalized_epoch[1:]):
        if current_epoch <= previous_epoch:
            raise ValueError("epoch grid must be strictly increasing.")

    filtered_points: list[tuple[float, list[float]]] = []
    for epoch_value, epoch_samples in zip(normalized_epoch, samples_by_epoch, strict=True):
        if not epoch_samples:
            raise ValueError("Each epoch must contribute at least one sample.")
        if (epoch_start is not None and epoch_value < epoch_start) or (
            epoch_end is not None and epoch_value > epoch_end
        ):
            continue
        filtered_points.append((epoch_value, [float(sample) for sample in epoch_samples]))

    if not filtered_points:
        raise ValueError("The requested epoch window does not contain any history points.")

    summarized_epoch: list[float] = []
    summarized_mean: list[float] = []
    summarized_lower: list[float] | None = [] if compute_band else None
    summarized_upper: list[float] | None = [] if compute_band else None

    anchor_epoch = filtered_points[0][0]
    epoch_bin_width = float(epoch_group_size)
    grouped_bins: list[list[tuple[float, list[float]]]] = []
    for epoch_value, epoch_samples in filtered_points:
        bin_index = math.floor(((epoch_value - anchor_epoch) / epoch_bin_width) + 1e-12)
        while len(grouped_bins) <= bin_index:
            grouped_bins.append([])
        grouped_bins[bin_index].append((epoch_value, epoch_samples))

    for grouped_points in grouped_bins:
        if not grouped_points:
            continue
        grouped_epoch = [point[0] for point in grouped_points]
        pooled_samples = [sample for _, samples in grouped_points for sample in samples]
        pooled_tensor = torch.tensor(pooled_samples, dtype=torch.float64)

        summarized_epoch.append(sum(grouped_epoch) / len(grouped_epoch))
        summarized_mean.append(float(pooled_tensor.mean().item()))
        if compute_band:
            quantiles = torch.quantile(
                pooled_tensor,
                torch.tensor(
                    [lower_percentile / 100.0, upper_percentile / 100.0],
                    dtype=pooled_tensor.dtype,
                    device=pooled_tensor.device,
                ),
            )
            assert summarized_lower is not None
            assert summarized_upper is not None
            summarized_lower.append(float(quantiles[0].item()))
            summarized_upper.append(float(quantiles[1].item()))

    return TemporalStatisticSummary(
        epoch=summarized_epoch,
        mean=summarized_mean,
        lower=summarized_lower,
        upper=summarized_upper,
    )


def plot_temporal_summary(
    ax: "Axes",
    *,
    summary: TemporalStatisticSummary,
    color: str,
    label: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "-",
    marker: str | None = None,
    markersize: float | None = None,
    band_alpha: float = 0.16,
    show_band: bool = True,
    limit_value: float | None = None,
    show_limit_label: bool = False,
    limit_label_x_axes: float = 0.82,
    limit_label_y_offset_points: float = 6,
) -> "Line2D":
    """Plot one summarized temporal series with an optional percentile band."""

    if show_limit_label and limit_value is None:
        raise ValueError("show_limit_label requires limit_value to be provided.")

    if show_band and summary.lower is not None and summary.upper is not None:
        from .visualization import plot_line_with_band

        line = plot_line_with_band(
            ax,
            x=summary.epoch,
            y=summary.mean,
            lower=summary.lower,
            upper=summary.upper,
            color=color,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            band_alpha=band_alpha,
        )
    else:
        plot_kwargs: dict[str, object] = {
            "color": color,
            "label": label,
            "linewidth": linewidth,
            "linestyle": linestyle,
        }
        if marker is not None:
            plot_kwargs["marker"] = marker
        if markersize is not None:
            plot_kwargs["markersize"] = markersize
        (line,) = ax.plot(summary.epoch, summary.mean, **plot_kwargs)

    if limit_value is not None:
        ax.axhline(
            limit_value,
            color=color,
            linewidth=0.5,
            alpha=0.85,
            zorder=1.1,
            label="_nolegend_",
        )
        if show_limit_label:
            from matplotlib.transforms import blended_transform_factory

            text_transform = blended_transform_factory(ax.transAxes, ax.transData)
            ax.annotate(
                f"{limit_value:.4f}",
                xy=(limit_label_x_axes, limit_value),
                xycoords=text_transform,
                xytext=(0, limit_label_y_offset_points),
                textcoords="offset points",
                color="black",
                ha="left",
                va="center",
            )

    return line


def _epoch_series(history: TrainingHistory, *, split: Literal["train", "test"]) -> list[int]:
    return list(history.train_epoch if split == "train" else history.test_epoch)


def _accuracy_series(history: TrainingHistory, *, split: Literal["train", "test"]) -> list[float]:
    metrics = history.train_metrics if split == "train" else history.test_metrics
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        raise ValueError(f"History is missing accuracy in {split}_metrics.")
    epoch = _epoch_series(history, split=split)
    if len(accuracy) != len(epoch):
        raise ValueError(
            f"History {split}_metrics['accuracy'] length {len(accuracy)} does not match "
            f"epoch length {len(epoch)}."
        )
    return accuracy


def summarize_accuracy_histories(
    histories: Sequence[TrainingHistory],
    *,
    split: Literal["train", "test"],
) -> MetricHistorySummary:
    """Aggregate per-epoch accuracy traces across multiple runs."""

    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")

    if not histories:
        raise ValueError("histories must contain at least one TrainingHistory.")

    reference_epoch = _epoch_series(histories[0], split=split)
    if not reference_epoch:
        raise ValueError("Training histories must contain at least one epoch.")

    accuracy_series: list[list[float]] = []
    for history in histories:
        if _epoch_series(history, split=split) != reference_epoch:
            raise ValueError("All histories must share the same epoch grid.")
        accuracy_series.append(_accuracy_series(history, split=split))

    mean: list[float] = []
    std: list[float] = []
    for epoch_index in range(len(reference_epoch)):
        epoch_values = [series[epoch_index] for series in accuracy_series]
        epoch_mean = sum(epoch_values) / len(epoch_values)
        epoch_variance = sum((value - epoch_mean) ** 2 for value in epoch_values) / len(epoch_values)
        mean.append(epoch_mean)
        std.append(math.sqrt(epoch_variance))

    return MetricHistorySummary(
        epoch=reference_epoch,
        mean=mean,
        std=std,
    )


def focus_metric_history(
    summary: MetricHistorySummary,
    *,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
) -> MetricHistorySummary:
    """Return a focused epoch window from an aggregated metric history."""

    if epoch_start is not None and epoch_end is not None and epoch_start > epoch_end:
        raise ValueError(
            f"epoch_start must be <= epoch_end, got {epoch_start} > {epoch_end}."
        )

    focused_points = [
        (epoch, mean, std)
        for epoch, mean, std in zip(summary.epoch, summary.mean, summary.std, strict=True)
        if (epoch_start is None or epoch >= epoch_start) and (epoch_end is None or epoch <= epoch_end)
    ]
    if not focused_points:
        raise ValueError("The requested epoch window does not contain any history points.")

    epoch, mean, std = zip(*focused_points, strict=True)
    return MetricHistorySummary(
        epoch=list(epoch),
        mean=list(mean),
        std=list(std),
    )
