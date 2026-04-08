from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypedDict, Sequence

from qcnn import (
    ARTICLE_PANEL_FIGSIZE,
    load_saved_parameter_stats_line,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_temporal_samples,
)
from qcnn.article_figures import figure_2_fixed_run_directory_name

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qcnn import TemporalStatisticSummary, TrainingHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_FULL_MNIST_CLASSICAL_ARTIFACTS_ROOT = (
    DEFAULT_ARTIFACTS_ROOT / "full_mnist_classical_baselines"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_SAMPLES_PER_CLASS = 1000
DEFAULT_CLASSICAL_IMAGE_SIZE = 32
DEFAULT_TRANSLATED_CLASSICAL_SCALED_IMAGE_SIZE = 16
DEFAULT_FULL_MNIST_CLASSICAL_SCALED_IMAGE_SIZE = 32
DEFAULT_QUANTUM_ARCHITECTURE_SWEEP_DIRECTORY_NAME = "pcsqcnn_architecture_sweep_full_readout"
DEFAULT_QUANTUM_FEATURE_QUBITS = 2
DEFAULT_QUANTUM_LAYERS = 3
DEFAULT_CLASSICAL_EPOCH_GROUP_SIZE = 50
DEFAULT_QUANTUM_EPOCH_GROUP_SIZE = 50
DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS = 7
DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS = 4
DEFAULT_CLASSICAL_LEGEND_LOCATION = "center right"
DEFAULT_QUANTUM_LEGEND_LOCATION = "lower right"


class _Figure2SeriesConfig(TypedDict):
    histories: Sequence["TrainingHistory"]
    test_label: str
    train_label: str
    color: object
    train_linestyle: str
    limit_label_y_offset_points: int


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_translated_mnist_baselines.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure 2(a,b)."
        ) from exc
    return plt


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _load_training_histories_or_warn(
    run_directory: Path,
    *,
    label: str,
) -> Sequence["TrainingHistory"] | None:
    try:
        return load_saved_training_histories(run_directory)
    except Exception as exc:
        _warn(
            f"Skipping {label}: could not load training histories from "
            f"{run_directory}: {exc}"
        )
        return None


def _print_parameter_stats_or_warn(*, label: str, run_directory: Path) -> None:
    try:
        parameter_stats_line = load_saved_parameter_stats_line(run_directory)
    except Exception as exc:
        _warn(
            f"Could not load parameter stats for {label} from "
            f"{run_directory}: {exc}"
        )
        return
    print(f"{label}: {parameter_stats_line}")


def _expand_y_limits_for_reference_levels(ax, *, levels: Sequence[float]) -> None:
    current_y_min, current_y_max = ax.get_ylim()
    current_span = current_y_max - current_y_min
    if current_span <= 0:
        current_span = 1.0

    reference_min = min(levels)
    reference_max = max(levels)
    new_y_min = min(current_y_min, reference_min - 0.02 * current_span)
    new_y_max = max(current_y_max, reference_max + 0.05 * current_span)
    ax.set_ylim(new_y_min, new_y_max)


def _metric_epoch_and_samples(
    histories: Sequence["TrainingHistory"],
    *,
    split: str,
) -> tuple[list[int], list[list[float]]]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    if not histories:
        raise ValueError("histories must contain at least one TrainingHistory.")

    reference_history = histories[0]
    reference_epoch = (
        list(reference_history.train_epoch) if split == "train" else list(reference_history.test_epoch)
    )
    if not reference_epoch:
        raise ValueError("Training histories must contain at least one epoch.")

    samples_by_epoch: list[list[float]] = [[] for _ in reference_epoch]
    for history in histories:
        epoch = list(history.train_epoch) if split == "train" else list(history.test_epoch)
        if epoch != reference_epoch:
            raise ValueError("All histories must share the same epoch grid.")
        metrics = history.train_metrics if split == "train" else history.test_metrics
        accuracy = metrics.get("accuracy")
        if accuracy is None:
            raise ValueError(f"History is missing accuracy in {split}_metrics.")
        if len(accuracy) != len(epoch):
            raise ValueError(
                f"History {split}_metrics['accuracy'] length {len(accuracy)} does not match "
                f"epoch length {len(epoch)}."
            )
        for epoch_index, value in enumerate(accuracy):
            samples_by_epoch[epoch_index].append(float(value))

    return reference_epoch, samples_by_epoch


def _summarize_accuracy_samples(
    histories: Sequence["TrainingHistory"],
    *,
    split: str,
    epoch_start: int | None,
    epoch_end: int | None,
    epoch_group_size: int,
    lower_percentile: float,
    upper_percentile: float,
    compute_band: bool = True,
) -> "TemporalStatisticSummary":
    epoch, samples_by_epoch = _metric_epoch_and_samples(histories, split=split)
    return summarize_temporal_samples(
        epoch,
        samples_by_epoch,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        compute_band=compute_band,
    )


def _plot_figure_2_panel(
    *,
    series_configs: Sequence[_Figure2SeriesConfig],
    epoch_start: int | None,
    epoch_end: int | None,
    epoch_group_size: int,
    lower_percentile: float,
    upper_percentile: float,
    figsize: tuple[float, float],
    legend_location: str = DEFAULT_QUANTUM_LEGEND_LOCATION,
) -> Figure:
    plt = _require_matplotlib()

    if not series_configs:
        raise ValueError("Figure 2 panel requires at least one series config.")

    summarized_series: list[dict[str, object]] = []
    for config in series_configs:
        train_summary_full = _summarize_accuracy_samples(
            config["histories"],
            split="train",
            epoch_start=None,
            epoch_end=None,
            epoch_group_size=1,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            compute_band=False,
        )
        test_summary_full = _summarize_accuracy_samples(
            config["histories"],
            split="test",
            epoch_start=None,
            epoch_end=None,
            epoch_group_size=1,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            compute_band=False,
        )
        summarized_series.append(
            {
                "config": config,
                "train_summary_full": train_summary_full,
                "test_summary_full": test_summary_full,
                "train_summary_focused": _summarize_accuracy_samples(
                    config["histories"],
                    split="train",
                    epoch_start=epoch_start,
                    epoch_end=epoch_end,
                    epoch_group_size=epoch_group_size,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                ),
                "test_summary_focused": _summarize_accuracy_samples(
                    config["histories"],
                    split="test",
                    epoch_start=epoch_start,
                    epoch_end=epoch_end,
                    epoch_group_size=epoch_group_size,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                ),
            }
        )

    figure, ax = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_epoch_values: list[float] = []
    for series in summarized_series:
        config = series["config"]
        final_value = series["test_summary_full"].mean[-1]
        plot_temporal_summary(
            ax,
            summary=series["test_summary_focused"],
            color=config["color"],
            label=config["test_label"],
            band_alpha=0.30,
            limit_value=final_value,
            show_limit_label=True,
            limit_label_x_axes=0.82,
            limit_label_y_offset_points=config["limit_label_y_offset_points"],
        )
        plot_temporal_summary(
            ax,
            summary=series["train_summary_focused"],
            color=config["color"],
            label=config["train_label"],
            linestyle=config["train_linestyle"],
            band_alpha=0.12,
        )
        all_epoch_values.extend(series["test_summary_focused"].epoch)
        all_epoch_values.extend(series["train_summary_focused"].epoch)

    final_test_levels = [series["test_summary_full"].mean[-1] for series in summarized_series]

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(min(all_epoch_values), max(all_epoch_values))
    _expand_y_limits_for_reference_levels(ax, levels=final_test_levels)
    legend_columns = 1 if len(series_configs) <= 2 else 2
    ax.legend(loc=legend_location, framealpha=0.8, ncols=legend_columns)

    figure.tight_layout()
    return figure


def _resolve_fixed_run_directory(
    artifacts_root: Path,
    *,
    base_name: str,
    samples_per_class: int,
) -> Path:
    return artifacts_root / figure_2_fixed_run_directory_name(
        base_name,
        samples_per_class=samples_per_class,
    )


def _resolve_classical_run_directory(
    artifacts_root: Path,
    *,
    base_name: str,
    scaled_image_size: int,
    image_size: int = DEFAULT_CLASSICAL_IMAGE_SIZE,
) -> Path:
    return artifacts_root / base_name / f"{scaled_image_size}on{image_size}"


def _resolve_architecture_sweep_run_directory(
    artifacts_root: Path,
    *,
    sweep_directory_name: str = DEFAULT_QUANTUM_ARCHITECTURE_SWEEP_DIRECTORY_NAME,
    feature_qubits: int = DEFAULT_QUANTUM_FEATURE_QUBITS,
    quantum_layers: int = DEFAULT_QUANTUM_LAYERS,
) -> Path:
    return artifacts_root / sweep_directory_name / f"fq{feature_qubits}_ql{quantum_layers}"


def _load_available_classical_series_configs(
    *,
    artifacts_root: Path,
    scaled_image_size: int,
    image_size: int = DEFAULT_CLASSICAL_IMAGE_SIZE,
    limit_label_y_offset_points: int = -DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS,
) -> tuple[list[_Figure2SeriesConfig], list[tuple[str, Path]]]:
    series_configs: list[_Figure2SeriesConfig] = []
    run_directories: list[tuple[str, Path]] = []
    series_specifications = (
        ("classical_cnn", "CNN", "Train CNN", "C0", "-."),
        ("classical_mlp", "MLP", "Train MLP", "C1", "--"),
    )

    for base_name, test_label, train_label, color, train_linestyle in series_specifications:
        run_directory = _resolve_classical_run_directory(
            artifacts_root,
            base_name=base_name,
            scaled_image_size=scaled_image_size,
            image_size=image_size,
        )
        stats_label = f"{base_name} {scaled_image_size}on{image_size}"
        histories = _load_training_histories_or_warn(run_directory, label=stats_label)
        if histories is None:
            continue
        series_configs.append(
            _Figure2SeriesConfig(
                histories=histories,
                test_label=test_label,
                train_label=train_label,
                color=color,
                train_linestyle=train_linestyle,
                limit_label_y_offset_points=limit_label_y_offset_points,
            )
        )
        run_directories.append((stats_label, run_directory))

    return series_configs, run_directories


def _load_available_quantum_series_configs(
    *,
    pcsqcnn_run_directory: Path,
    pcsqcnn_no_qft_run_directory: Path,
) -> tuple[list[_Figure2SeriesConfig], list[tuple[str, Path]]]:
    series_configs: list[_Figure2SeriesConfig] = []
    run_directories: list[tuple[str, Path]] = []
    series_specifications = (
        (
            "pcsqcnn_q3_nf2",
            pcsqcnn_run_directory,
            "PCS-QCNN",
            "Train PCS-QCNN",
            "C0",
            "-.",
        ),
        (
            "pcsqcnn_no_qft",
            pcsqcnn_no_qft_run_directory,
            "PCS-QCNN (no QFT)",
            "Train PCS-QCNN (no QFT)",
            "C1",
            "--",
        ),
    )

    for stats_label, run_directory, test_label, train_label, color, train_linestyle in series_specifications:
        histories = _load_training_histories_or_warn(run_directory, label=stats_label)
        if histories is None:
            continue
        series_configs.append(
            _Figure2SeriesConfig(
                histories=histories,
                test_label=test_label,
                train_label=train_label,
                color=color,
                train_linestyle=train_linestyle,
                limit_label_y_offset_points=DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS,
            )
        )
        run_directories.append((stats_label, run_directory))

    return series_configs, run_directories


def _load_required_classical_pair(
    *,
    artifacts_root: Path,
    scaled_image_size: int,
    image_size: int = DEFAULT_CLASSICAL_IMAGE_SIZE,
) -> tuple[Sequence["TrainingHistory"], Sequence["TrainingHistory"], list[tuple[str, Path]]]:
    cnn_run_directory = _resolve_classical_run_directory(
        artifacts_root,
        base_name="classical_cnn",
        scaled_image_size=scaled_image_size,
        image_size=image_size,
    )
    mlp_run_directory = _resolve_classical_run_directory(
        artifacts_root,
        base_name="classical_mlp",
        scaled_image_size=scaled_image_size,
        image_size=image_size,
    )
    try:
        cnn_histories = load_saved_training_histories(cnn_run_directory)
    except Exception as exc:
        raise ValueError(
            f"Could not load required classical baseline classical_cnn {scaled_image_size}on{image_size}: {exc}"
        ) from exc
    try:
        mlp_histories = load_saved_training_histories(mlp_run_directory)
    except Exception as exc:
        raise ValueError(
            f"Could not load required classical baseline classical_mlp {scaled_image_size}on{image_size}: {exc}"
        ) from exc

    return (
        cnn_histories,
        mlp_histories,
        [
            (f"classical_cnn {scaled_image_size}on{image_size}", cnn_run_directory),
            (f"classical_mlp {scaled_image_size}on{image_size}", mlp_run_directory),
        ],
    )


def plot_article_figure_2a(
    *,
    cnn_histories=None,
    mlp_histories=None,
    classical_series_configs: Sequence[_Figure2SeriesConfig] | None = None,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = DEFAULT_CLASSICAL_EPOCH_GROUP_SIZE,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> Figure:
    """Render article Figure 2(a) from saved classical training histories."""

    if classical_series_configs is None:
        if cnn_histories is None or mlp_histories is None:
            raise ValueError(
                "plot_article_figure_2a requires cnn_histories and mlp_histories "
                "when classical_series_configs is not provided."
            )
        classical_series_configs = [
            _Figure2SeriesConfig(
                histories=cnn_histories,
                test_label="CNN",
                train_label="Train CNN",
                color="C0",
                train_linestyle="-.",
                limit_label_y_offset_points=-DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS,
            ),
            _Figure2SeriesConfig(
                histories=mlp_histories,
                test_label="MLP",
                train_label="Train MLP",
                color="C1",
                train_linestyle="--",
                limit_label_y_offset_points=-DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS,
            ),
        ]

    return _plot_figure_2_panel(
        series_configs=classical_series_configs,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        figsize=figsize,
        legend_location=DEFAULT_CLASSICAL_LEGEND_LOCATION,
    )


def plot_article_figure_2b(
    *,
    pcsqcnn_histories=None,
    pcsqcnn_no_qft_histories=None,
    quantum_series_configs: Sequence[_Figure2SeriesConfig] | None = None,
    epoch_start: int | None = None,
    epoch_end: int | None = None,
    epoch_group_size: int = DEFAULT_QUANTUM_EPOCH_GROUP_SIZE,
    lower_percentile: float = 25.0,
    upper_percentile: float = 75.0,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> Figure:
    """Render article Figure 2(b) from saved PCS-QCNN training histories."""

    if quantum_series_configs is None:
        if pcsqcnn_histories is None or pcsqcnn_no_qft_histories is None:
            raise ValueError(
                "plot_article_figure_2b requires pcsqcnn_histories and "
                "pcsqcnn_no_qft_histories when quantum_series_configs is not provided."
            )
        quantum_series_configs = [
            _Figure2SeriesConfig(
                histories=pcsqcnn_histories,
                test_label="PCS-QCNN",
                train_label="Train PCS-QCNN",
                color="C0",
                train_linestyle="-.",
                limit_label_y_offset_points=DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS,
            ),
            _Figure2SeriesConfig(
                histories=pcsqcnn_no_qft_histories,
                test_label="PCS-QCNN (no QFT)",
                train_label="Train PCS-QCNN (no QFT)",
                color="C1",
                train_linestyle="--",
                limit_label_y_offset_points=DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS,
            ),
        ]

    return _plot_figure_2_panel(
        series_configs=quantum_series_configs,
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        epoch_group_size=epoch_group_size,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        figsize=figsize,
        legend_location=DEFAULT_QUANTUM_LEGEND_LOCATION,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render article Figure 2(a,b) from saved benchmark and sweep artifacts.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help=(
            "Directory containing translated Figure 2 artifacts and the architecture-sweep "
            "directory, for example classical_cnn/16on32, classical_mlp/16on32, "
            f"{DEFAULT_QUANTUM_ARCHITECTURE_SWEEP_DIRECTORY_NAME}/fq{DEFAULT_QUANTUM_FEATURE_QUBITS}_ql{DEFAULT_QUANTUM_LAYERS}, and "
            "pcsqcnn_no_qft_spc20."
        ),
    )
    parser.add_argument(
        "--full-mnist-classical-artifacts-root",
        type=Path,
        default=DEFAULT_FULL_MNIST_CLASSICAL_ARTIFACTS_ROOT,
        help=(
            "Directory containing the supplementary full-MNIST classical baselines, "
            "for example artifacts/full_mnist_classical_baselines."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where translated_mnist_classical_baselines.pdf and "
            "translated_mnist_quantum_ablation.pdf will be written, along with "
            "full_mnist_classical_baselines.pdf."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=(
            "Sample-count suffix used to resolve the fixed no-QFT Figure 2 run directory, "
            "for example pcsqcnn_no_qft_spc20."
        ),
    )
    parser.add_argument(
        "--translated-scaled-image-size",
        type=int,
        default=DEFAULT_TRANSLATED_CLASSICAL_SCALED_IMAGE_SIZE,
        help="Scaled-image size expected for the translated-MNIST classical panel.",
    )
    parser.add_argument(
        "--full-mnist-scaled-image-size",
        type=int,
        default=DEFAULT_FULL_MNIST_CLASSICAL_SCALED_IMAGE_SIZE,
        help="Scaled-image size expected for the supplementary full-MNIST classical panel.",
    )
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=None,
        help=(
            "Optional inclusive starting epoch for both classical history windows "
            "(translated main-text panel and supplementary full-MNIST control)."
        ),
    )
    parser.add_argument(
        "--epoch-end",
        type=int,
        default=None,
        help=(
            "Inclusive ending epoch for both classical history windows "
            "(translated main-text panel and supplementary full-MNIST control)."
        ),
    )
    parser.add_argument(
        "--epoch-group-size",
        type=int,
        default=None,
        help=(
            "Optional epoch-width grouping override shared by both panels. "
            f"By default Figure 2(a) uses {DEFAULT_CLASSICAL_EPOCH_GROUP_SIZE} and "
            f"Figure 2(b) uses {DEFAULT_QUANTUM_EPOCH_GROUP_SIZE}."
        ),
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
    parser.add_argument(
        "--quantum-epoch-start",
        type=int,
        default=None,
        help="Optional inclusive starting epoch for the quantum Figure 2(b) history window.",
    )
    parser.add_argument(
        "--quantum-epoch-end",
        type=int,
        default=None,
        help="Inclusive ending epoch for the quantum Figure 2(b) history window.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()

    artifacts_root = args.artifacts_root.expanduser().resolve()
    full_mnist_classical_artifacts_root = (
        args.full_mnist_classical_artifacts_root.expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pcsqcnn_run_directory = _resolve_architecture_sweep_run_directory(
        artifacts_root,
    )
    pcsqcnn_no_qft_run_directory = _resolve_fixed_run_directory(
        artifacts_root,
        base_name="pcsqcnn_no_qft",
        samples_per_class=args.samples_per_class,
    )

    translated_classical_series_configs, translated_classical_run_directories = (
        _load_available_classical_series_configs(
            artifacts_root=artifacts_root,
            scaled_image_size=args.translated_scaled_image_size,
        )
    )
    full_mnist_classical_series_configs, full_mnist_classical_run_directories = (
        _load_available_classical_series_configs(
            artifacts_root=full_mnist_classical_artifacts_root,
            scaled_image_size=args.full_mnist_scaled_image_size,
            limit_label_y_offset_points=DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS,
        )
    )
    quantum_series_configs, quantum_run_directories = _load_available_quantum_series_configs(
        pcsqcnn_run_directory=pcsqcnn_run_directory,
        pcsqcnn_no_qft_run_directory=pcsqcnn_no_qft_run_directory,
    )
    classical_epoch_group_size = (
        args.epoch_group_size
        if args.epoch_group_size is not None
        else DEFAULT_CLASSICAL_EPOCH_GROUP_SIZE
    )
    quantum_epoch_group_size = (
        args.epoch_group_size
        if args.epoch_group_size is not None
        else DEFAULT_QUANTUM_EPOCH_GROUP_SIZE
    )

    translated_classical_output_path = output_dir / "translated_mnist_classical_baselines.pdf"
    if translated_classical_series_configs:
        translated_classical_figure = None
        try:
            translated_classical_figure = plot_article_figure_2a(
                classical_series_configs=translated_classical_series_configs,
                epoch_start=args.epoch_start,
                epoch_end=args.epoch_end,
                epoch_group_size=classical_epoch_group_size,
                lower_percentile=args.lower_percentile,
                upper_percentile=args.upper_percentile,
            )
            translated_classical_figure.savefig(translated_classical_output_path)
        except Exception as exc:
            _warn(
                "Skipping translated_mnist_classical_baselines.pdf: could not render "
                f"the translated classical panel: {exc}"
            )
        finally:
            if translated_classical_figure is not None:
                plt.close(translated_classical_figure)
    else:
        _warn(
            "Skipping translated_mnist_classical_baselines.pdf: no translated "
            "classical baseline histories were available."
        )

    full_mnist_classical_output_path = output_dir / "full_mnist_classical_baselines.pdf"
    if full_mnist_classical_series_configs:
        full_mnist_classical_figure = None
        try:
            full_mnist_classical_figure = plot_article_figure_2a(
                classical_series_configs=full_mnist_classical_series_configs,
                epoch_start=args.epoch_start,
                epoch_end=args.epoch_end,
                epoch_group_size=classical_epoch_group_size,
                lower_percentile=args.lower_percentile,
                upper_percentile=args.upper_percentile,
            )
            full_mnist_classical_figure.savefig(full_mnist_classical_output_path)
        except Exception as exc:
            _warn(
                "Skipping full_mnist_classical_baselines.pdf: could not render "
                f"the full-MNIST classical panel: {exc}"
            )
        finally:
            if full_mnist_classical_figure is not None:
                plt.close(full_mnist_classical_figure)
    else:
        _warn(
            "Skipping full_mnist_classical_baselines.pdf: no full-MNIST classical "
            "baseline histories were available."
        )

    quantum_output_path = output_dir / "translated_mnist_quantum_ablation.pdf"
    if quantum_series_configs:
        quantum_figure = None
        try:
            quantum_figure = plot_article_figure_2b(
                quantum_series_configs=quantum_series_configs,
                epoch_start=args.quantum_epoch_start,
                epoch_end=args.quantum_epoch_end,
                epoch_group_size=quantum_epoch_group_size,
                lower_percentile=args.lower_percentile,
                upper_percentile=args.upper_percentile,
            )
            quantum_figure.savefig(quantum_output_path)
        except Exception as exc:
            _warn(
                "Skipping translated_mnist_quantum_ablation.pdf: could not render "
                f"the quantum panel: {exc}"
            )
        finally:
            if quantum_figure is not None:
                plt.close(quantum_figure)
    else:
        _warn(
            "Skipping translated_mnist_quantum_ablation.pdf: no quantum baseline "
            "histories were available."
        )

    for label, run_directory in translated_classical_run_directories:
        _print_parameter_stats_or_warn(label=label, run_directory=run_directory)
    for label, run_directory in full_mnist_classical_run_directories:
        _print_parameter_stats_or_warn(label=f"full_mnist {label}", run_directory=run_directory)
    for label, run_directory in quantum_run_directories:
        _print_parameter_stats_or_warn(label=label, run_directory=run_directory)

    if translated_classical_output_path.exists():
        print(translated_classical_output_path)
    if full_mnist_classical_output_path.exists():
        print(full_mnist_classical_output_path)
    if quantum_output_path.exists():
        print(quantum_output_path)


if __name__ == "__main__":
    main()
