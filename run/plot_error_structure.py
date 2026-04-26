from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qcnn import ARTICLE_PANEL_FIGSIZE, ErrorAnalysisPayload, configure_matplotlib_pdf_fonts
from qcnn.article_figures import load_saved_run_seeds, resolve_saved_run_seed
from qcnn.visualization import (
    DEFAULT_ERROR_ANALYSIS_GRID_SHAPE,
    DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES,
    _plot_error_analysis_confusion_panel,
    _plot_error_analysis_gallery_panel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs"
DEFAULT_SWEEP_DIRECTORY_NAME = "pcsqcnn_image_size_sweep"
DEFAULT_SCALED_IMAGE_SIZE = 16
DEFAULT_IMAGE_SIZE = 16
DEFAULT_CONFUSION_OUTPUT_FILENAME = "error_confusion_matrix.pdf"
DEFAULT_GALLERY_OUTPUT_FILENAME = "error_gallery.pdf"
DEFAULT_MAX_EXAMPLES = DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES
DEFAULT_GALLERY_GRID_SHAPE = DEFAULT_ERROR_ANALYSIS_GRID_SHAPE
DEFAULT_CONFUSION_CMAP = "Blues"
DEFAULT_PREDICTED_LABEL_COLOR = "midnightblue"
DEFAULT_TRUE_LABEL_COLOR = "tab:blue"


@dataclass(frozen=True)
class SavedErrorStructurePayload:
    seed: int
    image_size: int
    payload: ErrorAnalysisPayload


def load_manifest_seeds(run_directory: str | Path) -> list[int]:
    return load_saved_run_seeds(run_directory)


def resolve_error_structure_seed(
    run_directory: str | Path,
    *,
    seed: int | None,
) -> int:
    return resolve_saved_run_seed(run_directory, seed=seed)


def build_figure_5b_run_directory(
    artifacts_root: str | Path,
    *,
    scaled_image_size: int | None = None,
    image_size: int,
) -> Path:
    resolved_scaled_image_size = image_size if scaled_image_size is None else scaled_image_size
    if resolved_scaled_image_size <= 0:
        raise ValueError(f"scaled_image_size must be positive, got {resolved_scaled_image_size}.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    return (
        Path(artifacts_root)
        / DEFAULT_SWEEP_DIRECTORY_NAME
        / f"{resolved_scaled_image_size}on{image_size}"
    )


def build_error_structure_output_path(
    run_directory: str | Path,
    *,
    seed: int,
) -> Path:
    return Path(run_directory) / f"error_structure_seed{seed}.pt"


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_error_structure.py requires matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to render Figure S4."
        ) from exc
    return plt


def load_error_structure_payload(payload_path: str | Path) -> SavedErrorStructurePayload:
    payload = torch.load(Path(payload_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Figure S4 payload must deserialize to a mapping.")

    seed = payload.get("seed")
    image_size = payload.get("image_size")
    class_labels = payload.get("class_labels")
    confusion_matrix = payload.get("confusion_matrix")
    misclassified_images = payload.get("misclassified_images")
    misclassified_true_labels = payload.get("misclassified_true_labels")
    misclassified_predicted_labels = payload.get("misclassified_predicted_labels")

    if not isinstance(seed, int):
        raise ValueError("Figure S4 payload is missing integer 'seed'.")
    if not isinstance(image_size, int):
        raise ValueError("Figure S4 payload is missing integer 'image_size'.")
    if not isinstance(class_labels, list) or not all(isinstance(label, str) for label in class_labels):
        raise ValueError("Figure S4 payload must contain a 'class_labels' list of strings.")
    if not isinstance(confusion_matrix, torch.Tensor) or confusion_matrix.ndim != 2:
        raise ValueError("Figure S4 payload must contain a 2D tensor 'confusion_matrix'.")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("Figure S4 payload 'confusion_matrix' must be square.")
    if confusion_matrix.shape[0] != len(class_labels):
        raise ValueError("Figure S4 payload class_labels must match the confusion-matrix size.")
    if not isinstance(misclassified_images, torch.Tensor) or misclassified_images.ndim != 3:
        raise ValueError("Figure S4 payload must contain a 3D tensor 'misclassified_images'.")
    if not isinstance(misclassified_true_labels, torch.Tensor) or misclassified_true_labels.ndim != 1:
        raise ValueError("Figure S4 payload must contain a 1D tensor 'misclassified_true_labels'.")
    if not isinstance(misclassified_predicted_labels, torch.Tensor) or misclassified_predicted_labels.ndim != 1:
        raise ValueError("Figure S4 payload must contain a 1D tensor 'misclassified_predicted_labels'.")
    if misclassified_images.shape[0] != misclassified_true_labels.numel():
        raise ValueError("Figure S4 payload misclassified_images batch size must match the saved true labels.")
    if misclassified_true_labels.shape != misclassified_predicted_labels.shape:
        raise ValueError("Figure S4 payload misclassified label tensors must share the same shape.")

    return SavedErrorStructurePayload(
        seed=seed,
        image_size=image_size,
        payload=ErrorAnalysisPayload(
            class_labels=tuple(class_labels),
            confusion_matrix=confusion_matrix.clone(),
            misclassified_images=misclassified_images.clone(),
            misclassified_true_labels=misclassified_true_labels.clone(),
            misclassified_predicted_labels=misclassified_predicted_labels.clone(),
        ),
    )


def resolve_visible_example_count(
    payload: SavedErrorStructurePayload,
    *,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> int:
    if max_examples < 1:
        raise ValueError(f"max_examples must be positive, got {max_examples}.")
    return min(max_examples, payload.payload.num_examples)


def _plot_article_figure_s4a_from_saved_payload(
    saved_payload: SavedErrorStructurePayload,
    *,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()
    figure = plt.figure(figsize=figsize, layout="constrained")
    confusion_ax = figure.add_subplot(1, 1, 1)
    _plot_error_analysis_confusion_panel(
        confusion_ax,
        saved_payload.payload,
        confusion_cmap=DEFAULT_CONFUSION_CMAP,
    )
    return figure


def _plot_article_figure_s4b_from_saved_payload(
    saved_payload: SavedErrorStructurePayload,
    *,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    plt = _require_matplotlib()
    figure = plt.figure(figsize=figsize, layout="constrained")
    gallery_anchor_ax = figure.add_subplot(1, 1, 1)
    _plot_error_analysis_gallery_panel(
        gallery_anchor_ax,
        saved_payload.payload,
        max_examples=DEFAULT_MAX_EXAMPLES,
        example_grid_shape=DEFAULT_GALLERY_GRID_SHAPE,
        predicted_label_color=DEFAULT_PREDICTED_LABEL_COLOR,
        true_label_color=DEFAULT_TRUE_LABEL_COLOR,
    )
    return figure


def plot_article_figure_s4a(
    *,
    payload_path: str | Path,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    saved_payload = load_error_structure_payload(payload_path)
    return _plot_article_figure_s4a_from_saved_payload(saved_payload, figsize=figsize)


def plot_article_figure_s4b(
    *,
    payload_path: str | Path,
    figsize: tuple[float, float] = ARTICLE_PANEL_FIGSIZE,
) -> "Figure":
    saved_payload = load_error_structure_payload(payload_path)
    return _plot_article_figure_s4b_from_saved_payload(saved_payload, figsize=figsize)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure S4 panels from a saved error-structure payload.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_image_size_sweep run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where error_confusion_matrix.pdf and error_gallery.pdf will be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Figure 5b canvas size selecting the source run directory.",
    )
    parser.add_argument(
        "--scaled-image-size",
        type=int,
        default=DEFAULT_SCALED_IMAGE_SIZE,
        help="Figure 5b resized digit size selecting the source run directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional saved seed to plot. Defaults to the first seed recorded in manifest.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plt = _require_matplotlib()

    run_directory = build_figure_5b_run_directory(
        args.artifacts_root.expanduser().resolve(),
        scaled_image_size=args.scaled_image_size,
        image_size=args.image_size,
    )
    seed = resolve_error_structure_seed(run_directory, seed=args.seed)
    payload_path = build_error_structure_output_path(run_directory, seed=seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_payload = load_error_structure_payload(payload_path)

    confusion_figure = _plot_article_figure_s4a_from_saved_payload(saved_payload)
    confusion_output_path = output_dir / DEFAULT_CONFUSION_OUTPUT_FILENAME
    confusion_figure.savefig(confusion_output_path)
    plt.close(confusion_figure)

    gallery_figure = _plot_article_figure_s4b_from_saved_payload(saved_payload)
    gallery_output_path = output_dir / DEFAULT_GALLERY_OUTPUT_FILENAME
    gallery_figure.savefig(gallery_output_path)
    plt.close(gallery_figure)

    print(confusion_output_path)
    print(gallery_output_path)


if __name__ == "__main__":
    main()
