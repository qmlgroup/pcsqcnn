"""Notebook-facing plotting helpers for training curves and error analysis.

The module keeps visualization logic out of ``train_qcnn.py`` while preserving
the article-oriented presentation used in the paper figures. Importantly,
``matplotlib`` remains an optional dependency for the package: this module does
not import it at module load time and raises a targeted ``ImportError`` only
when one of the plotting helpers is called without the plotting extra
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch

from qcnn.matplotlib_config import configure_matplotlib_pdf_fonts
from qcnn.model import ImageClassifierRunner, TrainingHistory

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES = 16
DEFAULT_ERROR_ANALYSIS_GRID_SHAPE = (4, 4)
DEFAULT_ERROR_ANALYSIS_GALLERY_SPACING = 0.05
DEFAULT_ERROR_ANALYSIS_LABEL_FONTSIZE = 10


@dataclass(frozen=True)
class MisclassifiedExample:
    """One misclassified sample collected during error analysis."""

    image: torch.Tensor
    true_label: int
    predicted_label: int


@dataclass(frozen=True)
class ErrorAnalysisPayload:
    """Confusion matrix plus saved misclassified examples for one evaluation."""

    class_labels: tuple[str, ...]
    confusion_matrix: torch.Tensor
    misclassified_images: torch.Tensor
    misclassified_true_labels: torch.Tensor
    misclassified_predicted_labels: torch.Tensor

    @property
    def num_examples(self) -> int:
        return int(self.misclassified_true_labels.numel())

    def misclassified_examples(self) -> tuple[MisclassifiedExample, ...]:
        return tuple(
            MisclassifiedExample(
                image=self.misclassified_images[index],
                true_label=int(self.misclassified_true_labels[index].item()),
                predicted_label=int(self.misclassified_predicted_labels[index].item()),
            )
            for index in range(self.num_examples)
        )


@dataclass(frozen=True)
class _ErrorAnalysisRect:
    x0: float
    y0: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.x0 + self.width

    @property
    def y1(self) -> float:
        return self.y0 + self.height

    @property
    def center_x(self) -> float:
        return self.x0 + 0.5 * self.width

    @property
    def center_y(self) -> float:
        return self.y0 + 0.5 * self.height


@dataclass(frozen=True)
class _ErrorAnalysisGalleryGeometry:
    rows: int
    cols: int
    anchor_rect: _ErrorAnalysisRect
    container_rect: _ErrorAnalysisRect
    tile_rects: tuple[_ErrorAnalysisRect, ...]
    tile_size: float
    gap_size: float


@dataclass(frozen=True)
class _ErrorAnalysisLabelStyle:
    x: float
    y: float
    horizontal_alignment: str
    vertical_alignment: str
    fontsize: int
    bbox: dict[str, object] | None = None


def _require_matplotlib():
    try:
        import matplotlib

        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting helpers require matplotlib. Install qcnn with the "
            "'notebook' or 'test' extra to use plot_convergence() and "
            "plot_error_analysis()."
        ) from exc
    return plt


def plot_line_with_band(
    ax: "Axes",
    *,
    x: Sequence[float],
    y: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    color: str,
    label: str | None = None,
    linewidth: float = 1.5,
    linestyle: str = "-",
    marker: str | None = None,
    markersize: float | None = None,
    band_alpha: float = 0.16,
    band_linewidth: float = 0.0,
) -> "Line2D":
    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=band_alpha,
        linewidth=band_linewidth,
    )
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
    (line,) = ax.plot(x, y, **plot_kwargs)
    return line


def _validate_history(history: TrainingHistory, metric_name: str) -> None:
    if metric_name not in history.train_metrics:
        raise ValueError(f"metric_name {metric_name!r} is missing from history.train_metrics.")
    if metric_name not in history.test_metrics:
        raise ValueError(f"metric_name {metric_name!r} is missing from history.test_metrics.")

    expected_lengths = {
        "train_loss": (history.train_loss, len(history.train_epoch)),
        "test_loss": (history.test_loss, len(history.test_epoch)),
        f"train_metrics[{metric_name!r}]": (history.train_metrics[metric_name], len(history.train_epoch)),
        f"test_metrics[{metric_name!r}]": (history.test_metrics[metric_name], len(history.test_epoch)),
    }

    for series_name, (series, expected_length) in expected_lengths.items():
        if len(series) != expected_length:
            raise ValueError(f"history.{series_name} must have length {expected_length}, got {len(series)}.")


def _resolve_class_labels(
    *,
    class_labels: Sequence[str] | None,
    min_num_classes: int,
) -> tuple[str, ...]:
    if class_labels is None:
        return tuple(str(class_id) for class_id in range(min_num_classes))

    normalized_labels = tuple(str(label) for label in class_labels)
    if len(normalized_labels) < min_num_classes:
        raise ValueError(
            "class_labels must cover every class present in the data: "
            f"expected at least {min_num_classes}, got {len(normalized_labels)}."
        )
    return normalized_labels


def _validate_max_examples(max_examples: int | None) -> None:
    if max_examples is not None and max_examples < 1:
        raise ValueError(f"max_examples must be positive when provided, got {max_examples}.")


def _validate_example_grid_shape(
    example_grid_shape: tuple[int, int],
    *,
    max_examples: int,
) -> tuple[int, int]:
    rows, cols = example_grid_shape
    if rows < 1 or cols < 1:
        raise ValueError(
            "example_grid_shape must contain positive row/column counts, "
            f"got {example_grid_shape}."
        )
    if rows * cols < max_examples:
        raise ValueError(
            "example_grid_shape must have room for max_examples: "
            f"{rows} * {cols} < {max_examples}."
        )
    return rows, cols


def _resolve_error_analysis_gallery_container(
    anchor_rect: _ErrorAnalysisRect,
    *,
    rows: int,
    cols: int,
    spacing_ratio: float,
) -> _ErrorAnalysisRect:
    horizontal_units = cols + max(cols - 1, 0) * spacing_ratio
    vertical_units = rows + max(rows - 1, 0) * spacing_ratio
    width_limited_height = anchor_rect.width * vertical_units / horizontal_units

    if width_limited_height <= anchor_rect.height:
        return _ErrorAnalysisRect(
            x0=anchor_rect.x0,
            y0=anchor_rect.y0 + 0.5 * (anchor_rect.height - width_limited_height),
            width=anchor_rect.width,
            height=width_limited_height,
        )

    height_limited_width = anchor_rect.height * horizontal_units / vertical_units
    return _ErrorAnalysisRect(
        x0=anchor_rect.x0 + 0.5 * (anchor_rect.width - height_limited_width),
        y0=anchor_rect.y0,
        width=height_limited_width,
        height=anchor_rect.height,
    )


def _resolve_error_analysis_gallery_geometry(
    anchor_rect: _ErrorAnalysisRect,
    example_grid_shape: tuple[int, int] = DEFAULT_ERROR_ANALYSIS_GRID_SHAPE,
    *,
    max_examples: int = DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES,
    spacing_ratio: float = DEFAULT_ERROR_ANALYSIS_GALLERY_SPACING,
) -> _ErrorAnalysisGalleryGeometry:
    if spacing_ratio < 0.0:
        raise ValueError(f"spacing_ratio must be non-negative, got {spacing_ratio}.")

    rows, cols = _validate_example_grid_shape(example_grid_shape, max_examples=max_examples)
    container_rect = _resolve_error_analysis_gallery_container(
        anchor_rect,
        rows=rows,
        cols=cols,
        spacing_ratio=spacing_ratio,
    )

    horizontal_tile_size = container_rect.width / (
        cols + max(cols - 1, 0) * spacing_ratio
    )
    vertical_tile_size = container_rect.height / (
        rows + max(rows - 1, 0) * spacing_ratio
    )
    tile_size = min(horizontal_tile_size, vertical_tile_size)
    gap_size = spacing_ratio * tile_size

    grid_width = cols * tile_size + max(cols - 1, 0) * gap_size
    grid_height = rows * tile_size + max(rows - 1, 0) * gap_size
    grid_x0 = container_rect.x0 + 0.5 * (container_rect.width - grid_width)
    grid_y0 = container_rect.y0 + 0.5 * (container_rect.height - grid_height)

    tile_rects: list[_ErrorAnalysisRect] = []
    for row_index in range(rows):
        for column_index in range(cols):
            tile_rects.append(
                _ErrorAnalysisRect(
                    x0=grid_x0 + column_index * (tile_size + gap_size),
                    y0=grid_y0 + (rows - 1 - row_index) * (tile_size + gap_size),
                    width=tile_size,
                    height=tile_size,
                )
            )

    return _ErrorAnalysisGalleryGeometry(
        rows=rows,
        cols=cols,
        anchor_rect=anchor_rect,
        container_rect=container_rect,
        tile_rects=tuple(tile_rects),
        tile_size=tile_size,
        gap_size=gap_size,
    )


def _predicted_error_analysis_label_style() -> _ErrorAnalysisLabelStyle:
    return _ErrorAnalysisLabelStyle(
        x=0.06,
        y=0.08,
        horizontal_alignment="left",
        vertical_alignment="bottom",
        fontsize=DEFAULT_ERROR_ANALYSIS_LABEL_FONTSIZE,
        bbox=None,
    )


def _true_error_analysis_label_style() -> _ErrorAnalysisLabelStyle:
    return _ErrorAnalysisLabelStyle(
        x=0.94,
        y=0.08,
        horizontal_alignment="right",
        vertical_alignment="bottom",
        fontsize=DEFAULT_ERROR_ANALYSIS_LABEL_FONTSIZE,
        bbox=None,
    )


def _show_error_analysis_colorbar() -> bool:
    return False


def collect_error_analysis_payload(
    runner: ImageClassifierRunner,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    class_labels: Sequence[str] | None = None,
    max_examples: int | None = None,
) -> ErrorAnalysisPayload:
    """Collect confusion-matrix inputs and misclassified examples from a loader."""

    _validate_max_examples(max_examples)

    all_true_labels: list[torch.Tensor] = []
    all_predicted_labels: list[torch.Tensor] = []
    misclassified_images: list[torch.Tensor] = []
    misclassified_true_labels: list[int] = []
    misclassified_predicted_labels: list[int] = []
    observed_image_shape: tuple[int, ...] | None = None
    observed_image_dtype: torch.dtype | None = None

    was_training = runner.model.training
    try:
        runner.model.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                logits = runner.run_forward_pass(images)
                predicted_labels = logits.argmax(dim=1).detach().cpu().to(dtype=torch.long)
                true_labels = labels.detach().cpu().to(dtype=torch.long)

                all_true_labels.append(true_labels)
                all_predicted_labels.append(predicted_labels)

                cpu_images = images.detach().cpu()
                if observed_image_shape is None:
                    observed_image_shape = tuple(cpu_images.shape[1:])
                    observed_image_dtype = cpu_images.dtype

                if max_examples is not None and len(misclassified_images) >= max_examples:
                    continue

                mismatched = torch.nonzero(predicted_labels != true_labels, as_tuple=False).flatten()
                if mismatched.numel() == 0:
                    continue

                for sample_index in mismatched.tolist():
                    misclassified_images.append(cpu_images[sample_index].clone())
                    misclassified_true_labels.append(int(true_labels[sample_index].item()))
                    misclassified_predicted_labels.append(int(predicted_labels[sample_index].item()))
                    if max_examples is not None and len(misclassified_images) >= max_examples:
                        break
    finally:
        runner.model.train(was_training)

    if not all_true_labels:
        raise ValueError("data_loader must yield at least one sample.")
    if observed_image_shape is None or observed_image_dtype is None:
        raise ValueError("data_loader must yield at least one sample.")

    true_tensor = torch.cat(all_true_labels)
    predicted_tensor = torch.cat(all_predicted_labels)
    if true_tensor.numel() == 0:
        raise ValueError("data_loader must yield at least one sample.")

    max_label = int(torch.maximum(true_tensor.max(), predicted_tensor.max()).item())
    label_names = _resolve_class_labels(class_labels=class_labels, min_num_classes=max_label + 1)
    num_classes = len(label_names)

    flat_indices = true_tensor * num_classes + predicted_tensor
    confusion_matrix = torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    if misclassified_images:
        misclassified_image_tensor = torch.stack(misclassified_images)
    else:
        misclassified_image_tensor = torch.empty(
            (0, *observed_image_shape),
            dtype=observed_image_dtype,
        )

    return ErrorAnalysisPayload(
        class_labels=label_names,
        confusion_matrix=confusion_matrix,
        misclassified_images=misclassified_image_tensor,
        misclassified_true_labels=torch.tensor(misclassified_true_labels, dtype=torch.long),
        misclassified_predicted_labels=torch.tensor(misclassified_predicted_labels, dtype=torch.long),
    )


def _validate_error_analysis_payload(payload: ErrorAnalysisPayload) -> None:
    if payload.confusion_matrix.ndim != 2:
        raise ValueError("payload.confusion_matrix must be a 2D tensor.")
    num_classes, num_columns = payload.confusion_matrix.shape
    if num_classes != num_columns:
        raise ValueError("payload.confusion_matrix must be square.")
    if len(payload.class_labels) != num_classes:
        raise ValueError("payload.class_labels must match the confusion-matrix size.")
    if payload.misclassified_images.ndim < 1:
        raise ValueError("payload.misclassified_images must include a batch dimension.")
    if payload.misclassified_images.shape[0] != payload.num_examples:
        raise ValueError("payload.misclassified_images batch size must match the saved labels.")
    if payload.misclassified_predicted_labels.shape != payload.misclassified_true_labels.shape:
        raise ValueError("payload.misclassified label tensors must have identical shapes.")


def _plot_error_analysis_confusion_panel(
    confusion_ax: Axes,
    payload: ErrorAnalysisPayload,
    *,
    confusion_cmap: str = "gray_r",
) -> None:
    confusion_image = confusion_ax.imshow(
        payload.confusion_matrix.numpy(),
        cmap=confusion_cmap,
        interpolation="nearest",
    )
    num_classes = payload.confusion_matrix.shape[0]
    confusion_ax.set_xlabel("Predicted label")
    confusion_ax.set_ylabel("True label")
    confusion_ax.set_xticks(range(num_classes), labels=payload.class_labels)
    confusion_ax.set_yticks(range(num_classes), labels=payload.class_labels)

    threshold = float(payload.confusion_matrix.max().item()) / 2.0 if payload.confusion_matrix.numel() > 0 else 0.0
    for row_index in range(num_classes):
        for column_index in range(num_classes):
            cell_value = int(payload.confusion_matrix[row_index, column_index].item())
            text_color = "white" if cell_value > threshold and threshold > 0.0 else "black"
            confusion_ax.text(
                column_index,
                row_index,
                f"{cell_value:d}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    if _show_error_analysis_colorbar():
        confusion_ax.figure.colorbar(confusion_image, ax=confusion_ax, fraction=0.046, pad=0.04)


def _plot_error_analysis_gallery_panel(
    gallery_anchor_ax: Axes,
    payload: ErrorAnalysisPayload,
    *,
    max_examples: int = DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES,
    example_grid_shape: tuple[int, int] = DEFAULT_ERROR_ANALYSIS_GRID_SHAPE,
    predicted_label_color: str = "tab:red",
    true_label_color: str = "tab:blue",
) -> None:
    rows, cols = _validate_example_grid_shape(example_grid_shape, max_examples=max_examples)
    predicted_label_style = _predicted_error_analysis_label_style()
    true_label_style = _true_error_analysis_label_style()

    gallery_anchor_ax.axis("off")

    visible_examples = min(max_examples, payload.num_examples)
    if visible_examples <= 0:
        gallery_anchor_ax.text(0.5, 0.5, "No misclassified samples", ha="center", va="center")
        return

    figure = gallery_anchor_ax.figure
    figure.canvas.draw()
    anchor_bbox = gallery_anchor_ax.get_position()
    figure_width, figure_height = figure.get_size_inches()
    anchor_rect = _ErrorAnalysisRect(
        x0=anchor_bbox.x0 * figure_width,
        y0=anchor_bbox.y0 * figure_height,
        width=anchor_bbox.width * figure_width,
        height=anchor_bbox.height * figure_height,
    )
    gallery_geometry = _resolve_error_analysis_gallery_geometry(
        anchor_rect,
        example_grid_shape,
        max_examples=max_examples,
    )
    for gallery_index in range(rows * cols):
        tile_rect = gallery_geometry.tile_rects[gallery_index]
        gallery_ax = figure.add_axes(
            [
                tile_rect.x0 / figure_width,
                tile_rect.y0 / figure_height,
                tile_rect.width / figure_width,
                tile_rect.height / figure_height,
            ]
        )
        gallery_ax.set_in_layout(False)
        gallery_ax.set_xticks([])
        gallery_ax.set_yticks([])

        if gallery_index >= visible_examples:
            gallery_ax.axis("off")
            continue

        image = payload.misclassified_images[gallery_index]
        true_label = int(payload.misclassified_true_labels[gallery_index].item())
        predicted_label = int(payload.misclassified_predicted_labels[gallery_index].item())
        gallery_ax.imshow(image.numpy(), cmap="gray_r", interpolation="nearest")
        gallery_ax.text(
            predicted_label_style.x,
            predicted_label_style.y,
            payload.class_labels[predicted_label],
            transform=gallery_ax.transAxes,
            ha=predicted_label_style.horizontal_alignment,
            va=predicted_label_style.vertical_alignment,
            color=predicted_label_color,
            fontsize=predicted_label_style.fontsize,
        )
        gallery_ax.text(
            true_label_style.x,
            true_label_style.y,
            payload.class_labels[true_label],
            transform=gallery_ax.transAxes,
            ha=true_label_style.horizontal_alignment,
            va=true_label_style.vertical_alignment,
            color=true_label_color,
            fontsize=true_label_style.fontsize,
        )


def plot_error_analysis_payload(
    payload: ErrorAnalysisPayload,
    *,
    max_examples: int = DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES,
    example_grid_shape: tuple[int, int] = DEFAULT_ERROR_ANALYSIS_GRID_SHAPE,
    figsize: tuple[float, float] = (12, 4.5),
    confusion_cmap: str = "gray_r",
    predicted_label_color: str = "tab:red",
    true_label_color: str = "tab:blue",
) -> Axes:
    """Render a previously collected error-analysis payload."""

    _validate_max_examples(max_examples)
    _validate_example_grid_shape(example_grid_shape, max_examples=max_examples)
    _validate_error_analysis_payload(payload)
    plt = _require_matplotlib()

    figure = plt.figure(figsize=figsize, layout="constrained")
    outer_grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.1))

    confusion_ax = figure.add_subplot(outer_grid[0, 0])
    gallery_anchor_ax = figure.add_subplot(outer_grid[0, 1])
    _plot_error_analysis_confusion_panel(
        confusion_ax,
        payload,
        confusion_cmap=confusion_cmap,
    )
    _plot_error_analysis_gallery_panel(
        gallery_anchor_ax,
        payload,
        max_examples=max_examples,
        example_grid_shape=example_grid_shape,
        predicted_label_color=predicted_label_color,
        true_label_color=true_label_color,
    )

    return confusion_ax


def plot_convergence(
    history: TrainingHistory,
    *,
    metric_name: str = "accuracy",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the epoch-level loss and metric traces stored in ``TrainingHistory``.

    Args:
        history: Epoch-level training history returned by
            ``ImageClassifierRunner.fit``.
        metric_name: Name of the metric from ``history.train_metrics`` /
            ``history.test_metrics`` to render on the right-hand subplot.
        figsize: Forwarded to ``matplotlib`` when constructing the figure.

    Returns:
        A tuple ``(figure, (loss_ax, metric_ax))`` containing the newly created
        ``matplotlib`` figure and its two main axes.

    Notes:
        The helper reproduces the training notebook's compact two-panel layout:
        loss on the left and one selected metric on the right. It validates that
        every rendered series has the same epoch length before plotting.
    """

    plt = _require_matplotlib()
    _validate_history(history, metric_name)

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, layout="compressed")
    loss_ax, metric_ax = axes

    loss_ax.plot(history.train_epoch, history.train_loss, label="Train")
    loss_ax.plot(history.test_epoch, history.test_loss, label="Test")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-Entropy")
    loss_ax.set_title("Loss")
    loss_ax.legend()

    metric_label = "Accuracy" if metric_name == "accuracy" else metric_name
    metric_ax.plot(history.train_epoch, history.train_metrics[metric_name], label="Train")
    metric_ax.plot(history.test_epoch, history.test_metrics[metric_name], label="Test")
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_ylabel(metric_label)
    metric_ax.set_title(metric_label)
    metric_ax.legend()

    return figure, (loss_ax, metric_ax)


def plot_error_analysis(
    runner: ImageClassifierRunner,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    class_labels: Sequence[str] | None = None,
    max_examples: int = DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES,
    example_grid_shape: tuple[int, int] = DEFAULT_ERROR_ANALYSIS_GRID_SHAPE,
    figsize: tuple[float, float] = (12, 4.5),
) -> Axes:
    """Plot a confusion matrix and a gallery of misclassified samples.

    Args:
        runner: ``ImageClassifierRunner`` used to compute logits for the
            provided loader.
        data_loader: Iterable of ``(images, labels)`` batches following the
            standard ``[B, X, Y]`` grayscale-image contract.
        class_labels: Optional class names used for the confusion-matrix tick
            labels and the gallery corner annotations.
        max_examples: Maximum number of misclassified examples to display in
            the right-hand gallery.
        example_grid_shape: Grid layout ``(rows, cols)`` for the gallery.
            ``rows * cols`` must be at least ``max_examples``.
        figsize: Forwarded to ``matplotlib`` when constructing the figure.

    Returns:
        The left-hand confusion-matrix ``Axes`` from a newly created
        ``matplotlib`` figure that also contains the misclassification gallery
        on the right. Returning the axis keeps the helper directly compatible
        with ``mo.ui.matplotlib(...)`` in the notebook.

    Notes:
        The helper performs one no-grad pass over ``data_loader`` using
        ``runner.run_forward_pass``. It temporarily switches the wrapped model
        into evaluation mode and restores the original ``model.training`` flag
        before returning, even if plotting fails partway through.
    """
    payload = collect_error_analysis_payload(
        runner,
        data_loader,
        class_labels=class_labels,
        max_examples=max_examples,
    )
    return plot_error_analysis_payload(
        payload,
        max_examples=max_examples,
        example_grid_shape=example_grid_shape,
        figsize=figsize,
    )
