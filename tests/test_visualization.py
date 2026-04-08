import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
import qcnn.visualization as visualization_module
import torch
from torch import nn
from torch.utils.data import DataLoader

from qcnn import (
    ImageClassifierRunner,
    TensorImageDataset,
    TrainingHistory,
    collect_error_analysis_payload,
    plot_convergence,
)


class LookupLogitModel(nn.Module):
    def __init__(self, logits_by_code: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits_by_code", logits_by_code.to(dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sample_codes = images[:, 0, 0].round().to(dtype=torch.long)
        return self.logits_by_code[sample_codes].to(device=images.device)


def make_history() -> TrainingHistory:
    return TrainingHistory(
        train_epoch=[1, 2, 3],
        test_epoch=[1, 2, 3],
        train_loss=[1.2, 0.9, 0.6],
        test_loss=[1.3, 1.0, 0.8],
        train_metrics={"accuracy": [0.5, 0.7, 0.8]},
        test_metrics={"accuracy": [0.4, 0.6, 0.75]},
    )


def make_lookup_loader(labels: list[int]) -> DataLoader:
    sample_codes = torch.arange(len(labels), dtype=torch.float32)
    images = sample_codes[:, None, None].expand(-1, 4, 4).clone()
    dataset = TensorImageDataset(images, torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=2, shuffle=False)


def make_error_analysis_payload() -> visualization_module.ErrorAnalysisPayload:
    return visualization_module.ErrorAnalysisPayload(
        class_labels=("0", "1", "2"),
        confusion_matrix=torch.tensor(
            [
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=torch.long,
        ),
        misclassified_images=torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4),
        misclassified_true_labels=torch.tensor([1, 2, 0], dtype=torch.long),
        misclassified_predicted_labels=torch.tensor([2, 1, 1], dtype=torch.long),
    )


def test_plot_convergence_returns_expected_axes_labels_and_titles() -> None:
    history = make_history()

    figure, (loss_ax, metric_ax) = plot_convergence(history)

    assert loss_ax.get_xlabel() == "Epoch"
    assert loss_ax.get_ylabel() == "Cross-Entropy"
    assert loss_ax.get_title() == "Loss"
    assert metric_ax.get_xlabel() == "Epoch"
    assert metric_ax.get_ylabel() == "Accuracy"
    assert metric_ax.get_title() == "Accuracy"
    assert len(loss_ax.lines) == 2
    assert len(metric_ax.lines) == 2

    plt.close(figure)


def test_plot_convergence_rejects_missing_metric_name() -> None:
    history = make_history()

    with pytest.raises(ValueError, match="metric_name"):
        plot_convergence(history, metric_name="gap")


def test_plot_convergence_rejects_inconsistent_history_lengths() -> None:
    with pytest.raises(ValueError, match="train_loss"):
        TrainingHistory(
            train_epoch=[1, 2],
            test_epoch=[1, 2],
            train_loss=[1.0],
            test_loss=[1.1, 0.9],
            train_metrics={"accuracy": [0.4, 0.5]},
            test_metrics={"accuracy": [0.3, 0.4]},
        )


def test_plot_line_with_band_adds_matching_fill_and_line() -> None:
    figure, ax = plt.subplots()

    line = visualization_module.plot_line_with_band(
        ax,
        x=[1.0, 2.0, 3.0],
        y=[0.3, 0.5, 0.7],
        lower=[0.2, 0.4, 0.6],
        upper=[0.4, 0.6, 0.8],
        color="C2",
        label="demo",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=4.0,
        band_alpha=0.23,
        band_linewidth=0.0,
    )

    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    assert ax.lines[0] is line

    expected_rgba = mcolors.to_rgba("C2", alpha=0.23)
    facecolor = ax.collections[0].get_facecolor()[0]
    assert mcolors.to_rgba(line.get_color())[:3] == pytest.approx(expected_rgba[:3])
    assert facecolor[:3] == pytest.approx(expected_rgba[:3])
    assert facecolor[3] == pytest.approx(expected_rgba[3])
    assert line.get_label() == "demo"
    assert line.get_linestyle() == "--"
    assert line.get_linewidth() == pytest.approx(2.0)
    assert line.get_marker() == "o"
    assert line.get_markersize() == pytest.approx(4.0)

    plt.close(figure)


def test_error_analysis_geometry_helpers_maximize_gallery_size_and_equalize_gaps() -> None:
    tall_anchor_rect = visualization_module._ErrorAnalysisRect(
        x0=0.5,
        y0=0.1,
        width=0.3,
        height=0.6,
    )
    tall_container_rect = visualization_module._resolve_error_analysis_gallery_container(
        tall_anchor_rect,
        rows=4,
        cols=4,
        spacing_ratio=visualization_module.DEFAULT_ERROR_ANALYSIS_GALLERY_SPACING,
    )
    geometry = visualization_module._resolve_error_analysis_gallery_geometry(tall_anchor_rect)

    assert visualization_module.DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES == 16
    assert visualization_module.DEFAULT_ERROR_ANALYSIS_GRID_SHAPE == (4, 4)
    assert visualization_module._show_error_analysis_colorbar() is False
    assert tall_container_rect.width == pytest.approx(tall_anchor_rect.width)
    assert tall_container_rect.x0 == pytest.approx(tall_anchor_rect.x0)
    assert tall_container_rect.x1 == pytest.approx(tall_anchor_rect.x1)
    assert tall_container_rect.center_y == pytest.approx(tall_anchor_rect.center_y)
    assert geometry.rows == 4
    assert geometry.cols == 4
    assert geometry.container_rect == tall_container_rect
    assert len(geometry.tile_rects) == 16
    assert geometry.tile_size > 0.0
    assert geometry.gap_size > 0.0
    assert all(tile.width == pytest.approx(tile.height) for tile in geometry.tile_rects)

    wide_anchor_rect = visualization_module._ErrorAnalysisRect(
        x0=0.2,
        y0=0.1,
        width=0.6,
        height=0.3,
    )
    wide_container_rect = visualization_module._resolve_error_analysis_gallery_container(
        wide_anchor_rect,
        rows=4,
        cols=4,
        spacing_ratio=visualization_module.DEFAULT_ERROR_ANALYSIS_GALLERY_SPACING,
    )
    assert wide_container_rect.height == pytest.approx(wide_anchor_rect.height)
    assert wide_container_rect.y0 == pytest.approx(wide_anchor_rect.y0)
    assert wide_container_rect.y1 == pytest.approx(wide_anchor_rect.y1)
    assert wide_container_rect.center_x == pytest.approx(wide_anchor_rect.center_x)

    first_tile = geometry.tile_rects[0]
    second_tile = geometry.tile_rects[1]
    below_first_tile = geometry.tile_rects[4]

    assert first_tile.width == pytest.approx(first_tile.height)
    assert second_tile.width == pytest.approx(first_tile.width)
    assert below_first_tile.height == pytest.approx(first_tile.height)
    assert second_tile.x0 - first_tile.x1 == pytest.approx(geometry.gap_size)
    assert first_tile.y0 - below_first_tile.y1 == pytest.approx(geometry.gap_size)


def test_error_analysis_label_styles_are_inset_and_unboxed() -> None:
    predicted_style = visualization_module._predicted_error_analysis_label_style()
    true_style = visualization_module._true_error_analysis_label_style()

    assert predicted_style.bbox is None
    assert true_style.bbox is None
    assert 0.02 < predicted_style.x < 0.5
    assert 0.04 < predicted_style.y < 0.5
    assert predicted_style.horizontal_alignment == "left"
    assert predicted_style.vertical_alignment == "bottom"
    assert true_style.horizontal_alignment == "right"
    assert true_style.vertical_alignment == "bottom"
    assert 0.5 < true_style.x < 0.98
    assert true_style.y == pytest.approx(predicted_style.y)
    assert predicted_style.fontsize == 10
    assert true_style.fontsize == 10


def test_plot_error_analysis_forwards_payload_and_new_defaults_without_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = make_error_analysis_payload()
    captured: dict[str, object] = {}
    sentinel_axis = object()
    sentinel_runner = object()
    sentinel_loader = object()

    def fake_collect_error_analysis_payload(
        runner: object,
        data_loader: object,
        *,
        class_labels: tuple[str, ...] | None = None,
        max_examples: int | None = None,
    ) -> visualization_module.ErrorAnalysisPayload:
        captured["collect"] = {
            "runner": runner,
            "data_loader": data_loader,
            "class_labels": class_labels,
            "max_examples": max_examples,
        }
        return payload

    def fake_plot_error_analysis_payload(
        forwarded_payload: visualization_module.ErrorAnalysisPayload,
        *,
        max_examples: int,
        example_grid_shape: tuple[int, int],
        figsize: tuple[float, float],
    ) -> object:
        captured["plot"] = {
            "payload": forwarded_payload,
            "max_examples": max_examples,
            "example_grid_shape": example_grid_shape,
            "figsize": figsize,
        }
        return sentinel_axis

    monkeypatch.setattr(
        visualization_module,
        "collect_error_analysis_payload",
        fake_collect_error_analysis_payload,
    )
    monkeypatch.setattr(
        visualization_module,
        "plot_error_analysis_payload",
        fake_plot_error_analysis_payload,
    )

    returned_axis = visualization_module.plot_error_analysis(sentinel_runner, sentinel_loader)

    assert returned_axis is sentinel_axis
    assert captured["collect"] == {
        "runner": sentinel_runner,
        "data_loader": sentinel_loader,
        "class_labels": None,
        "max_examples": 16,
    }
    assert captured["plot"] == {
        "payload": payload,
        "max_examples": 16,
        "example_grid_shape": (4, 4),
        "figsize": (12, 4.5),
    }


def test_collect_error_analysis_payload_returns_full_confusion_matrix_and_examples() -> None:
    model = LookupLogitModel(
        torch.tensor(
            [
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, 5.0],
                [-5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
            ]
        )
    )
    runner = ImageClassifierRunner(model=model)
    runner.model.train(True)
    loader = make_lookup_loader([0, 1, 2, 0])

    payload = collect_error_analysis_payload(runner, loader, max_examples=None)

    assert payload.confusion_matrix.tolist() == [
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
    assert payload.class_labels == ("0", "1", "2")
    assert payload.num_examples == 3
    assert payload.misclassified_images.shape == (3, 4, 4)
    assert payload.misclassified_true_labels.tolist() == [1, 2, 0]
    assert payload.misclassified_predicted_labels.tolist() == [2, 1, 1]
    assert [(item.true_label, item.predicted_label) for item in payload.misclassified_examples()] == [
        (1, 2),
        (2, 1),
        (0, 1),
    ]
    assert runner.model.training is True


def test_collect_error_analysis_payload_respects_max_examples_limit() -> None:
    model = LookupLogitModel(
        torch.tensor(
            [
                [5.0, -5.0, -5.0],
                [-5.0, -5.0, 5.0],
                [-5.0, 5.0, -5.0],
                [-5.0, 5.0, -5.0],
            ]
        )
    )
    runner = ImageClassifierRunner(model=model)
    loader = make_lookup_loader([0, 1, 2, 0])

    payload = collect_error_analysis_payload(runner, loader, max_examples=2)

    assert payload.num_examples == 2
    assert payload.misclassified_true_labels.tolist() == [1, 2]
    assert payload.misclassified_predicted_labels.tolist() == [2, 1]
