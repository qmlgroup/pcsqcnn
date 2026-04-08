import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

import pytest
import qcnn.visualization as visualization_module
import torch
import matplotlib.pyplot as plt

from .script_loading import load_run_script

plot_figure_s4 = load_run_script(
    "plot_error_structure_module",
    "plot_error_structure.py",
)


def make_run_directory(
    tmp_path: Path,
    *,
    scaled_image_size: int = 16,
    image_size: int = 16,
    seeds: list[int] | None = None,
) -> Path:
    run_directory = (
        tmp_path
        / "artifacts"
        / plot_figure_s4.DEFAULT_SWEEP_DIRECTORY_NAME
        / f"{scaled_image_size}on{image_size}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(
            {"seeds": [7] if seeds is None else seeds, "runs": []},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run_directory


def write_payload(
    run_directory: Path,
    *,
    seed: int = 7,
    image_size: int = 16,
    num_examples: int = 3,
) -> Path:
    payload_path = plot_figure_s4.build_error_structure_output_path(run_directory, seed=seed)
    payload = {
        "seed": seed,
        "image_size": image_size,
        "class_labels": [str(index) for index in range(10)],
        "confusion_matrix": torch.eye(10, dtype=torch.long),
        "misclassified_images": torch.arange(num_examples * 4 * 4, dtype=torch.float32).reshape(num_examples, 4, 4),
        "misclassified_true_labels": torch.arange(num_examples, dtype=torch.long),
        "misclassified_predicted_labels": torch.arange(num_examples, dtype=torch.long).flip(0),
    }
    torch.save(payload, payload_path)
    return payload_path


def test_parse_args_defaults_to_project_paths() -> None:
    args = plot_figure_s4.parse_args([])

    assert args.artifacts_root == plot_figure_s4.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_s4.DEFAULT_OUTPUT_DIR
    assert args.image_size == plot_figure_s4.DEFAULT_IMAGE_SIZE
    assert args.scaled_image_size == plot_figure_s4.DEFAULT_SCALED_IMAGE_SIZE
    assert args.seed is None


def test_parse_args_accepts_overrides() -> None:
    args = plot_figure_s4.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
            "--image-size",
            "32",
            "--scaled-image-size",
            "28",
            "--seed",
            "11",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")
    assert args.image_size == 32
    assert args.scaled_image_size == 28
    assert args.seed == 11


def test_build_figure_5b_run_directory_and_output_path(tmp_path: Path) -> None:
    run_directory = plot_figure_s4.build_figure_5b_run_directory(
        tmp_path / "artifacts",
        scaled_image_size=8,
        image_size=8,
    )
    output_path = plot_figure_s4.build_error_structure_output_path(run_directory, seed=5)

    assert run_directory == tmp_path / "artifacts" / "pcsqcnn_image_size_sweep" / "8on8"
    assert output_path == run_directory / "error_structure_seed5.pt"


def test_resolve_error_structure_seed_supports_singleton_and_explicit_seed(tmp_path: Path) -> None:
    singleton_run = make_run_directory(tmp_path, seeds=[7])
    multi_seed_run = make_run_directory(tmp_path / "other", seeds=[3, 5, 7])

    assert plot_figure_s4.resolve_error_structure_seed(singleton_run, seed=None) == 7
    assert plot_figure_s4.resolve_error_structure_seed(multi_seed_run, seed=None) == 3
    assert plot_figure_s4.resolve_error_structure_seed(multi_seed_run, seed=5) == 5


def test_resolve_error_structure_seed_rejects_missing_seed(tmp_path: Path) -> None:
    run_directory = make_run_directory(tmp_path, seeds=[3, 5, 7])

    with pytest.raises(ValueError, match="Available seeds: \\[3, 5, 7\\]"):
        plot_figure_s4.resolve_error_structure_seed(run_directory, seed=11)


def test_load_error_structure_payload_validates_and_restores_saved_payload(tmp_path: Path) -> None:
    run_directory = make_run_directory(tmp_path, seeds=[7])
    payload_path = write_payload(run_directory, num_examples=3)

    saved_payload = plot_figure_s4.load_error_structure_payload(payload_path)

    assert saved_payload.seed == 7
    assert saved_payload.image_size == 16
    assert saved_payload.payload.class_labels == tuple(str(index) for index in range(10))
    assert saved_payload.payload.confusion_matrix.shape == (10, 10)
    assert saved_payload.payload.num_examples == 3


def test_load_error_structure_payload_rejects_mismatched_example_lengths(tmp_path: Path) -> None:
    payload_path = tmp_path / "bad.pt"
    torch.save(
        {
            "seed": 7,
            "image_size": 16,
            "class_labels": [str(index) for index in range(10)],
            "confusion_matrix": torch.eye(10, dtype=torch.long),
            "misclassified_images": torch.zeros((2, 4, 4), dtype=torch.float32),
            "misclassified_true_labels": torch.tensor([1], dtype=torch.long),
            "misclassified_predicted_labels": torch.tensor([2], dtype=torch.long),
        },
        payload_path,
    )

    with pytest.raises(ValueError, match="batch size must match the saved true labels"):
        plot_figure_s4.load_error_structure_payload(payload_path)


def test_resolve_visible_example_count_caps_gallery_to_sixteen(tmp_path: Path) -> None:
    run_directory = make_run_directory(tmp_path, seeds=[7])
    payload_path = write_payload(run_directory, num_examples=25)
    saved_payload = plot_figure_s4.load_error_structure_payload(payload_path)

    assert plot_figure_s4.resolve_visible_example_count(saved_payload) == 16
    assert plot_figure_s4.resolve_visible_example_count(saved_payload, max_examples=7) == 7


def test_plot_article_figure_s4a_forwards_confusion_panel_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = plot_figure_s4.SavedErrorStructurePayload(
        seed=7,
        image_size=16,
        payload=visualization_module.ErrorAnalysisPayload(
            class_labels=tuple(str(index) for index in range(10)),
            confusion_matrix=torch.eye(10, dtype=torch.long),
            misclassified_images=torch.zeros((2, 4, 4), dtype=torch.float32),
            misclassified_true_labels=torch.tensor([1, 2], dtype=torch.long),
            misclassified_predicted_labels=torch.tensor([2, 1], dtype=torch.long),
        ),
    )
    captured: dict[str, object] = {}

    def fake_load_error_structure_payload(payload_path: Path) -> plot_figure_s4.SavedErrorStructurePayload:
        captured["payload_path"] = payload_path
        return payload

    def fake_plot_error_analysis_confusion_panel(
        confusion_ax: object,
        forwarded_payload: visualization_module.ErrorAnalysisPayload,
        *,
        confusion_cmap: str,
    ) -> None:
        captured["panel_kwargs"] = {
            "confusion_ax": confusion_ax,
            "payload": forwarded_payload,
            "confusion_cmap": confusion_cmap,
        }

    monkeypatch.setattr(plot_figure_s4, "load_error_structure_payload", fake_load_error_structure_payload)
    monkeypatch.setattr(
        plot_figure_s4,
        "_plot_error_analysis_confusion_panel",
        fake_plot_error_analysis_confusion_panel,
    )

    figure = plot_figure_s4.plot_article_figure_s4a(payload_path=Path("/tmp/error_structure_seed7.pt"))

    assert captured["payload_path"] == Path("/tmp/error_structure_seed7.pt")
    assert captured["panel_kwargs"]["payload"] is payload.payload
    assert captured["panel_kwargs"]["confusion_cmap"] == plot_figure_s4.DEFAULT_CONFUSION_CMAP
    assert captured["panel_kwargs"]["confusion_ax"].figure is figure
    plt.close(figure)


def test_plot_article_figure_s4b_forwards_gallery_panel_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = plot_figure_s4.SavedErrorStructurePayload(
        seed=7,
        image_size=16,
        payload=visualization_module.ErrorAnalysisPayload(
            class_labels=tuple(str(index) for index in range(10)),
            confusion_matrix=torch.eye(10, dtype=torch.long),
            misclassified_images=torch.zeros((2, 4, 4), dtype=torch.float32),
            misclassified_true_labels=torch.tensor([1, 2], dtype=torch.long),
            misclassified_predicted_labels=torch.tensor([2, 1], dtype=torch.long),
        ),
    )
    captured: dict[str, object] = {}

    def fake_load_error_structure_payload(payload_path: Path) -> plot_figure_s4.SavedErrorStructurePayload:
        captured["payload_path"] = payload_path
        return payload

    def fake_plot_error_analysis_gallery_panel(
        gallery_anchor_ax: object,
        forwarded_payload: visualization_module.ErrorAnalysisPayload,
        *,
        max_examples: int,
        example_grid_shape: tuple[int, int],
        predicted_label_color: str,
        true_label_color: str,
    ) -> None:
        captured["panel_kwargs"] = {
            "gallery_anchor_ax": gallery_anchor_ax,
            "payload": forwarded_payload,
            "max_examples": max_examples,
            "example_grid_shape": example_grid_shape,
            "predicted_label_color": predicted_label_color,
            "true_label_color": true_label_color,
        }

    monkeypatch.setattr(plot_figure_s4, "load_error_structure_payload", fake_load_error_structure_payload)
    monkeypatch.setattr(
        plot_figure_s4,
        "_plot_error_analysis_gallery_panel",
        fake_plot_error_analysis_gallery_panel,
    )

    figure = plot_figure_s4.plot_article_figure_s4b(payload_path=Path("/tmp/error_structure_seed7.pt"))

    assert captured["payload_path"] == Path("/tmp/error_structure_seed7.pt")
    assert captured["panel_kwargs"]["payload"] is payload.payload
    assert captured["panel_kwargs"]["max_examples"] == visualization_module.DEFAULT_ERROR_ANALYSIS_MAX_EXAMPLES
    assert captured["panel_kwargs"]["example_grid_shape"] == visualization_module.DEFAULT_ERROR_ANALYSIS_GRID_SHAPE
    assert captured["panel_kwargs"]["predicted_label_color"] == plot_figure_s4.DEFAULT_PREDICTED_LABEL_COLOR
    assert captured["panel_kwargs"]["true_label_color"] == plot_figure_s4.DEFAULT_TRUE_LABEL_COLOR
    assert captured["panel_kwargs"]["gallery_anchor_ax"].figure is figure
    plt.close(figure)


def test_main_saves_both_panel_outputs_without_rendering(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "error_structure_seed7.pt"
    output_dir = tmp_path / "figs"
    saved_payload = plot_figure_s4.SavedErrorStructurePayload(
        seed=7,
        image_size=16,
        payload=visualization_module.ErrorAnalysisPayload(
            class_labels=tuple(str(index) for index in range(10)),
            confusion_matrix=torch.eye(10, dtype=torch.long),
            misclassified_images=torch.zeros((2, 4, 4), dtype=torch.float32),
            misclassified_true_labels=torch.tensor([1, 2], dtype=torch.long),
            misclassified_predicted_labels=torch.tensor([2, 1], dtype=torch.long),
        ),
    )
    saved_paths: list[Path] = []
    closed_figures: list[object] = []

    class FakeFigure:
        def __init__(self, label: str) -> None:
            self.label = label

        def savefig(self, path: Path) -> None:
            saved_paths.append(path)

    class FakePyplot:
        @staticmethod
        def close(figure: object) -> None:
            closed_figures.append(figure)

    confusion_figure = FakeFigure("S4a")
    gallery_figure = FakeFigure("S4b")

    monkeypatch.setattr(
        plot_figure_s4,
        "parse_args",
        lambda argv=None: plot_figure_s4.argparse.Namespace(
            artifacts_root=tmp_path / "artifacts",
            output_dir=output_dir,
            image_size=16,
            scaled_image_size=16,
            seed=None,
        ),
    )
    monkeypatch.setattr(plot_figure_s4, "_require_matplotlib", lambda: FakePyplot())
    monkeypatch.setattr(plot_figure_s4, "build_figure_5b_run_directory", lambda *args, **kwargs: tmp_path / "run")
    monkeypatch.setattr(plot_figure_s4, "resolve_error_structure_seed", lambda *args, **kwargs: 7)
    monkeypatch.setattr(plot_figure_s4, "build_error_structure_output_path", lambda *args, **kwargs: payload_path)
    monkeypatch.setattr(plot_figure_s4, "load_error_structure_payload", lambda *args, **kwargs: saved_payload)
    monkeypatch.setattr(
        plot_figure_s4,
        "_plot_article_figure_s4a_from_saved_payload",
        lambda saved_payload, *, figsize=plot_figure_s4.ARTICLE_PANEL_FIGSIZE: confusion_figure,
    )
    monkeypatch.setattr(
        plot_figure_s4,
        "_plot_article_figure_s4b_from_saved_payload",
        lambda saved_payload, *, figsize=plot_figure_s4.ARTICLE_PANEL_FIGSIZE: gallery_figure,
    )

    plot_figure_s4.main([])

    assert saved_paths == [
        output_dir.resolve() / plot_figure_s4.DEFAULT_CONFUSION_OUTPUT_FILENAME,
        output_dir.resolve() / plot_figure_s4.DEFAULT_GALLERY_OUTPUT_FILENAME,
    ]
    assert closed_figures == [confusion_figure, gallery_figure]
    assert capsys.readouterr().out.splitlines() == [
        str(output_dir.resolve() / plot_figure_s4.DEFAULT_CONFUSION_OUTPUT_FILENAME),
        str(output_dir.resolve() / plot_figure_s4.DEFAULT_GALLERY_OUTPUT_FILENAME),
    ]
