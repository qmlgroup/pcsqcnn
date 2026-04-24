import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

from qcnn import TrainingHistory

PLOT_FIGURE_5B_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_pcsqcnn_image_size_sweep.py"
PLOT_FIGURE_5B_SPEC = importlib.util.spec_from_file_location("plot_pcsqcnn_image_size_sweep_module", PLOT_FIGURE_5B_PATH)
if PLOT_FIGURE_5B_SPEC is None or PLOT_FIGURE_5B_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_pcsqcnn_image_size_sweep.py from {PLOT_FIGURE_5B_PATH}.")
plot_figure_5b = importlib.util.module_from_spec(PLOT_FIGURE_5B_SPEC)
sys.modules[PLOT_FIGURE_5B_SPEC.name] = plot_figure_5b
PLOT_FIGURE_5B_SPEC.loader.exec_module(plot_figure_5b)


def make_history(
    *,
    train_epoch: list[int] | None = None,
    test_epoch: list[int] | None = None,
    test_accuracy: list[float] | None = None,
) -> TrainingHistory:
    resolved_train_epoch = train_epoch or list(range(1, 31))
    resolved_test_epoch = test_epoch or [10, 20, 30]
    resolved_test_accuracy = (
        test_accuracy
        if test_accuracy is not None
        else [0.4 + 0.1 * index for index in range(len(resolved_test_epoch))]
    )
    return TrainingHistory(
        train_epoch=resolved_train_epoch,
        test_epoch=resolved_test_epoch,
        train_loss=[1.5 - 0.02 * index for index in range(len(resolved_train_epoch))],
        test_loss=[1.3 - 0.1 * index for index in range(len(resolved_test_epoch))],
        train_metrics={"accuracy": [0.3 + 0.02 * index for index in range(len(resolved_train_epoch))]},
        test_metrics={"accuracy": resolved_test_accuracy},
    )


def write_fake_artifact_run(
    run_directory: Path,
    *,
    seed_order: list[int],
    payloads_by_seed: dict[int, dict[str, object]],
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in seed_order:
        result_name = f"result_seed{seed}.pt"
        torch.save(payloads_by_seed[seed], run_directory / result_name)
        runs.append({"seed": seed, "result": result_name})

    manifest = {"seeds": seed_order, "runs": runs}
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
def test_parse_image_size_run_directory_name_extracts_scaled_and_canvas_sizes() -> None:
    assert plot_figure_5b.parse_image_size_run_directory_name("8on8") == (8, 8)
    assert plot_figure_5b.parse_image_size_run_directory_name("28on32") == (28, 32)


def test_parse_args_defaults_to_project_artifacts_and_output_dirs() -> None:
    args = plot_figure_5b.parse_args([])

    assert args.artifacts_root == plot_figure_5b.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_5b.DEFAULT_OUTPUT_DIR
    assert args.epoch_group_size == plot_figure_5b.DEFAULT_EPOCH_GROUP_SIZE


def test_parse_args_accepts_artifacts_and_output_dir_overrides() -> None:
    args = plot_figure_5b.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")


def test_parse_args_accepts_temporal_summary_overrides() -> None:
    args = plot_figure_5b.parse_args(
        [
            "--epoch-start",
            "100",
            "--epoch-end",
            "500",
            "--epoch-group-size",
            "3",
            "--lower-percentile",
            "5",
            "--upper-percentile",
            "95",
        ]
    )

    assert args.epoch_start == 100
    assert args.epoch_end == 500
    assert args.epoch_group_size == 3
    assert args.lower_percentile == 5.0
    assert args.upper_percentile == 95.0


def test_summarize_image_size_sweep_groups_and_sorts_runs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_image_size_sweep"
    test_epoch = [10, 20, 30]

    write_fake_artifact_run(
        sweep_root / "16on16",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.4, 0.6, 0.7])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.6, 0.8, 0.9])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "8on8",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.3, 0.5, 0.6])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.5, 0.7, 0.8])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "32on32",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.2, 0.4, 0.5])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.4, 0.6, 0.7])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "28on32",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.35, 0.55, 0.65])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.45, 0.65, 0.75])},
        },
    )

    series_by_size = plot_figure_5b.summarize_image_size_sweep(sweep_root=sweep_root)

    assert [(series.scaled_image_size, series.image_size) for series in series_by_size] == [
        (8, 8),
        (16, 16),
        (28, 32),
        (32, 32),
    ]
    assert [series.label for series in series_by_size] == [
        "8(8)",
        "16(16)",
        "28(32)",
        "32(32)",
    ]
    assert series_by_size[0].summary.epoch == [10, 20, 30]
    assert series_by_size[0].summary.mean == pytest.approx([0.4, 0.6, 0.7])
    assert series_by_size[0].summary.lower == pytest.approx([0.35, 0.55, 0.65])
    assert series_by_size[0].summary.upper == pytest.approx([0.45, 0.65, 0.75])
    assert series_by_size[2].summary.mean == pytest.approx([0.40, 0.60, 0.70])


def test_summarize_image_size_sweep_supports_epoch_window_and_grouping(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_image_size_sweep"
    write_fake_artifact_run(
        sweep_root / "16on16",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=[10, 20, 30], test_accuracy=[0.2, 0.4, 0.8])},
            1: {"seed": 1, "training_history": make_history(test_epoch=[10, 20, 30], test_accuracy=[0.4, 0.6, 1.0])},
        },
    )

    series_by_size = plot_figure_5b.summarize_image_size_sweep(
        sweep_root=sweep_root,
        epoch_end=20,
        epoch_group_size=20,
    )

    assert len(series_by_size) == 1
    assert series_by_size[0].summary.epoch == pytest.approx([15.0])
    assert series_by_size[0].summary.mean == pytest.approx([0.4])
    assert series_by_size[0].summary.lower == pytest.approx([0.35])
    assert series_by_size[0].summary.upper == pytest.approx([0.45])
