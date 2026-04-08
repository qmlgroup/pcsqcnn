import importlib.util
import json
from pathlib import Path
import sys
import pytest
import torch

from qcnn import TrainingHistory

PLOT_BRIGHTNESS_SWEEP_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_brightness_sweep.py"
PLOT_BRIGHTNESS_SWEEP_SPEC = importlib.util.spec_from_file_location(
    "plot_brightness_sweep_module",
    PLOT_BRIGHTNESS_SWEEP_PATH,
)
if PLOT_BRIGHTNESS_SWEEP_SPEC is None or PLOT_BRIGHTNESS_SWEEP_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_brightness_sweep.py from {PLOT_BRIGHTNESS_SWEEP_PATH}.")
plot_brightness_sweep = importlib.util.module_from_spec(PLOT_BRIGHTNESS_SWEEP_SPEC)
sys.modules[PLOT_BRIGHTNESS_SWEEP_SPEC.name] = plot_brightness_sweep
PLOT_BRIGHTNESS_SWEEP_SPEC.loader.exec_module(plot_brightness_sweep)


def make_history(
    *,
    train_epoch: list[int] | None = None,
    test_epoch: list[int] | None = None,
    test_accuracy: list[float] | None = None,
) -> TrainingHistory:
    resolved_train_epoch = train_epoch or [1, 150, 300]
    resolved_test_epoch = test_epoch or [1, 150, 300]
    resolved_test_accuracy = (
        test_accuracy
        if test_accuracy is not None
        else [0.4 + 0.1 * idx for idx in range(len(resolved_test_epoch))]
    )
    return TrainingHistory(
        train_epoch=resolved_train_epoch,
        test_epoch=resolved_test_epoch,
        train_loss=[1.2 - 0.1 * idx for idx in range(len(resolved_train_epoch))],
        test_loss=[1.3 - 0.1 * idx for idx in range(len(resolved_test_epoch))],
        train_metrics={"accuracy": [0.5 + 0.1 * idx for idx in range(len(resolved_train_epoch))]},
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
def test_parse_sweep_run_directory_name_extracts_architecture_and_brightness() -> None:
    architecture, brightness_pi = plot_brightness_sweep.parse_sweep_run_directory_name("fq1_ql2_u2by25pi")

    assert architecture.feature_qubits == 1
    assert architecture.quantum_layers == 2
    assert architecture.label == r"Q=2, $n_f=1$"
    assert brightness_pi == pytest.approx(2 / 25)

    architecture, brightness_pi = plot_brightness_sweep.parse_sweep_run_directory_name("fq2_ql1_u2by5pi")
    assert architecture.feature_qubits == 2
    assert architecture.quantum_layers == 1
    assert brightness_pi == pytest.approx(2 / 5)


def test_parse_args_defaults_to_project_artifacts_output_dir_and_epochs() -> None:
    args = plot_brightness_sweep.parse_args([])

    assert args.artifacts_root == plot_brightness_sweep.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_brightness_sweep.DEFAULT_OUTPUT_DIR
    assert args.epochs == list(plot_brightness_sweep.DEFAULT_EPOCHS)


def test_parse_args_accepts_artifacts_output_dir_and_epoch_overrides() -> None:
    default_args = plot_brightness_sweep.parse_args([])
    custom_epoch = max(default_args.epochs) + 1
    args = plot_brightness_sweep.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
            "--epochs",
            str(custom_epoch),
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")
    assert len(args.epochs) == 1
    assert args.epochs != default_args.epochs
    assert args.epochs[0] > max(default_args.epochs)


def test_summarize_brightness_sweep_epoch_groups_and_sorts_available_runs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_brightness_sweep"
    epoch_grid = [1, 150, 300]

    write_fake_artifact_run(
        sweep_root / "fq1_ql1_u4by25pi",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.2, 0.60, 0.80])},
            1: {"seed": 1, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.3, 0.80, 1.00])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq1_ql1_u2by25pi",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.1, 0.50, 0.70])},
            1: {"seed": 1, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.2, 0.70, 0.90])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq2_ql1_u2by25pi",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.1, 0.40, 0.60])},
            1: {"seed": 1, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.2, 0.60, 0.80])},
        },
    )

    series_by_architecture = plot_brightness_sweep.summarize_brightness_sweep_epoch(
        sweep_root=sweep_root,
        epoch=150,
    )

    assert [series.architecture.label for series in series_by_architecture] == [
        r"Q=1, $n_f=1$",
        r"Q=1, $n_f=2$",
    ]
    assert series_by_architecture[0].brightness_pi == pytest.approx([2 / 25, 4 / 25])
    assert series_by_architecture[0].mean == pytest.approx([0.60, 0.70])
    assert series_by_architecture[0].minimum == pytest.approx([0.50, 0.60])
    assert series_by_architecture[0].maximum == pytest.approx([0.70, 0.80])
    assert series_by_architecture[1].brightness_pi == pytest.approx([2 / 25])
    assert series_by_architecture[1].mean == pytest.approx([0.50])
    assert series_by_architecture[1].minimum == pytest.approx([0.40])
    assert series_by_architecture[1].maximum == pytest.approx([0.60])


def test_summarize_brightness_sweep_epoch_warns_and_skips_incomplete_runs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_brightness_sweep"
    epoch_grid = [1, 150, 300]

    (sweep_root / "fq1_ql1_u2by25pi").mkdir(parents=True, exist_ok=True)
    write_fake_artifact_run(
        sweep_root / "fq1_ql1_u4by25pi",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=[1, 2, 3], test_accuracy=[0.1, 0.2, 0.3])},
            1: {"seed": 1, "training_history": make_history(test_epoch=[1, 2, 3], test_accuracy=[0.2, 0.3, 0.4])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq1_ql1_u6by25pi",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.2, 0.60, 0.80])},
            1: {"seed": 1, "training_history": make_history(test_epoch=epoch_grid, test_accuracy=[0.3, 0.80, 1.00])},
        },
    )

    with pytest.warns(UserWarning) as recorded_warnings:
        series_by_architecture = plot_brightness_sweep.summarize_brightness_sweep_epoch(
            sweep_root=sweep_root,
            epoch=150,
        )

    warning_messages = [str(warning.message) for warning in recorded_warnings]
    assert any("fq1_ql1_u2by25pi" in message for message in warning_messages)
    assert any("fq1_ql1_u4by25pi" in message for message in warning_messages)
    assert len(series_by_architecture) == 1
    assert series_by_architecture[0].brightness_pi == pytest.approx([6 / 25])
