import importlib.util
import inspect
import json
from pathlib import Path
import sys

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcnn import TrainingHistory


PLOT_FIGURE_2_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_translated_mnist_baselines.py"
PLOT_FIGURE_2_SPEC = importlib.util.spec_from_file_location("plot_translated_mnist_baselines_module", PLOT_FIGURE_2_PATH)
if PLOT_FIGURE_2_SPEC is None or PLOT_FIGURE_2_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_translated_mnist_baselines.py from {PLOT_FIGURE_2_PATH}.")
plot_figure_2 = importlib.util.module_from_spec(PLOT_FIGURE_2_SPEC)
sys.modules[PLOT_FIGURE_2_SPEC.name] = plot_figure_2
PLOT_FIGURE_2_SPEC.loader.exec_module(plot_figure_2)


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
        checkpoint_name = f"checkpoint_final_seed{seed}.pt"
        torch.save(payloads_by_seed[seed], run_directory / result_name)
        torch.save({"seed": seed}, run_directory / checkpoint_name)
        runs.append(
            {
                "seed": seed,
                "result": result_name,
                "checkpoint_final": checkpoint_name,
                "snapshots": {},
            }
        )

    manifest = {"seeds": seed_order, "runs": runs}
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_plot_figure_2_parse_args_defaults_match_plot_function_defaults() -> None:
    args = plot_figure_2.parse_args([])
    classical_panel_signature = inspect.signature(plot_figure_2.plot_article_figure_2a)
    quantum_panel_signature = inspect.signature(plot_figure_2.plot_article_figure_2b)

    assert args.samples_per_class == plot_figure_2.DEFAULT_SAMPLES_PER_CLASS
    assert args.artifacts_root == plot_figure_2.DEFAULT_ARTIFACTS_ROOT
    assert args.full_mnist_classical_artifacts_root == (
        plot_figure_2.DEFAULT_FULL_MNIST_CLASSICAL_ARTIFACTS_ROOT
    )
    assert (
        args.translated_scaled_image_size
        == plot_figure_2.DEFAULT_TRANSLATED_CLASSICAL_SCALED_IMAGE_SIZE
    )
    assert (
        args.full_mnist_scaled_image_size
        == plot_figure_2.DEFAULT_FULL_MNIST_CLASSICAL_SCALED_IMAGE_SIZE
    )
    assert args.epoch_start is None
    assert args.epoch_group_size is None
    assert args.lower_percentile == 25.0
    assert args.upper_percentile == 75.0
    assert (
        classical_panel_signature.parameters["epoch_group_size"].default
        == plot_figure_2.DEFAULT_CLASSICAL_EPOCH_GROUP_SIZE
    )
    assert (
        quantum_panel_signature.parameters["epoch_group_size"].default
        == plot_figure_2.DEFAULT_QUANTUM_EPOCH_GROUP_SIZE
    )
    assert args.quantum_epoch_start == quantum_panel_signature.parameters["epoch_start"].default


def test_plot_figure_2_parse_args_accepts_samples_per_class_override() -> None:
    default_args = plot_figure_2.parse_args([])
    custom_samples_per_class = default_args.samples_per_class + 1
    args = plot_figure_2.parse_args(["--samples-per-class", str(custom_samples_per_class)])

    assert args.samples_per_class != default_args.samples_per_class
    assert args.samples_per_class > 0


def test_plot_figure_2_parse_args_accepts_classical_root_and_size_overrides(tmp_path: Path) -> None:
    args = plot_figure_2.parse_args(
        [
            "--full-mnist-classical-artifacts-root",
            str(tmp_path / "full"),
            "--translated-scaled-image-size",
            "12",
            "--full-mnist-scaled-image-size",
            "32",
        ]
    )

    assert args.full_mnist_classical_artifacts_root == tmp_path / "full"
    assert args.translated_scaled_image_size == 12
    assert args.full_mnist_scaled_image_size == 32


def test_plot_figure_2_parse_args_accepts_epoch_window_overrides() -> None:
    custom_epoch_start = 25
    custom_epoch_end = 200
    args = plot_figure_2.parse_args(
        ["--epoch-start", str(custom_epoch_start), "--epoch-end", str(custom_epoch_end)]
    )

    assert args.epoch_start == custom_epoch_start
    assert args.epoch_end == custom_epoch_end
    assert args.epoch_start < args.epoch_end


def test_plot_figure_2_parse_args_accepts_temporal_summary_overrides() -> None:
    args = plot_figure_2.parse_args(
        [
            "--epoch-group-size",
            "5",
            "--lower-percentile",
            "10",
            "--upper-percentile",
            "90",
        ]
    )

    assert args.epoch_group_size == 5
    assert args.lower_percentile == 10.0
    assert args.upper_percentile == 90.0


def test_plot_figure_2_parse_args_accepts_quantum_epoch_window_overrides() -> None:
    custom_quantum_epoch_start = 50
    custom_quantum_epoch_end = 800
    args = plot_figure_2.parse_args(
        [
            "--quantum-epoch-start",
            str(custom_quantum_epoch_start),
            "--quantum-epoch-end",
            str(custom_quantum_epoch_end),
        ]
    )

    assert args.quantum_epoch_start == custom_quantum_epoch_start
    assert args.quantum_epoch_end == custom_quantum_epoch_end
    assert args.quantum_epoch_start < args.quantum_epoch_end


def test_resolve_fixed_run_directory_uses_requested_samples_per_class_suffix(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    custom_samples_per_class = plot_figure_2.DEFAULT_SAMPLES_PER_CLASS + 1
    expected_directory = artifacts_root / f"classical_cnn_spc{custom_samples_per_class}"
    expected_directory.mkdir(parents=True)

    resolved_directory = plot_figure_2._resolve_fixed_run_directory(
        artifacts_root,
        base_name="classical_cnn",
        samples_per_class=custom_samples_per_class,
    )

    assert resolved_directory == expected_directory


def test_resolve_classical_run_directory_uses_size_subdirectory(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"

    resolved_directory = plot_figure_2._resolve_classical_run_directory(
        artifacts_root,
        base_name="classical_cnn",
        scaled_image_size=16,
    )

    assert resolved_directory == artifacts_root / "classical_cnn" / "16on32"


def test_resolve_architecture_sweep_run_directory_uses_q3_nf2_reference_subdirectory(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"

    resolved_directory = plot_figure_2._resolve_architecture_sweep_run_directory(artifacts_root)

    assert resolved_directory == (
        artifacts_root
        / plot_figure_2.DEFAULT_QUANTUM_ARCHITECTURE_SWEEP_DIRECTORY_NAME
        / "fq2_ql3"
    )


def test_load_required_classical_pair_returns_both_model_histories_and_directories(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    seed_order = [0, 1]
    payloads_by_seed = {
        0: {
            "seed": 0,
            "training_history": make_history(test_accuracy=[0.4, 0.6, 0.7]),
            "parameter_stats_line": "params 1",
        },
        1: {
            "seed": 1,
            "training_history": make_history(test_accuracy=[0.5, 0.7, 0.8]),
            "parameter_stats_line": "params 1",
        },
    }
    write_fake_artifact_run(
        artifacts_root / "classical_cnn" / "16on32",
        seed_order=seed_order,
        payloads_by_seed=payloads_by_seed,
    )
    write_fake_artifact_run(
        artifacts_root / "classical_mlp" / "16on32",
        seed_order=seed_order,
        payloads_by_seed=payloads_by_seed,
    )

    cnn_histories, mlp_histories, run_directories = plot_figure_2._load_required_classical_pair(
        artifacts_root=artifacts_root,
        scaled_image_size=16,
    )

    assert len(cnn_histories) == 2
    assert len(mlp_histories) == 2
    assert [label for label, _ in run_directories] == ["classical_cnn 16on32", "classical_mlp 16on32"]


def test_load_required_classical_pair_rejects_when_one_model_is_missing(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    seed_order = [0]
    payload = {
        0: {
            "seed": 0,
            "training_history": make_history(),
            "parameter_stats_line": "params 1",
        }
    }
    write_fake_artifact_run(
        artifacts_root / "classical_cnn" / "16on32",
        seed_order=seed_order,
        payloads_by_seed=payload,
    )

    with pytest.raises(ValueError, match="classical_mlp 16on32"):
        plot_figure_2._load_required_classical_pair(
            artifacts_root=artifacts_root,
            scaled_image_size=16,
        )


def test_load_available_classical_series_configs_accepts_limit_label_offset_override(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    seed_order = [0]
    payload = {
        0: {
            "seed": 0,
            "training_history": make_history(),
            "parameter_stats_line": "params 1",
        }
    }
    write_fake_artifact_run(
        artifacts_root / "classical_cnn" / "32on32",
        seed_order=seed_order,
        payloads_by_seed=payload,
    )
    write_fake_artifact_run(
        artifacts_root / "classical_mlp" / "32on32",
        seed_order=seed_order,
        payloads_by_seed=payload,
    )

    series_configs, _ = plot_figure_2._load_available_classical_series_configs(
        artifacts_root=artifacts_root,
        scaled_image_size=32,
        limit_label_y_offset_points=plot_figure_2.DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS,
    )

    assert len(series_configs) == 2
    assert all(
        config["limit_label_y_offset_points"] == plot_figure_2.DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS
        for config in series_configs
    )
    assert all(config["limit_label_y_offset_points"] > 0 for config in series_configs)


def test_plot_article_figure_2a_uses_grouped_percentiles_and_full_history_limit_guides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cnn_histories = [
        make_history(test_epoch=[10, 20, 30], test_accuracy=[0.2, 0.4, 0.8]),
        make_history(test_epoch=[10, 20, 30], test_accuracy=[0.4, 0.6, 0.9]),
    ]
    mlp_histories = [
        make_history(test_epoch=[10, 20, 30], test_accuracy=[0.1, 0.3, 0.5]),
        make_history(test_epoch=[10, 20, 30], test_accuracy=[0.2, 0.4, 0.7]),
    ]
    captured_calls: list[dict[str, object]] = []

    def fake_plot_temporal_summary(ax, **kwargs):
        captured_calls.append(kwargs)
        (line,) = ax.plot(kwargs["summary"].epoch, kwargs["summary"].mean, label=kwargs["label"])
        return line

    monkeypatch.setattr(plot_figure_2, "plot_temporal_summary", fake_plot_temporal_summary)

    figure = plot_figure_2.plot_article_figure_2a(
        cnn_histories=cnn_histories,
        mlp_histories=mlp_histories,
        epoch_end=20,
        epoch_group_size=20,
    )

    calls_by_label = {call["label"]: call for call in captured_calls}
    assert calls_by_label["CNN"]["summary"].epoch == pytest.approx([15.0])
    assert calls_by_label["CNN"]["summary"].mean == pytest.approx([0.4])
    assert calls_by_label["CNN"]["summary"].lower == pytest.approx([0.35])
    assert calls_by_label["CNN"]["summary"].upper == pytest.approx([0.45])
    assert calls_by_label["CNN"]["limit_value"] == pytest.approx(0.85)
    assert calls_by_label["CNN"]["show_limit_label"] is True
    assert calls_by_label["CNN"]["limit_label_y_offset_points"] == (
        -plot_figure_2.DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS
    )
    assert calls_by_label["Train CNN"].get("limit_value") is None
    assert calls_by_label["MLP"]["limit_value"] == pytest.approx(0.6)
    assert calls_by_label["MLP"]["limit_label_y_offset_points"] == (
        -plot_figure_2.DEFAULT_CLASSICAL_LIMIT_LABEL_Y_OFFSET_POINTS
    )
    assert calls_by_label["Train MLP"].get("limit_value") is None

    plt.close(figure)


def test_plot_article_figure_2b_places_both_limit_labels_slightly_above_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, object]] = []

    def fake_plot_temporal_summary(ax, **kwargs):
        captured_calls.append(kwargs)
        (line,) = ax.plot(kwargs["summary"].epoch, kwargs["summary"].mean, label=kwargs["label"])
        return line

    monkeypatch.setattr(plot_figure_2, "plot_temporal_summary", fake_plot_temporal_summary)

    figure = plot_figure_2.plot_article_figure_2b(
        pcsqcnn_histories=[make_history()],
        pcsqcnn_no_qft_histories=[make_history()],
    )

    calls_by_label = {call["label"]: call for call in captured_calls}
    assert calls_by_label["PCS-QCNN"]["limit_label_y_offset_points"] == (
        plot_figure_2.DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS
    )
    assert calls_by_label["PCS-QCNN"]["limit_label_y_offset_points"] > 0
    assert calls_by_label["PCS-QCNN (no QFT)"]["limit_label_y_offset_points"] == (
        plot_figure_2.DEFAULT_QUANTUM_LIMIT_LABEL_Y_OFFSET_POINTS
    )
    assert calls_by_label["PCS-QCNN (no QFT)"]["limit_label_y_offset_points"] > 0

    plt.close(figure)


def test_plot_article_figure_2a_uses_unsuffixed_labels_for_single_regime() -> None:
    figure = plot_figure_2.plot_article_figure_2a(
        cnn_histories=[make_history()],
        mlp_histories=[make_history()],
    )

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert legend._loc == 7
    assert [text.get_text() for text in legend.texts] == ["CNN", "Train CNN", "MLP", "Train MLP"]

    plt.close(figure)


def test_plot_article_figure_2a_epoch_width_grouping_keeps_early_and_late_test_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_epoch = list(range(10, 2001, 10))
    test_accuracy = [0.4 + 0.0002 * index for index in range(len(test_epoch))]
    histories = [
        make_history(train_epoch=list(range(1, 101)), test_epoch=test_epoch, test_accuracy=test_accuracy),
        make_history(
            train_epoch=list(range(1, 101)),
            test_epoch=test_epoch,
            test_accuracy=[value + 0.01 for value in test_accuracy],
        ),
    ]
    captured_calls: list[dict[str, object]] = []

    def fake_plot_temporal_summary(ax, **kwargs):
        captured_calls.append(kwargs)
        (line,) = ax.plot(kwargs["summary"].epoch, kwargs["summary"].mean, label=kwargs["label"])
        return line

    monkeypatch.setattr(plot_figure_2, "plot_temporal_summary", fake_plot_temporal_summary)

    figure = plot_figure_2.plot_article_figure_2a(
        cnn_histories=histories,
        mlp_histories=histories,
        epoch_group_size=50,
    )

    test_summary = next(call["summary"] for call in captured_calls if call["label"] == "CNN")
    assert test_summary.epoch[0] == pytest.approx(30.0)
    assert test_summary.epoch[-1] == pytest.approx(1980.0)

    plt.close(figure)


def test_main_warns_and_continues_when_some_artifacts_are_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts_root = tmp_path / "artifacts"
    full_mnist_root = tmp_path / "full_mnist"
    output_dir = tmp_path / "figs"
    payloads_by_seed = {
        0: {
            "seed": 0,
            "training_history": make_history(),
            "parameter_stats_line": "params 1",
        }
    }

    write_fake_artifact_run(
        artifacts_root / "classical_cnn" / "16on32",
        seed_order=[0],
        payloads_by_seed=payloads_by_seed,
    )
    write_fake_artifact_run(
        plot_figure_2._resolve_fixed_run_directory(
            artifacts_root,
            base_name="pcsqcnn_no_qft",
            samples_per_class=plot_figure_2.DEFAULT_SAMPLES_PER_CLASS,
        ),
        seed_order=[0],
        payloads_by_seed=payloads_by_seed,
    )

    plot_figure_2.main(
        [
            "--artifacts-root",
            str(artifacts_root),
            "--full-mnist-classical-artifacts-root",
            str(full_mnist_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert "Warning: Skipping classical_mlp 16on32" in captured.err
    assert "Warning: Skipping classical_cnn 32on32" in captured.err
    assert "Warning: Skipping classical_mlp 32on32" in captured.err
    assert "Warning: Skipping pcsqcnn_q3_nf2" in captured.err
    assert "Skipping full_mnist_classical_baselines.pdf" in captured.err

    assert (output_dir / "translated_mnist_classical_baselines.pdf").exists()
    assert not (output_dir / "full_mnist_classical_baselines.pdf").exists()
    assert (output_dir / "translated_mnist_quantum_ablation.pdf").exists()

    assert "classical_cnn 16on32: params 1" in captured.out
    assert "pcsqcnn_no_qft: params 1" in captured.out
