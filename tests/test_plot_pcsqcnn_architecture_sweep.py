import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from qcnn import TrainingHistory

PLOT_FIGURE_5A_PATH = Path(__file__).resolve().parents[1] / "run" / "plot_pcsqcnn_architecture_sweep.py"
PLOT_FIGURE_5A_SPEC = importlib.util.spec_from_file_location("plot_pcsqcnn_architecture_sweep_module", PLOT_FIGURE_5A_PATH)
if PLOT_FIGURE_5A_SPEC is None or PLOT_FIGURE_5A_SPEC.loader is None:
    raise RuntimeError(f"Could not load plot_pcsqcnn_architecture_sweep.py from {PLOT_FIGURE_5A_PATH}.")
plot_figure_5a = importlib.util.module_from_spec(PLOT_FIGURE_5A_SPEC)
sys.modules[PLOT_FIGURE_5A_SPEC.name] = plot_figure_5a
PLOT_FIGURE_5A_SPEC.loader.exec_module(plot_figure_5a)


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


def test_parse_architecture_run_directory_name_extracts_layer_feature_pair() -> None:
    architecture = plot_figure_5a.parse_architecture_run_directory_name("fq2_ql5")

    assert architecture.feature_qubits == 2
    assert architecture.quantum_layers == 5
    assert architecture.label == r"Q=5, $n_f=2$"


def test_parse_args_defaults_to_project_artifacts_and_output_dirs() -> None:
    args = plot_figure_5a.parse_args([])

    assert args.artifacts_root == plot_figure_5a.DEFAULT_ARTIFACTS_ROOT
    assert args.output_dir == plot_figure_5a.DEFAULT_OUTPUT_DIR
    assert args.epoch_group_size == plot_figure_5a.DEFAULT_EPOCH_GROUP_SIZE


def test_parse_args_accepts_artifacts_output_dir_overrides() -> None:
    args = plot_figure_5a.parse_args(
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
    args = plot_figure_5a.parse_args(
        [
            "--epoch-start",
            "100",
            "--epoch-end",
            "500",
            "--epoch-group-size",
            "4",
            "--lower-percentile",
            "10",
            "--upper-percentile",
            "90",
        ]
    )

    assert args.epoch_start == 100
    assert args.epoch_end == 500
    assert args.epoch_group_size == 4
    assert args.lower_percentile == 10.0
    assert args.upper_percentile == 90.0


def test_build_output_filename_and_sweep_directory_name_use_article_full_readout_paths() -> None:
    assert plot_figure_5a.build_sweep_directory_name() == "pcsqcnn_architecture_sweep_full_readout"
    assert plot_figure_5a.build_output_filename() == "pcsqcnn_architecture_sweep_full_readout.pdf"


def test_reserved_style_palettes_provide_at_least_six_entries() -> None:
    assert len(plot_figure_5a.LAYER_COLOR_CYCLE) >= 6
    assert len(plot_figure_5a.FEATURE_LINESTYLE_CYCLE) >= 6


def test_summarize_architecture_sweep_groups_and_sorts_runs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_architecture_sweep_full_readout"
    test_epoch = [10, 20, 30]

    write_fake_artifact_run(
        sweep_root / "fq2_ql1",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.40, 0.60, 0.70])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.60, 0.80, 0.90])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq1_ql1",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.30, 0.50, 0.60])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.50, 0.70, 0.80])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq3_ql5",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.20, 0.40, 0.50])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.40, 0.60, 0.70])},
        },
    )

    series_by_architecture = plot_figure_5a.summarize_architecture_sweep(sweep_root=sweep_root)

    assert [series.label for series in series_by_architecture] == [
        r"Q=1, $n_f=1$",
        r"Q=1, $n_f=2$",
        r"Q=5, $n_f=3$",
    ]
    assert series_by_architecture[0].summary.epoch == [10, 20, 30]
    assert series_by_architecture[0].summary.mean == pytest.approx([0.40, 0.60, 0.70])
    assert series_by_architecture[0].summary.lower == pytest.approx([0.35, 0.55, 0.65])
    assert series_by_architecture[0].summary.upper == pytest.approx([0.45, 0.65, 0.75])
    assert series_by_architecture[2].summary.mean == pytest.approx([0.30, 0.50, 0.60])


def test_summarize_architecture_sweep_supports_epoch_window_and_grouping(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_architecture_sweep_full_readout"
    write_fake_artifact_run(
        sweep_root / "fq1_ql1",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=[10, 20, 30], test_accuracy=[0.2, 0.4, 0.8])},
            1: {"seed": 1, "training_history": make_history(test_epoch=[10, 20, 30], test_accuracy=[0.4, 0.6, 1.0])},
        },
    )

    series_by_architecture = plot_figure_5a.summarize_architecture_sweep(
        sweep_root=sweep_root,
        epoch_end=20,
        epoch_group_size=20,
    )

    assert len(series_by_architecture) == 1
    assert series_by_architecture[0].summary.epoch == pytest.approx([15.0])
    assert series_by_architecture[0].summary.mean == pytest.approx([0.4])
    assert series_by_architecture[0].summary.lower == pytest.approx([0.35])
    assert series_by_architecture[0].summary.upper == pytest.approx([0.45])


def test_detect_observed_layers_and_features_use_only_valid_saved_runs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "artifacts" / "pcsqcnn_architecture_sweep_full_readout"
    test_epoch = [10, 20, 30]

    write_fake_artifact_run(
        sweep_root / "fq1_ql1",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.20, 0.40, 0.50])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.30, 0.50, 0.60])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq3_ql5",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.40, 0.60, 0.70])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.50, 0.70, 0.80])},
        },
    )
    (sweep_root / "fq2_ql7").mkdir(parents=True, exist_ok=True)

    with pytest.warns(UserWarning):
        series_by_architecture = plot_figure_5a.summarize_architecture_sweep(sweep_root=sweep_root)

    assert plot_figure_5a.detect_observed_layers(series_by_architecture) == [1, 5]
    assert plot_figure_5a.detect_observed_feature_qubits(series_by_architecture) == [1, 3]


def test_resolve_layer_color_uses_value_based_reserved_palette() -> None:
    assert plot_figure_5a.resolve_layer_color(1) == plot_figure_5a.LAYER_COLOR_CYCLE[0]
    assert plot_figure_5a.resolve_layer_color(5) == plot_figure_5a.LAYER_COLOR_CYCLE[4]


def test_resolve_feature_linestyle_uses_value_based_reserved_palette() -> None:
    assert plot_figure_5a.resolve_feature_linestyle(1) == plot_figure_5a.FEATURE_LINESTYLE_CYCLE[0]
    assert plot_figure_5a.resolve_feature_linestyle(3) == plot_figure_5a.FEATURE_LINESTYLE_CYCLE[2]


def test_resolve_architecture_style_combines_layer_color_and_feature_linestyle() -> None:
    style = plot_figure_5a.resolve_architecture_style(
        plot_figure_5a.ArchitectureKey(quantum_layers=5, feature_qubits=3)
    )

    assert style.color == plot_figure_5a.LAYER_COLOR_CYCLE[4]
    assert style.linestyle == plot_figure_5a.FEATURE_LINESTYLE_CYCLE[2]


def test_style_resolution_rejects_values_beyond_reserved_palettes() -> None:
    with pytest.raises(ValueError, match="layer palette only defines"):
        plot_figure_5a.resolve_layer_color(len(plot_figure_5a.LAYER_COLOR_CYCLE) + 1)

    with pytest.raises(ValueError, match="feature palette only defines"):
        plot_figure_5a.resolve_feature_linestyle(len(plot_figure_5a.FEATURE_LINESTYLE_CYCLE) + 1)


def test_legend_entry_builders_only_include_detected_values() -> None:
    layer_entries = plot_figure_5a.build_layer_legend_entries([5, 1, 5])
    feature_entries = plot_figure_5a.build_feature_legend_entries([3, 1, 3])

    assert [entry.label for entry in layer_entries] == ["Q=1", "Q=5"]
    assert [entry.color for entry in layer_entries] == [
        plot_figure_5a.LAYER_COLOR_CYCLE[0],
        plot_figure_5a.LAYER_COLOR_CYCLE[4],
    ]
    assert [entry.label for entry in feature_entries] == [r"$n_f=1$", r"$n_f=3$"]
    assert all(entry.color == "black" for entry in feature_entries)
    assert [entry.linestyle for entry in feature_entries] == [
        plot_figure_5a.FEATURE_LINESTYLE_CYCLE[0],
        plot_figure_5a.FEATURE_LINESTYLE_CYCLE[2],
    ]


def test_plot_article_figure_5a_places_layer_and_feature_legends_in_requested_positions(tmp_path: Path) -> None:
    from matplotlib.legend import Legend

    sweep_root = tmp_path / "artifacts" / "pcsqcnn_architecture_sweep_full_readout"
    test_epoch = [10, 20, 30]

    write_fake_artifact_run(
        sweep_root / "fq1_ql1",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.20, 0.40, 0.50])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.30, 0.50, 0.60])},
        },
    )
    write_fake_artifact_run(
        sweep_root / "fq3_ql5",
        seed_order=[0, 1],
        payloads_by_seed={
            0: {"seed": 0, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.40, 0.60, 0.70])},
            1: {"seed": 1, "training_history": make_history(test_epoch=test_epoch, test_accuracy=[0.50, 0.70, 0.80])},
        },
    )

    figure = plot_figure_5a.plot_article_figure_5a(sweep_root=sweep_root)
    legends = [child for child in figure.axes[0].get_children() if isinstance(child, Legend)]
    legends_by_labels = {
        tuple(text.get_text() for text in legend.texts): legend
        for legend in legends
    }

    assert len(legends) == 2
    assert legends_by_labels[("Q=1", "Q=5")]._loc == 4
    assert legends_by_labels[(r"$n_f=1$", r"$n_f=3$")]._loc == 8

    plot_figure_5a._require_matplotlib().close(figure)
