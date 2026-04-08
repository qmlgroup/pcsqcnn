import json
from pathlib import Path
import matplotlib
import pytest
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcnn import (
    TrainingHistory,
    focus_metric_history,
    load_saved_parameter_stats_line,
    load_saved_training_histories,
    plot_temporal_summary,
    summarize_accuracy_histories,
    summarize_temporal_samples,
)


def make_history(
    *,
    train_epoch: list[int] | None = None,
    test_epoch: list[int] | None = None,
    train_accuracy: list[float] | None = None,
    test_accuracy: list[float] | None = None,
) -> TrainingHistory:
    resolved_train_epoch = train_epoch or [1, 2, 3]
    resolved_test_epoch = test_epoch or [1, 2, 3]
    resolved_train_accuracy = (
        train_accuracy
        if train_accuracy is not None
        else [0.5 + 0.1 * idx for idx in range(len(resolved_train_epoch))]
    )
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
        train_metrics={"accuracy": resolved_train_accuracy},
        test_metrics={"accuracy": resolved_test_accuracy},
    )


def write_fake_artifact_run(
    run_directory: Path,
    *,
    seed_order: list[int],
    payloads_by_seed: dict[int, dict[str, object]],
    run_entry_order: list[int] | None = None,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    ordered_run_seeds = run_entry_order or seed_order

    runs = []
    for seed in ordered_run_seeds:
        result_name = f"result_seed{seed}.pt"
        torch.save(payloads_by_seed[seed], run_directory / result_name)
        runs.append(
            {
                "seed": seed,
                "result": result_name,
            }
        )

    manifest = {
        "seeds": seed_order,
        "runs": runs,
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_summarize_accuracy_histories_returns_mean_and_population_std() -> None:
    summary = summarize_accuracy_histories(
        [
            make_history(test_accuracy=[0.4, 0.6, 0.8]),
            make_history(test_accuracy=[0.6, 0.8, 1.0]),
        ],
        split="test",
    )

    assert summary.epoch == [1, 2, 3]
    assert summary.mean == pytest.approx([0.5, 0.7, 0.9])
    assert summary.std == pytest.approx([0.1, 0.1, 0.1])


def test_focus_metric_history_trims_to_inclusive_epoch_window() -> None:
    focused = focus_metric_history(
        summary=summarize_accuracy_histories([make_history(test_epoch=[1, 2, 3, 4])], split="test"),
        epoch_start=2,
        epoch_end=3,
    )

    assert focused.epoch == [2, 3]
    assert focused.mean == pytest.approx([0.5, 0.6])
    assert focused.std == pytest.approx([0.0, 0.0])


def test_focus_metric_history_accepts_open_ended_ranges() -> None:
    focused = focus_metric_history(
        summary=summarize_accuracy_histories([make_history(test_epoch=[1, 2, 3, 4])], split="test"),
        epoch_end=2,
    )

    assert focused.epoch == [1, 2]


def test_focus_metric_history_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="epoch_start must be <="):
        focus_metric_history(
            summary=summarize_accuracy_histories([make_history()], split="test"),
            epoch_start=4,
            epoch_end=2,
        )


def test_focus_metric_history_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="does not contain any history points"):
        focus_metric_history(
            summary=summarize_accuracy_histories([make_history()], split="test"),
            epoch_start=10,
            epoch_end=20,
        )


def test_summarize_accuracy_histories_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_accuracy_histories([], split="test")


def test_summarize_accuracy_histories_rejects_mismatched_epoch_grid() -> None:
    with pytest.raises(ValueError, match="epoch grid"):
        summarize_accuracy_histories(
            [
                make_history(test_epoch=[1, 2, 3]),
                make_history(test_epoch=[1, 3, 5]),
            ],
            split="test",
        )


def test_summarize_accuracy_histories_rejects_inconsistent_accuracy_length() -> None:
    with pytest.raises(ValueError, match="must have length 3"):
        TrainingHistory(
            train_epoch=[1, 2, 3],
            test_epoch=[1, 2, 3],
            train_loss=[1.0, 0.8, 0.6],
            test_loss=[1.1, 0.9, 0.7],
            train_metrics={"accuracy": [0.5, 0.6, 0.7]},
            test_metrics={"accuracy": [0.4, 0.5]},
        )


def test_summarize_accuracy_histories_rejects_missing_accuracy() -> None:
    with pytest.raises(ValueError, match="missing accuracy"):
        summarize_accuracy_histories(
            [
                TrainingHistory(
                    train_epoch=[1, 2, 3],
                    test_epoch=[1, 2, 3],
                    train_loss=[1.0, 0.8, 0.6],
                    test_loss=[1.1, 0.9, 0.7],
                    train_metrics={"accuracy": [0.5, 0.6, 0.7]},
                    test_metrics={"gap": [0.4, 0.5, 0.6]},
                )
            ],
            split="test",
        )


def test_summarize_temporal_samples_pools_mean_and_percentiles() -> None:
    summary = summarize_temporal_samples(
        [10],
        [[1.0, 3.0, 5.0, 7.0]],
    )

    assert summary.epoch == [10.0]
    assert summary.mean == pytest.approx([4.0])
    assert summary.lower == pytest.approx([2.5])
    assert summary.upper == pytest.approx([5.5])


def test_summarize_temporal_samples_groups_epoch_width_bins_and_pools_uniformly() -> None:
    summary = summarize_temporal_samples(
        [10, 20, 30, 40],
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ],
        epoch_group_size=20,
    )

    assert summary.epoch == pytest.approx([15.0, 35.0])
    assert summary.mean == pytest.approx([1.5, 5.5])
    assert summary.lower == pytest.approx([0.75, 4.75])
    assert summary.upper == pytest.approx([2.25, 6.25])


def test_summarize_temporal_samples_filters_before_grouping() -> None:
    summary = summarize_temporal_samples(
        [10, 20, 30, 40],
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
        ],
        epoch_start=20,
        epoch_end=35,
        epoch_group_size=20,
    )

    assert summary.epoch == pytest.approx([25.0])
    assert summary.mean == pytest.approx([3.5])
    assert summary.lower == pytest.approx([2.75])
    assert summary.upper == pytest.approx([4.25])


def test_summarize_temporal_samples_places_boundary_epoch_in_next_bin() -> None:
    summary = summarize_temporal_samples(
        [10, 30, 40],
        [
            [1.0],
            [3.0],
            [4.0],
        ],
        epoch_group_size=20,
    )

    assert summary.epoch == pytest.approx([10.0, 35.0])
    assert summary.mean == pytest.approx([1.0, 3.5])


def test_summarize_temporal_samples_skips_empty_epoch_width_bins() -> None:
    summary = summarize_temporal_samples(
        [10, 50, 100],
        [
            [1.0],
            [5.0],
            [10.0],
        ],
        epoch_group_size=20,
    )

    assert summary.epoch == pytest.approx([10.0, 50.0, 100.0])
    assert summary.mean == pytest.approx([1.0, 5.0, 10.0])


def test_summarize_temporal_samples_rejects_invalid_percentiles() -> None:
    with pytest.raises(ValueError, match="Percentiles must satisfy"):
        summarize_temporal_samples(
            [10, 20],
            [[0.0], [1.0]],
            lower_percentile=80.0,
            upper_percentile=20.0,
        )


def test_plot_temporal_summary_supports_line_only_mode() -> None:
    figure, ax = plt.subplots()

    plot_temporal_summary(
        ax,
        summary=summarize_temporal_samples([10, 20], [[0.2, 0.4], [0.6, 0.8]]),
        color="C0",
        show_band=False,
    )

    assert len(ax.lines) == 1
    assert len(ax.collections) == 0

    plt.close(figure)


def test_plot_temporal_summary_can_draw_limit_guide_and_label() -> None:
    figure, ax = plt.subplots()

    plot_temporal_summary(
        ax,
        summary=summarize_temporal_samples([10, 20], [[0.2, 0.4], [0.6, 0.8]]),
        color="C1",
        limit_value=0.75,
        show_limit_label=True,
    )

    assert len(ax.lines) == 2
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "0.7500"

    plt.close(figure)


def test_load_saved_training_histories_uses_manifest_seed_order(tmp_path: Path) -> None:
    run_directory = tmp_path / "classical_cnn"
    write_fake_artifact_run(
        run_directory,
        seed_order=[4, 2],
        run_entry_order=[2, 4],
        payloads_by_seed={
            2: {"seed": 2, "training_history": make_history(test_accuracy=[0.2, 0.3, 0.4])},
            4: {"seed": 4, "training_history": make_history(test_accuracy=[0.8, 0.9, 1.0])},
        },
    )

    histories = load_saved_training_histories(run_directory)

    assert [history.test_metrics["accuracy"][-1] for history in histories] == [1.0, 0.4]


def test_load_saved_training_histories_rejects_manifest_result_seed_mismatch(tmp_path: Path) -> None:
    run_directory = tmp_path / "classical_cnn"
    write_fake_artifact_run(
        run_directory,
        seed_order=[1],
        payloads_by_seed={
            1: {"seed": 9, "training_history": make_history()},
        },
    )

    with pytest.raises(ValueError, match="seed mismatch"):
        load_saved_training_histories(run_directory)


def test_load_saved_parameter_stats_line_uses_manifest_seed_order_and_shared_value(tmp_path: Path) -> None:
    run_directory = tmp_path / "classical_cnn"
    write_fake_artifact_run(
        run_directory,
        seed_order=[4, 2],
        run_entry_order=[2, 4],
        payloads_by_seed={
            2: {
                "seed": 2,
                "training_history": make_history(test_accuracy=[0.2, 0.3, 0.4]),
                "parameter_stats_line": "c160 c4640 c13872 c27712 c650 Q0 C47034",
            },
            4: {
                "seed": 4,
                "training_history": make_history(test_accuracy=[0.8, 0.9, 1.0]),
                "parameter_stats_line": "c160 c4640 c13872 c27712 c650 Q0 C47034",
            },
        },
    )

    assert load_saved_parameter_stats_line(run_directory) == "c160 c4640 c13872 c27712 c650 Q0 C47034"


def test_load_saved_parameter_stats_line_rejects_missing_values(tmp_path: Path) -> None:
    run_directory = tmp_path / "classical_cnn"
    write_fake_artifact_run(
        run_directory,
        seed_order=[1, 2],
        payloads_by_seed={
            1: {"seed": 1, "training_history": make_history()},
            2: {
                "seed": 2,
                "training_history": make_history(),
                "parameter_stats_line": "c160 c4640 c13872 c27712 c650 Q0 C47034",
            },
        },
    )

    with pytest.raises(ValueError, match="missing parameter_stats_line"):
        load_saved_parameter_stats_line(run_directory)


def test_load_saved_parameter_stats_line_rejects_inconsistent_seed_values(tmp_path: Path) -> None:
    run_directory = tmp_path / "classical_cnn"
    write_fake_artifact_run(
        run_directory,
        seed_order=[1, 2],
        payloads_by_seed={
            1: {
                "seed": 1,
                "training_history": make_history(),
                "parameter_stats_line": "c799 c9024 c35532 c1890 Q0 C47245",
            },
            2: {
                "seed": 2,
                "training_history": make_history(),
                "parameter_stats_line": "c160 c4640 c13872 c27712 c650 Q0 C47034",
            },
        },
    )

    with pytest.raises(ValueError, match="do not match across seeds"):
        load_saved_parameter_stats_line(run_directory)
