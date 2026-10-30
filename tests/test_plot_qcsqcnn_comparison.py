from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from qcnn import TrainingHistory
from tests.script_loading import load_run_script


plot_qcs = load_run_script("plot_qcsqcnn_comparison_module", "plot_qcsqcnn_comparison.py")


def make_history(final_test_accuracy: float, *, epochs: int = 100) -> TrainingHistory:
    test_epoch = list(range(10, epochs + 1, 10))
    return TrainingHistory(
        train_epoch=list(range(1, epochs + 1)),
        test_epoch=test_epoch,
        train_loss=[1.0] * epochs,
        test_loss=[1.0] * len(test_epoch),
        train_metrics={"accuracy": [min(0.99, 0.1 + 0.008 * index) for index in range(epochs)]},
        test_metrics={
            "accuracy": [
                final_test_accuracy * (index + 1) / len(test_epoch) for index in range(len(test_epoch))
            ]
        },
    )


def write_fake_run(run_directory: Path, *, final_accuracies: list[float], stats_line: str) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed, accuracy in enumerate(final_accuracies):
        torch.save(
            {"seed": seed, "training_history": make_history(accuracy), "parameter_stats_line": stats_line},
            run_directory / f"result_seed{seed}.pt",
        )
        runs.append({"seed": seed, "result": f"result_seed{seed}.pt"})
    (run_directory / "manifest.json").write_text(
        json.dumps({"seeds": list(range(len(final_accuracies))), "runs": runs}),
        encoding="utf-8",
    )


def write_fake_artifacts(root: Path) -> None:
    write_fake_run(
        root / "pcsqcnn_architecture_sweep_full_readout/fq2_ql3",
        final_accuracies=[0.76, 0.75, 0.77],
        stats_line="q16384 q16384 q4096 c2570 Q36864 C2570",
    )
    write_fake_run(
        root / "pcsqcnn_architecture_sweep_full_readout/fq1_ql1",
        final_accuracies=[0.50, 0.52, 0.51],
        stats_line="q4096 c20490 Q4096 C20490",
    )
    write_fake_run(
        root / "qcsqcnn_translated_mnist/fq2_ql3_sl8",
        final_accuracies=[0.55, 0.56, 0.54],
        stats_line="q4096 q16384 q16384 c2570 Q36864 C2570",
    )
    write_fake_run(
        root / "henderson_qcnn_translated_mnist/ch12_ql3",
        final_accuracies=[0.66, 0.68, 0.70],
        stats_line="q3456 q3456 q3072 c30730 Q9984 C30730",
    )
    write_fake_run(
        root / "hur_qcnn_translated_mnist/su4_l1",
        final_accuracies=[0.30, 0.32, 0.31],
        stats_line="q15 q2 q15 c330 Q32 C330",
    )
    write_fake_run(
        root / "hur_qcnn_translated_mnist/su4_l16",
        final_accuracies=[0.40, 0.42, 0.41],
        stats_line="q240 q2 q240 c330 Q482 C330",
    )
    write_fake_run(
        root / "hur_qcnn_translated_mnist/ch112_su4_l1",
        final_accuracies=[0.50, 0.52, 0.51],
        stats_line="q1680 q224 q1680 c35850 Q3584 C35850",
    )
    write_fake_run(
        root / "pcsqcnn_no_qft_spc1000",
        final_accuracies=[0.41, 0.40, 0.42],
        stats_line="q16384 q16384 q4096 c2570 Q36864 C2570",
    )


def test_parse_parameter_totals_reads_quantum_and_classical_totals() -> None:
    assert plot_qcs.parse_parameter_totals("q3456 q3456 q3072 c30730 Q9984 C30730") == (9984, 30730)
    assert plot_qcs.parse_parameter_totals("c160 c650 Q0 C47034") == (0, 47034)
    with pytest.raises(ValueError):
        plot_qcs.parse_parameter_totals("not a stats line")


def test_main_renders_available_figures_and_tables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_dir = tmp_path / "figs"
    write_fake_artifacts(artifacts_root)

    plot_qcs.main(["--artifacts-root", str(artifacts_root), "--output-dir", str(output_dir), "--epoch-group-size", "20"])

    assert (output_dir / plot_qcs.COMPARISON_FILENAME).is_file()
    assert (output_dir / plot_qcs.VARIANTS_FILENAME).is_file()
    assert (output_dir / plot_qcs.HENDERSON_FILENAME).is_file()
    assert (output_dir / plot_qcs.HUR_FILENAME).is_file()
    assert (output_dir / plot_qcs.HUR_VARIANTS_FILENAME).is_file()
    assert (output_dir / plot_qcs.HUR_FILTERS_FILENAME).is_file()
    assert (output_dir / plot_qcs.PARAMETERS_FILENAME).is_file()
    markdown = (output_dir / f"{plot_qcs.TABLE_STEM}.md").read_text(encoding="utf-8")
    latex = (output_dir / f"{plot_qcs.TABLE_STEM}.tex").read_text(encoding="utf-8")
    assert "| PCS-QCNN | 36,864 | 2,570 | 39,434 | 76.00 ± 1.00 | 76.00, 75.00, 77.00 | 41.80 ± 0.55 |" in markdown
    assert "| QCS-QCNN | 36,864 | 2,570 | 39,434 | 55.00 ± 1.00 | 55.00, 56.00, 54.00 | 30.25 ± 0.55 |" in markdown
    assert "| Henderson-QCNN | 9,984 | 30,730 | 40,714 | 68.00 ± 2.00 |" in markdown
    assert "| Hur-QCNN | 32 | 330 | 362 | 31.00 ± 1.00 |" in markdown
    assert "| Hur-QCNN l=16 | 482 | 330 | 812 | 41.00 ± 1.00 |" in markdown
    assert "| Hur-QCNN x112 (parameter-matched) | 3,584 | 35,850 | 39,434 | 51.00 ± 1.00 |" in markdown
    assert "| RBC-QCNN |" in markdown
    assert "Classical CNN" not in markdown
    assert r"QCS-QCNN & 36{,}864 & 2{,}570 & 39{,}434 & $55.00 \pm 1.00$ \\" in latex
    captured = capsys.readouterr()
    assert "Skipping Classical CNN" in captured.err


def test_main_skips_panels_without_their_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_dir = tmp_path / "figs"
    write_fake_run(
        artifacts_root / "pcsqcnn_architecture_sweep_full_readout/fq2_ql3",
        final_accuracies=[0.76, 0.75, 0.77],
        stats_line="q16384 q16384 q4096 c2570 Q36864 C2570",
    )

    plot_qcs.main(["--artifacts-root", str(artifacts_root), "--output-dir", str(output_dir)])

    assert not (output_dir / plot_qcs.COMPARISON_FILENAME).exists()
    assert not (output_dir / plot_qcs.VARIANTS_FILENAME).exists()
    assert not (output_dir / plot_qcs.HENDERSON_FILENAME).exists()
    assert not (output_dir / plot_qcs.HUR_FILENAME).exists()
    assert (output_dir / plot_qcs.PARAMETERS_FILENAME).is_file()
    assert "Skipping translated_mnist_qcs_comparison.pdf: none of ['QCS-QCNN', 'QCS-QCNN (tuned)'] is available." in capsys.readouterr().err


def test_main_renders_tuning_curves_from_a_search_summary(tmp_path: Path) -> None:
    import math

    artifacts_root = tmp_path / "artifacts"
    output_dir = tmp_path / "figs"
    write_fake_artifacts(artifacts_root)
    names = [f"lr{rate:g}_init{scale:.4g}" for rate in (1e-3, 3e-2) for scale in (0.05, 2 * math.pi)]
    summary = {
        "learning_rates": [1e-3, 3e-2],
        "init_scales": [0.05, 2 * math.pi],
        "rung_epochs": [20, 40],
        "selected": {"name": names[0]},
        "histories": {
            name: {"epoch": [10, 20, 30, 40], "validation_accuracy": [0.1, 0.2, 0.3, 0.1 * index]}
            for index, name in enumerate(names)
        },
    }
    for relative_summary, _ in plot_qcs.TUNING_SEARCHES:
        summary_path = artifacts_root / relative_summary
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    plot_qcs.main(["--artifacts-root", str(artifacts_root), "--output-dir", str(output_dir)])

    assert (output_dir / plot_qcs.TUNING_FILENAME).is_file()
    assert (output_dir / plot_qcs.HUR_TUNING_FILENAME).is_file()
