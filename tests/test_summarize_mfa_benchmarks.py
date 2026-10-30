import json
import os
from pathlib import Path

import torch

from qcnn.model import TrainingHistory
from tests.script_loading import load_run_script


summary_script = load_run_script("summarize_mfa_benchmarks_module", "summarize_mfa_benchmarks.py")


def test_mfa_summary_reads_final_accuracy_and_learning_curves(tmp_path: Path) -> None:
    run = tmp_path / "full_translated_seed0"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"seeds": [0]}), encoding="utf-8")
    (run / "checkpoint_final_seed0.pt").touch()
    (run / "runtime.json").write_text(json.dumps({"wall_seconds": 123.0}), encoding="utf-8")
    history = TrainingHistory(
        train_epoch=list(range(1, 2001)),
        test_epoch=[2000],
        train_loss=[2.0] * 2000,
        test_loss=[2.1],
        train_metrics={"accuracy": [0.4] * 2000},
        test_metrics={"accuracy": [0.35]},
    )
    torch.save(
        {
            "training_history": history,
            "device": "cuda",
            "loss_name": "MFAProbabilityNLLLoss",
            "final_summaries": {
                "train": {"loss": 2.0, "metrics": {"accuracy": 0.4}},
                "test": {"loss": 2.1, "metrics": {"accuracy": 0.35}},
            },
            "parameter_stats_line": "q105 Q105 C0",
            "data_fingerprints": {"train": "train-hash", "test": "test-hash"},
        },
        run / "result_seed0.pt",
    )

    summary, curves = summary_script.summarize(tmp_path, require_complete=False)
    translated = summary["regimes"]["translated"]
    assert translated["test_accuracy"]["mean"] == 0.35
    assert translated["test_accuracy"]["count"] == 1
    assert translated["runs"][0]["training_device"] == "cuda"
    assert translated["runs"][0]["run_wall_seconds"] == 123.0
    assert translated["runs"][0]["run_wall_seconds_estimated"] is False
    assert translated["runs"][0]["run_wall_seconds_current_segment"] == 123.0
    assert translated["runs"][0]["run_wall_seconds_prior_estimate"] is None
    assert len(curves) == 2001
    assert curves[-1] == {
        "regime": "translated", "seed": 0, "phase": "test", "epoch": 2000,
        "loss": 2.1, "accuracy": 0.35,
    }

    (run / "prior_runtime.json").write_text(
        json.dumps({"completed_epochs": 50, "wall_seconds_estimate": 600.0}),
        encoding="utf-8",
    )
    resumed_summary, _ = summary_script.summarize(tmp_path, require_complete=False)
    resumed_run = resumed_summary["regimes"]["translated"]["runs"][0]
    assert resumed_run["run_wall_seconds"] == 723.0
    assert resumed_run["run_wall_seconds_estimated"] is True
    assert resumed_run["run_wall_seconds_current_segment"] == 123.0
    assert resumed_run["run_wall_seconds_prior_estimate"] == 600.0

    # The orchestrator was paused after seed 0 finished, so its raw wall time
    # includes time spent waiting for unrelated seeds.
    raw_runtime = {"start_unix_time": 900.0, "end_unix_time": 1500.0,
                   "wall_seconds": 600.0}
    (run / "runtime.json").write_text(json.dumps(raw_runtime), encoding="utf-8")
    os.utime(run / "manifest.json", (1000.0, 1000.0))
    (run / "runtime_end_from_manifest.json").write_text(
        json.dumps({"method": "manifest_mtime_after_paused_orchestrator"}),
        encoding="utf-8",
    )
    corrected_summary, _ = summary_script.summarize(tmp_path, require_complete=False)
    corrected = corrected_summary["regimes"]["translated"]["runs"][0]
    assert corrected["run_wall_seconds_current_segment"] == 100.0
    assert corrected["run_wall_seconds"] == 700.0
    assert corrected["run_wall_seconds_pause_adjusted"] is True
    assert corrected["run_wall_seconds_original_orchestrator"] == 600.0
    assert json.loads((run / "runtime_orchestrator_raw.json").read_text()) == raw_runtime
    assert json.loads((run / "runtime.json").read_text())["wall_seconds"] == 100.0
    repeated_summary, _ = summary_script.summarize(tmp_path, require_complete=False)
    assert repeated_summary == corrected_summary
