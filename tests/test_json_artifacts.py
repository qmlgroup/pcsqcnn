from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from qcnn import load_saved_parameter_stats_line, load_saved_training_histories
from qcnn.json_artifacts import from_json_data, load_payload, resolve_payload_path, save_json_payload, to_json_data
from qcnn.model import TrainingHistory


def make_history() -> TrainingHistory:
    return TrainingHistory(
        train_epoch=[1, 2],
        test_epoch=[2],
        train_loss=[2.3, 1.1],
        test_loss=[1.2],
        train_metrics={"accuracy": [0.1, 0.5]},
        test_metrics={"accuracy": [0.4]},
    )


def test_round_trip_keeps_tensors_dataclasses_and_containers_exact() -> None:
    payload = {
        "float32": torch.rand(3, 4) * 1e-3,
        "float64": torch.rand(2, dtype=torch.float64),
        "int64": torch.arange(6).reshape(2, 3),
        "bool": torch.tensor([True, False]),
        "scalar": torch.tensor(0.1, dtype=torch.float32),
        "empty": torch.zeros(0, 5),
        "history": make_history(),
        "pair": (1, "a"),
        "int_keys": {(0, 1): 2.5, 3: None},
        "tag_like": {"__tensor__": "plain string"},
        "special": [math.inf, -math.inf],
    }

    decoded = from_json_data(json.loads(json.dumps(to_json_data(payload))))

    for key in ("float32", "float64", "int64", "bool", "scalar", "empty"):
        assert decoded[key].dtype == payload[key].dtype and decoded[key].shape == payload[key].shape
        assert torch.equal(decoded[key], payload[key]), key
    assert decoded["history"] == payload["history"]
    assert decoded["pair"] == (1, "a")
    assert decoded["int_keys"] == {(0, 1): 2.5, 3: None}
    assert decoded["tag_like"] == {"__tensor__": "plain string"}
    assert decoded["special"] == [math.inf, -math.inf]
    nan = from_json_data(json.loads(json.dumps(to_json_data(torch.tensor([math.nan])))))
    assert torch.isnan(nan).all()


def test_load_payload_falls_back_to_the_json_export(tmp_path: Path) -> None:
    payload = {"runs": {"a": torch.ones(2, 2)}}
    torch.save(payload, tmp_path / "saved.pt")
    save_json_payload(payload, tmp_path / "exported.json")

    assert torch.equal(load_payload(tmp_path / "saved.pt")["runs"]["a"], torch.ones(2, 2))
    assert resolve_payload_path(tmp_path / "exported.pt") == tmp_path / "exported.json"
    assert torch.equal(load_payload(tmp_path / "exported.pt")["runs"]["a"], torch.ones(2, 2))


def test_saved_run_loaders_read_json_result_exports(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    runs = []
    for seed in (0, 1):
        save_json_payload(
            {"seed": seed, "parameter_stats_line": "q4 c10 Q4 C10", "training_history": make_history()},
            run / f"result_seed{seed}.json",
        )
        runs.append({"seed": seed, "result": f"result_seed{seed}.pt", "checkpoint_final": f"checkpoint_final_seed{seed}.pt"})
    (run / "manifest.json").write_text(json.dumps({"seeds": [0, 1], "runs": runs}))

    assert load_saved_training_histories(run) == [make_history(), make_history()]
    assert load_saved_parameter_stats_line(run) == "q4 c10 Q4 C10"
