from __future__ import annotations

import json
from pathlib import Path

import torch

from qcnn import load_saved_parameter_stats_line, load_saved_training_histories
from qcnn.model import TrainingHistory
from tests.script_loading import load_run_script


export = load_run_script("export_plot_artifacts_module", "export_plot_artifacts.py")


def make_run(directory: Path) -> list[TrainingHistory]:
    directory.mkdir(parents=True)
    histories, runs = [], []
    for seed in (0, 1):
        history = TrainingHistory(
            train_epoch=[1, 2],
            test_epoch=[2],
            train_loss=[2.0, 1.5 + seed],
            test_loss=[1.7],
            train_metrics={"accuracy": [0.2, 0.3]},
            test_metrics={"accuracy": [0.25 + seed / 10]},
        )
        torch.save(
            {"seed": seed, "parameter_stats_line": "q4 c10 Q4 C10", "training_history": history},
            directory / f"result_seed{seed}.pt",
        )
        runs.append({"seed": seed, "result": f"result_seed{seed}.pt", "checkpoint_final": f"checkpoint_final_seed{seed}.pt"})
        histories.append(history)
    (directory / "manifest.json").write_text(json.dumps({"seeds": [0, 1], "runs": runs}))
    return histories


def test_extra_runs_are_exported_with_their_training_histories(tmp_path: Path) -> None:
    source, destination = tmp_path / "full", tmp_path / "published"
    histories = make_run(source / "family" / "run")

    files = export.run_files(source, "family/run")
    exported = [export.export_file(source, destination, relative) for relative in files]

    assert files == ["family/run/manifest.json", "family/run/result_seed0.pt", "family/run/result_seed1.pt"]
    assert exported == ["family/run/manifest.json", "family/run/result_seed0.json", "family/run/result_seed1.json"]
    assert not list(destination.rglob("*.pt"))
    assert load_saved_training_histories(destination / "family" / "run") == histories
    assert load_saved_parameter_stats_line(destination / "family" / "run") == "q4 c10 Q4 C10"


def test_export_selection_uses_the_common_frqi_plot():
    names = {p.name for p in export.data_plot_scripts()}
    assert "plot_frqi_comparison.py" in names
    assert "plot_translated_mnist_baselines.py" in names
    assert not names.intersection({"plot_encoding_swap.py", "plot_translation_generalization.py",
                                   "plot_pcsqcnn_encoding_readout.py", "plot_qcsqcnn_comparison.py"})


def test_compare_renders_reports_changed_and_missing_outputs(tmp_path: Path) -> None:
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for folder in (reference, candidate):
        folder.mkdir()
        (folder / "same.pdf").write_bytes(b"x")
    (reference / "changed.pdf").write_bytes(b"a")
    (candidate / "changed.pdf").write_bytes(b"b")
    (reference / "missing.pdf").write_bytes(b"c")

    assert export.compare_renders(reference, candidate) == ["changed.pdf", "missing.pdf"]


def test_default_export_includes_table_only_runs_and_validation_search(tmp_path, monkeypatch):
    source, destination = tmp_path / "full", tmp_path / "published"
    directories = ("qcsqcnn_translated_mnist/fq2_ql3_sl1",
                   "qcsqcnn_translated_mnist/fq2_ql3_sl8")
    expected = {directory: make_run(source / directory) for directory in directories}
    for directory in directories:
        (source / directory / "checkpoint_final_seed0.pt").write_bytes(b"excluded weights")
    summary = "qcsqcnn_tuning/fq2_ql3_sl8/search_summary.json"
    (source / summary).parent.mkdir(parents=True)
    (source / summary).write_text(json.dumps({"selected": {"learning_rate": 0.003}}))
    # Exercise the default CLI selection; plotting is independent of table evidence.
    monkeypatch.setattr(export, "data_plot_scripts", lambda: [])
    export.main(["--source", str(source), "--destination", str(destination)])
    index = json.loads((destination / "INDEX.json").read_text())
    assert index["files"] == 7
    assert set(index["runs"]) == {*directories, summary}
    for directory in directories:
        assert load_saved_training_histories(destination / directory) == expected[directory]
        assert len(index["runs"][directory]) == 3
    assert (destination / summary).read_bytes() == (source / summary).read_bytes()
    assert not list(destination.rglob("*.pt"))
    assert not list(destination.rglob("checkpoint*"))


def test_portable_export_preserves_values_and_checkpoint_locations(tmp_path):
    import math
    source, destination = tmp_path / 'raw', tmp_path / 'portable'
    source.mkdir()
    payload = {'resolved_config': {'dataset': {'root': '/home/researcher/work/data'},
               'output': {'root': '/home/researcher/work/artifacts'},
               'training': {'init_checkpoint': '/scratch/work/artifacts/first/checkpoint_seed{seed}.pt'}},
               'values': torch.tensor([0.0, math.pi, float('inf')], dtype=torch.float64)}
    torch.save(payload, source / 'result.pt')
    export.export_file(source, destination, 'result.pt')
    from qcnn.json_artifacts import load_payload
    restored = load_payload(destination / 'result.json')
    torch.testing.assert_close(restored['values'], payload['values'], rtol=0, atol=0)
    assert restored['resolved_config']['dataset']['root'] == 'data'
    assert restored['resolved_config']['output']['root'] == 'artifacts'
    assert restored['resolved_config']['training']['init_checkpoint'] == 'artifacts/first/checkpoint_seed{seed}.pt'
    assert '/home/' not in (destination / 'result.json').read_text()


def test_unknown_absolute_export_paths_are_rejected():
    import pytest
    with pytest.raises(ValueError, match='Unrecognized absolute path'):
        export.portable_metadata({'path': '/private/notes/file.txt'})


def test_export_rejects_a_destination_inside_source(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match='separate folder'):
        export.main(['--source', str(tmp_path), '--destination', str(tmp_path / 'nested')])
