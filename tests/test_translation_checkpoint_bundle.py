import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
from benchmark_pcs_translation import MODELS
from evaluate_summed_translation import file_hash, write_json
from pcs_translation_bundle import export_checkpoints, reconstruct_arrays, resolve_checkpoint
from qcnn.hybrid import PCSQCNN
from qcnn.model_spec import ModelSpec
from qcnn.serialization import load_model_checkpoint, save_model_checkpoint


@pytest.fixture
def checkpoint_bundle(tmp_path):
    source, bundle, weights = (tmp_path / name for name in ("source", "bundle", "weights"))
    runs = {"runs": {}}
    for name, (kind, summed, epoch) in MODELS.items():
        runs["runs"][name] = {}
        kwargs = dict(image_size=8, num_classes=10, feature_qubits=2, quantum_layers=3,
                      multiplexer_kind=kind, reduce_readout_to_feature_distribution=summed)
        model = PCSQCNN(**kwargs)
        spec = ModelSpec(module="qcnn.hybrid", class_name="PCSQCNN", constructor_kwargs=kwargs)
        for seed in range(3):
            path = source / f"{name}_seed{seed}.pt"
            save_model_checkpoint(model, spec, path, training_state={
                "optimizer": {"private_path": "/private/optimizer"}, "rng": torch.ones(20)})
            payload = torch.load(path, weights_only=True)
            sha = file_hash(path)
            runs["runs"][name][str(seed)] = {str(epoch): {"checkpoint_sha256": sha}}
            write_json(bundle / "diagnostics" / f"{name}_seed{seed}.json", {
                "identity": {"checkpoint_sha256": sha, "dataset_sha256": "dataset"},
                "model_spec": payload["model_spec"]})
    write_json(bundle / "runs.json", runs)
    return source, bundle, weights


def test_export_preserves_exact_weights_without_optimizer_or_paths(checkpoint_bundle):
    source, bundle, weights = checkpoint_bundle
    index = export_checkpoints(source, weights, bundle)
    assert sum(len(seeds) for seeds in index["checkpoints"].values()) == 12
    for original in source.glob("*.pt"):
        clean = torch.load(weights / original.name, weights_only=True)
        assert set(clean) == {"checkpoint_type", "format_version", "model_spec", "model_state"}
        before = load_model_checkpoint(original).model.state_dict()
        after = load_model_checkpoint(weights / original.name).model.state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[key], after[key]) for key in before)
    assert "/private/" not in (weights / "INDEX.json").read_text()
    with pytest.raises(ValueError, match="must be empty"):
        export_checkpoints(source, weights, bundle)


def test_export_rejects_other_weights_before_writing_anything(checkpoint_bundle):
    source, bundle, weights = checkpoint_bundle
    path = source / "local_summed_seed2.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="reported run"):
        export_checkpoints(source, weights, bundle)
    assert not weights.exists()


def test_reconstruction_requires_matching_final_weights(checkpoint_bundle):
    source, bundle, weights = checkpoint_bundle
    ref = json.loads((bundle / "diagnostics/general_full_seed0.json").read_text())
    with pytest.raises(FileNotFoundError, match="does not start training"):
        resolve_checkpoint(bundle, weights, "general_full", 0, ref)
    export_checkpoints(source, weights, bundle)
    path = resolve_checkpoint(bundle, weights, "general_full", 0, ref)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint(bundle, weights, "general_full", 0, ref)


@pytest.mark.parametrize("change", ["path", "source", "runs"])
def test_index_cannot_redirect_or_substitute_a_run(checkpoint_bundle, change):
    source, bundle, weights = checkpoint_bundle
    ref = json.loads((bundle / "diagnostics/general_full_seed0.json").read_text())
    index = export_checkpoints(source, weights, bundle)
    entry = index["checkpoints"]["general_full"]["0"]
    if change == "path":
        entry["file"] = "../general_full_seed0.pt"
    elif change == "source":
        entry["source_checkpoint_sha256"] = "another checkpoint"
    else:
        index["runs_sha256"] = "another result bundle"
    write_json(weights / "INDEX.json", index)
    with pytest.raises(ValueError):
        resolve_checkpoint(bundle, weights, "general_full", 0, ref)


def test_reconstruction_prepares_data_and_separates_changed_environments(checkpoint_bundle, monkeypatch):
    source, bundle, weights = checkpoint_bundle
    export_checkpoints(source, weights, bundle)
    ref = json.loads((bundle / "diagnostics/general_full_seed0.json").read_text())
    calls = []
    def evaluate(checkpoint, output, **kwargs):
        calls.append((checkpoint, output, kwargs))
        return {**ref, "arrays_file": "diagnostics.npz"}
    monkeypatch.setattr("evaluate_pcs_translation.evaluate_checkpoint", evaluate)
    cache, data = bundle.parent / "cache", bundle.parent / "data"
    args = dict(data_root=data, device="cpu", weights=weights, cache=cache)
    first, _ = reconstruct_arrays(bundle, "general_full", 0, ref, **args)
    same, _ = reconstruct_arrays(bundle, "general_full", 0, ref, **args)
    monkeypatch.setattr(torch, "__version__", "different-runtime")
    changed, _ = reconstruct_arrays(bundle, "general_full", 0, ref, **args)
    assert first == same and first != changed
    assert all(call[2] == {"data_root": data, "device": "cpu", "download": True} for call in calls)
    assert first.is_relative_to(cache)
    assert not (bundle / "diagnostics/general_full_seed0.npz").exists()


def test_new_training_run_uses_its_own_weights_even_with_release_bundle_present(checkpoint_bundle):
    source, bundle, weights = checkpoint_bundle
    export_checkpoints(source, weights, bundle)
    runs = json.loads((bundle / "runs.json").read_text())
    original = source / "general_full_seed0.pt"
    runs["runs"]["general_full"]["0"]["2000"]["checkpoint"] = str(original)
    write_json(bundle / "runs.json", runs)
    ref = json.loads((bundle / "diagnostics/general_full_seed0.json").read_text())
    assert resolve_checkpoint(bundle, weights, "general_full", 0, ref) == original


def test_published_weights_remain_usable_if_original_path_contains_a_different_run(checkpoint_bundle):
    source, bundle, weights = checkpoint_bundle
    runs = json.loads((bundle / "runs.json").read_text())
    original = source / "general_full_seed0.pt"
    runs["runs"]["general_full"]["0"]["2000"]["checkpoint"] = str(original)
    write_json(bundle / "runs.json", runs)
    export_checkpoints(source, weights, bundle)
    original.write_bytes(original.read_bytes()+b"different run")
    ref = json.loads((bundle / "diagnostics/general_full_seed0.json").read_text())
    assert resolve_checkpoint(bundle, weights, "general_full", 0, ref) == weights / original.name
