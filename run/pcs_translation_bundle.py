"""Compact final checkpoints and automatic reconstruction of translation arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from benchmark_pcs_translation import MODELS, read_plain
from evaluate_summed_translation import file_hash, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results/pcs_translation_2026-10-25"
DEFAULT_WEIGHTS = ROOT / "artifacts/pcs_translation_weights"
DEFAULT_CACHE = ROOT / "artifacts/pcs_translation_recomputed"


def canonical(value):
    return json.dumps(value, sort_keys=True)


def export_checkpoints(source, destination, bundle=DEFAULT_BUNDLE):
    """Export only architecture and tensors, checking the original run hashes.

    Source files are named ``<model>_seed<seed>.pt``. Optimizer state, paths,
    RNG state and training metadata are not copied into the inference bundle.
    """
    source, destination, bundle = Path(source), Path(destination), Path(bundle)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Checkpoint destination must be empty: {destination}")
    runs = read_plain(bundle / "runs.json")["runs"]
    prepared = []
    for name, (_, _, epoch) in MODELS.items():
        for seed in (0, 1, 2):
            entry = runs[name][str(seed)][str(epoch)]
            reference = read_plain(bundle / "diagnostics" / f"{name}_seed{seed}.json")
            filename = f"{name}_seed{seed}.pt"
            original = source / filename
            expected = entry["checkpoint_sha256"]
            if (file_hash(original) != expected
                    or reference["identity"]["checkpoint_sha256"] != expected):
                raise ValueError(f"Checkpoint does not match the reported run: {filename}")
            payload = torch.load(original, map_location="cpu", weights_only=True)
            spec = payload["model_spec"]
            if (spec["module"] != "qcnn.hybrid" or spec["class_name"] != "PCSQCNN"
                    or canonical(spec) != canonical(reference["model_spec"])):
                raise ValueError(f"Checkpoint architecture mismatch: {filename}")
            clean = {key: payload[key] for key in
                     ("checkpoint_type", "format_version", "model_spec", "model_state")}
            prepared.append((name, seed, epoch, filename, expected, clean))
    destination.mkdir(parents=True, exist_ok=True)
    index = {"format_version": 1, "runs_sha256": file_hash(bundle / "runs.json"),
             "checkpoints": {}}
    for name, seed, epoch, filename, original_hash, clean in prepared:
        target = destination / filename
        torch.save(clean, target)
        restored = torch.load(target, map_location="cpu", weights_only=True)
        if any(not torch.equal(tensor, restored["model_state"][key])
               for key, tensor in clean["model_state"].items()):
            raise ValueError(f"Checkpoint tensors changed during export: {filename}")
        index["checkpoints"].setdefault(name, {})[str(seed)] = {
            "file": filename, "sha256": file_hash(target),
            "source_checkpoint_sha256": original_hash, "epoch": epoch,
            "tensor_bytes": sum(v.numel() * v.element_size() for v in clean["model_state"].values())}
    write_json(destination / "INDEX.json", index)
    return index


def resolve_checkpoint(bundle, weights, name, seed, reference):
    bundle, weights = Path(bundle), Path(weights)
    index_path = weights / "INDEX.json"
    runs = read_plain(bundle / "runs.json")["runs"]
    original = runs[name][str(seed)][str(MODELS[name][2])].get("checkpoint")
    original_mismatch = False
    if original:
        path = Path(original)
        path = path if path.is_absolute() else ROOT / path
        if path.is_file():
            if file_hash(path) == reference["identity"]["checkpoint_sha256"]:
                return path
            original_mismatch = True
    if not index_path.is_file():
        if original_mismatch:
            raise ValueError(f"Original checkpoint checksum mismatch: {path}")
        raise FileNotFoundError(
            f"Missing final weights at {weights}. Extract the supplied "
            "pcsqcnn-plot-data.zip at the repository root; it includes the weights. "
            "Reproduction does not start training automatically.")
    index = read_plain(index_path)
    if index.get("format_version") != 1 or index["runs_sha256"] != file_hash(bundle / "runs.json"):
        raise ValueError("Checkpoint index does not match the result bundle")
    entry = index["checkpoints"][name][str(seed)]
    filename = entry["file"]
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Checkpoint paths must be .pt basenames")
    if (entry["epoch"] != MODELS[name][2]
            or entry["source_checkpoint_sha256"] != reference["identity"]["checkpoint_sha256"]):
        raise ValueError("Final weights belong to a different reported checkpoint")
    path = weights / filename
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return path


def reconstruct_arrays(bundle, name, seed, reference, *, data_root, device="auto",
                       weights=DEFAULT_WEIGHTS, cache=DEFAULT_CACHE):
    """Re-evaluate fixed weights, retaining generated arrays in an ignored cache."""
    from evaluate_pcs_translation import evaluate_checkpoint

    checkpoint = resolve_checkpoint(bundle, weights, name, seed, reference)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or auto")
    sources = sorted((ROOT / "src/qcnn").rglob("*.py"))
    sources += [Path(__file__), Path(__file__).with_name("evaluate_pcs_translation.py"),
                Path(__file__).with_name("evaluate_summed_translation.py")]
    fingerprint = {
        "checkpoint": file_hash(checkpoint), "dataset": reference["identity"]["dataset_sha256"],
        "source": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": device,
        "accelerator": torch.cuda.get_device_name() if device == "cuda" else "cpu"}
    key = hashlib.sha256(canonical(fingerprint).encode()).hexdigest()[:24]
    output = Path(cache) / f"{name}_seed{seed}" / key / "diagnostics.json"
    print(f"Checking fixed weights: {name}, seed {seed} ({device})", flush=True)
    result = evaluate_checkpoint(checkpoint, output, data_root=data_root, device=device, download=True)
    if (result["identity"]["dataset_sha256"] != reference["identity"]["dataset_sha256"]
            or canonical(result["model_spec"]) != canonical(reference["model_spec"])):
        raise ValueError("Recomputed diagnostics use different data or architecture")
    return output.parent / result["arrays_file"], result
