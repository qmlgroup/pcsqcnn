"""PCS translation maps from 16 direct quantum readouts and the trained head.

For full readout, reconstruct all cyclic translations by the known coarse
spatial permutation. Direct simulations at wrapping shifts independently
validate this acceleration. Feature TV is measured after undoing the coarse
translation; the classifier always receives the original, unaligned readout.
Diagnostics use double precision at the unchanged trained weights. A separate
audit reports the effect of the checkpoint's original floating-point dtype.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qcnn.data import prepare_mnist_splits
from qcnn.serialization import load_model_checkpoint
from evaluate_summed_translation import data_hash, file_hash, outputs, write_json


def summarize_arrays(predictions, labels, cell_readouts):
    """Recompute every plotted value from per-image arrays (row=y, column=x)."""
    origin = predictions[0, 0]
    accuracy = (predictions == labels).mean(axis=-1)
    changed = (predictions != origin).mean(axis=-1)
    cell = cell_readouts.shape[0]
    feature_tv = np.empty((cell, cell))
    base = cell_readouts[0, 0].reshape(len(labels), -1).astype(np.float64)
    for y in range(cell):
        for x in range(cell):
            q = cell_readouts[y, x].reshape(len(labels), -1).astype(np.float64)
            feature_tv[y, x] = np.abs(q-base).sum(axis=1).mean()/2
    return {"accuracy": accuracy.tolist(), "changed_labels": changed.tolist(),
            "aligned_feature_tv": feature_tv.tolist()}


@torch.no_grad()
def translation_arrays(model, images, labels, *, batch_size=1600, validation_images=200):
    cell = 2 ** (model.quantum_layers-1)
    canvas = images.shape[1]
    summed = model.reduce_readout_to_feature_distribution
    if canvas % cell:
        raise ValueError("Canvas must be divisible by the pooling cell")
    q = torch.stack([torch.stack([outputs(model, images, (y, x), batch_size)[0]
                                 for x in range(cell)]) for y in range(cell)])
    if not torch.isfinite(q).all():
        raise ValueError("Non-finite quantum readout")
    normalization_error = float((q.flatten(3).sum(-1)-1).abs().max())
    probability_tolerance = 1e-10 if q.dtype == torch.float64 else 1e-5
    logit_tolerance = 1e-8 if q.dtype == torch.float64 else 1e-3
    if normalization_error > probability_tolerance or float(q.min()) < -probability_tolerance:
        raise ValueError(f"Invalid probabilities: normalization error {normalization_error}")

    extent = cell if summed else canvas
    predictions = np.empty((extent, extent, len(images)), dtype=np.uint8)
    for y in range(cell):
        for x in range(cell):
            base = q[y, x].to(images.device)
            for dy in range(0, extent, cell):
                for dx in range(0, extent, cell):
                    moved = torch.roll(base, (dy//cell, dx//cell), (1, 2))
                    logits = model.classify_readout_histogram(moved)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Non-finite classifier output")
                    predictions[y+dy, x+dx] = logits.argmax(1).cpu().numpy()

    # These are fresh quantum simulations, not reconstructed outputs.
    # Cover every residue, both axes, and joint boundary/corner crossings.
    coarse = ((cell, 0), (0, cell), (3*cell, 0), (0, 3*cell),
              (3*cell, 3*cell), (4*cell, 5*cell))
    n = min(validation_images, len(images))
    pairs = []
    for y in range(cell):
        for x in range(cell):
            for dy, dx in coarse:
                shift = ((y+dy) % canvas, (x+dx) % canvas)
                direct_q, direct_z = outputs(model, images[:n], shift, batch_size)
                expected_q = torch.roll(q[y, x, :n], (dy//cell, dx//cell), (1, 2))
                expected_z = model.classify_readout_histogram(expected_q.to(images.device)).cpu()
                pairs.append({"shift_yx": list(shift), "residue_yx": [y, x],
                    "readout_l1_max": float((direct_q-expected_q).abs().flatten(1).sum(1).max()),
                    "logit_linf_max": float((direct_z-expected_z).abs().max()),
                    "changed_labels": int((direct_z.argmax(1) != expected_z.argmax(1)).sum())})
    validation = {"n_images": n, "pairs": pairs,
                  "readout_l1_max": max(p["readout_l1_max"] for p in pairs),
                  "logit_linf_max": max(p["logit_linf_max"] for p in pairs),
                  "changed_labels_max": max(p["changed_labels"] for p in pairs),
                  "normalization_linf_max": normalization_error,
                  "probability_tolerance": probability_tolerance,
                  "logit_tolerance": logit_tolerance}
    validation["passed"] = (validation["readout_l1_max"] < probability_tolerance
                            and validation["logit_linf_max"] < logit_tolerance
                            and validation["changed_labels_max"] == 0)
    if not validation["passed"]:
        raise ValueError(f"Direct validation of reconstruction failed: {validation}")
    arrays = {"predictions": predictions, "labels": labels.cpu().numpy(),
              "cell_readouts": q.numpy()}
    return summarize_arrays(**arrays), arrays, validation


@torch.no_grad()
def precision_audit(native_model, diagnostic_model, images, labels, arrays, *, batch_size=1600):
    """Compare the original-dtype and diagnostic grids without renormalization."""
    cell = arrays["cell_readouts"].shape[0]
    extent = arrays["predictions"].shape[0]
    q_error = z_error = norm_error = 0.
    changes = native_correct = diagnostic_correct = 0
    native_dtype = next(native_model.parameters()).dtype
    diagnostic_dtype = next(diagnostic_model.parameters()).dtype
    for y in range(cell):
        for x in range(cell):
            p, z = outputs(native_model, images.to(dtype=native_dtype), (y, x), batch_size)
            reference = torch.from_numpy(arrays["cell_readouts"][y, x])
            q_error = max(q_error, float((p.double()-reference.double()).abs().flatten(1).sum(1).max()))
            norm_error = max(norm_error, float((p.flatten(1).sum(1)-1).abs().max()))
            reference_z = diagnostic_model.classify_readout_histogram(
                reference.to(device=images.device, dtype=diagnostic_dtype)).cpu()
            z_error = max(z_error, float((z.double()-reference_z.double()).abs().max()))
            p = p.to(images.device)
            for dy in range(0, extent, cell):
                for dx in range(0, extent, cell):
                    moved = torch.roll(p, (dy//cell, dx//cell), (1, 2))
                    logits = native_model.classify_readout_histogram(moved)
                    if not torch.isfinite(logits).all():
                        raise ValueError("Non-finite output in original-precision audit")
                    pred = logits.argmax(1).cpu().numpy()
                    reference_pred = arrays["predictions"][y+dy, x+dx]
                    changes += int(np.sum(pred != reference_pred))
                    native_correct += int(np.sum(pred == labels.cpu().numpy()))
                    diagnostic_correct += int(np.sum(reference_pred == labels.cpu().numpy()))
    count = extent*extent*len(images)
    return {"original_dtype": str(native_dtype), "diagnostic_dtype": str(diagnostic_dtype),
            "cell_readout_l1_max": q_error, "cell_logit_linf_max": z_error,
            "original_normalization_linf_max": norm_error,
            "grid_predictions": count, "changed_predictions": changes,
            "changed_prediction_fraction": changes/count,
            "original_grid_accuracy": native_correct/count,
            "diagnostic_grid_accuracy": diagnostic_correct/count,
            "method": "same 16 direct readouts and known coarse permutations in both precisions; no renormalization"}


def evaluate_checkpoint(checkpoint, output, *, data_root, device="cuda", batch_size=1600, download=False):
    checkpoint, output = Path(checkpoint), Path(output)
    torch.set_float32_matmul_precision("highest")
    loaded = load_model_checkpoint(checkpoint, map_location=device)
    original_dtype = str(next(loaded.model.parameters()).dtype)
    model = loaded.model.eval().double()
    if (model.image_size != 32 or model.quantum_layers != 3 or model.feature_qubits != 2
            or model.encoding != "frqi" or model.shot_budget is not None):
        raise ValueError("Expected exact FRQI PCS, 32x32, three layers, two feature qubits")
    test = prepare_mnist_splits(root=data_root, samples_per_class=1000, image_size=32,
                               scaled_image_size=16, max_offset=0, seed=0, download=download).test
    source_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in
                     ("evaluate_pcs_translation.py", "evaluate_summed_translation.py")}
    source_root = Path(__file__).resolve().parents[1]
    source_hashes.update({str(p.relative_to(source_root)): file_hash(p)
                         for p in sorted((source_root/"src/qcnn").rglob("*.py"))})
    identity = {"checkpoint_sha256": file_hash(checkpoint),
                "dataset_sha256": data_hash(test.images, test.labels),
                "source_sha256": source_hashes, "matmul_precision": "highest",
                "evaluation_dtype": "torch.float64", "checkpoint_dtype": original_dtype,
                "torch": str(torch.__version__), "cuda": torch.version.cuda}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Existing evaluation is incompatible: {output}; use a new output root")
        cached_arrays = output.parent/saved["arrays_file"]
        if cached_arrays.exists():
            if file_hash(cached_arrays) != saved["arrays_sha256"]:
                raise ValueError(f"Evaluation arrays were modified: {output}")
            return saved
        # Arrays are disposable inference products; regenerate a missing cache.
    metrics, arrays, validation = translation_arrays(
        model, test.images.to(device=device, dtype=torch.float64), test.labels, batch_size=batch_size)
    native = load_model_checkpoint(checkpoint, map_location=device).model.eval()
    numerical_audit = precision_audit(native, model, test.images.to(device), test.labels, arrays,
                                    batch_size=batch_size)
    del native
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix(".npz")
    temporary = array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(array_path)
    result = {"identity": identity, "checkpoint": str(checkpoint.resolve()),
              "model_spec": {"module": loaded.model_spec.module,
                             "class_name": loaded.model_spec.class_name,
                             "constructor_kwargs": dict(loaded.model_spec.constructor_kwargs)},
              "parameters": sum(p.numel() for p in model.parameters()),
              "summed": model.reduce_readout_to_feature_distribution,
              "n_images": len(test), "cell": 4, "canvas": 32,
              "coordinate_order": "row=delta_y, column=delta_x; torch image axes 1,2",
              "method": "16 direct quantum readouts; known coarse permutations; original trained head",
              "feature_metric": "mean per-image TV after undoing the coarse spatial permutation",
              "metrics": metrics, "direct_validation": validation,
              "precision_audit": numerical_audit,
              "arrays_file": array_path.name, "arrays_sha256": file_hash(array_path)}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1600)
    torch.set_num_threads(4)
    result = evaluate_checkpoint(**vars(parser.parse_args()))
    print(json.dumps({"identity": result["identity"], "direct_validation_passed": True}))


if __name__ == "__main__":
    main()
