"""Measure PCS readout symmetry and accuracy, including periodic boundaries.

All shifts are evaluated explicitly: no prediction is filled in using the
periodicity being tested. The 4x4 cell, axis scans and corner shifts use all
10,000 test digits; per-image probability/logit residuals use the first 200.
Saved predictions allow agreement statistics to be checked independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qcnn.data import prepare_mnist_splits
from qcnn.serialization import load_model_checkpoint


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_hash(images: torch.Tensor, labels: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in (images, labels):
        value = value.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# Cached Pauli/phase tensors must remain usable by later training.
# inference_mode would poison those caches when diagnostics run first.
@torch.no_grad()
def outputs(model, images: torch.Tensor, shift: tuple[int, int], batch_size: int):
    readouts, logits = [], []
    for batch in images.split(batch_size):
        p = model.exact_quantum_readout_probabilities(torch.roll(batch, shift, (1, 2)))
        z = model.classify_readout_histogram(p)
        if not torch.isfinite(p).all() or not torch.isfinite(z).all():
            raise FloatingPointError(f"Non-finite output at shift {shift}")
        readouts.append(p.cpu())
        logits.append(z.cpu())
    return torch.cat(readouts), torch.cat(logits)


def periodicity(model, images: torch.Tensor, *, cell: int = 4, batch_size: int = 256) -> dict:
    """Compare every residue to explicit coarse translations, including wrap.

    For full readout, also compare with the *known* spatial permutation.
    This does not search for a best permutation after observing the output.
    """
    translations = ((cell, 0), (0, cell), (3 * cell, 0), (0, 3 * cell),
                    (3 * cell, 3 * cell), (4 * cell, 5 * cell))
    cache = {}

    def get(shift):
        key = tuple(s % images.shape[1] for s in shift)
        if key not in cache:
            cache[key] = outputs(model, images, key, batch_size)
        return cache[key]

    pairs = []
    for row in range(cell):
        for col in range(cell):
            base_p, base_z = get((row, col))
            for dy, dx in translations:
                moved_p, moved_z = get((row + dy, col + dx))
                expected = torch.roll(base_p, (dy // cell, dx // cell), (1, 2))
                pairs.append({
                    "residue": [row, col], "translation": [dy, dx],
                    "readout_invariance_l1_max": float((moved_p-base_p).abs().flatten(1).sum(1).max()),
                    "readout_equivariance_l1_max": float((moved_p-expected).abs().flatten(1).sum(1).max()),
                    "logit_linf_max": float((moved_z-base_z).abs().max()),
                    "prediction_agreement": float((moved_z.argmax(1)==base_z.argmax(1)).double().mean()),
                })
    norm_error = max(float((p.flatten(1).sum(1)-1).abs().max()) for p, _ in cache.values())
    return {
        "n_images": len(images), "cell": cell, "pairs": pairs,
        "readout_invariance_l1_max": max(p["readout_invariance_l1_max"] for p in pairs),
        "readout_equivariance_l1_max": max(p["readout_equivariance_l1_max"] for p in pairs),
        "logit_linf_max": max(p["logit_linf_max"] for p in pairs),
        "prediction_agreement_min": min(p["prediction_agreement"] for p in pairs),
        "normalization_linf_max": norm_error,
    }


def accuracy_scan(model, images, labels, *, batch_size=256, cell=4):
    canvas = images.shape[1]
    # Values are canonical representatives in Z_canvas x Z_canvas.
    corners = ((12, 12), (16, 20), (23, 23), (31, 31))
    shifts = sorted({(r, s) for r in range(cell) for s in range(cell)}
                    | {(k, 0) for k in range(canvas)}
                    | {(0, k) for k in range(canvas)}
                    | {(r % canvas, s % canvas) for r, s in corners})
    predictions = []
    labels = labels.cpu()
    for shift in shifts:
        _, logits = outputs(model, images, shift, batch_size)
        predictions.append(logits.argmax(1))
    predictions = torch.stack(predictions)
    reference = predictions[shifts.index((0, 0))]
    acc = (predictions == labels).double().mean(1)
    agree = (predictions == reference).double().mean(1)
    indices = [[shifts.index((r, s)) for s in range(cell)] for r in range(cell)]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    cell_acc = acc[index_tensor]
    cell_predictions = predictions[[i for row in indices for i in row]]
    pairwise = [[float((a == b).double().mean()) for b in cell_predictions] for a in cell_predictions]
    rows = [{"shift": list(shift), "accuracy": float(acc[i]), "agreement_with_origin": float(agree[i])}
            for i, shift in enumerate(shifts)]
    wrapping = [i for i, (r, s) in enumerate(shifts)
                if (r == 0 and 9 <= s <= 23) or (s == 0 and 9 <= r <= 23)]
    return {
        "n_images": len(labels), "shifts": rows,
        "cell_accuracy": cell_acc.tolist(), "cell_agreement_with_origin": agree[index_tensor].tolist(),
        "cell_pairwise_agreement": pairwise,
        "cell_accuracy_mean": float(cell_acc.mean()),
        "cell_accuracy_min": float(cell_acc.min()), "cell_accuracy_max": float(cell_acc.max()),
        "cell_accuracy_range": float(cell_acc.max()-cell_acc.min()),
        "wrapping_axis_accuracy_mean": float(acc[wrapping].mean()),
        "wrapping_axis_agreement_mean": float(agree[wrapping].mean()),
    }, np.asarray(shifts, dtype=np.int16), predictions.numpy().astype(np.uint8)


def evaluate_checkpoint(checkpoint: Path, output: Path, *, data_root: Path, device="cuda", batch_size=256):
    """Evaluate one final checkpoint; provenance is bound to the actual bytes."""
    torch.set_float32_matmul_precision("highest")
    loaded = load_model_checkpoint(checkpoint, map_location=device)
    model = loaded.model.eval()
    if getattr(model, "shot_budget", None) not in (None, 0):
        raise ValueError("Symmetry diagnostics require exact probabilities, not finite-shot samples")
    cell = 2 ** (model.quantum_layers - 1)
    if model.image_size != 32 or cell != 4 or model.encoding != "frqi":
        raise ValueError("This benchmark requires 32x32 FRQI PCS with three layers")
    test = prepare_mnist_splits(root=data_root, samples_per_class=1000, image_size=32,
                               scaled_image_size=16, max_offset=0, seed=0, download=False).test
    fingerprint = data_hash(test.images, test.labels)
    images, labels = test.images.to(device), test.labels
    residual = periodicity(model, images[:200], batch_size=batch_size)
    accuracy, shifts, predictions = accuracy_scan(model, images, labels, batch_size=batch_size)
    predictions_path = output.with_suffix(".npz")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, shifts=shifts, predictions=predictions, labels=labels.numpy())
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "model_spec": {"module": loaded.model_spec.module, "class_name": loaded.model_spec.class_name,
                       "constructor_kwargs": dict(loaded.model_spec.constructor_kwargs)},
        "dataset_sha256": fingerprint, "dtype": str(next(model.parameters()).dtype),
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
        "matmul_precision": torch.get_float32_matmul_precision(), "shot_budget": None,
        "periodicity": residual, "accuracy": accuracy,
        "predictions_file": predictions_path.name, "predictions_sha256": file_hash(predictions_path),
        "evaluator_sha256": file_hash(Path(__file__)),
    }
    write_json(output, result)
    print(json.dumps({"output": str(output), "cell_accuracy_mean": accuracy["cell_accuracy_mean"],
                      "periodicity_readout_error": residual["readout_equivariance_l1_max"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = vars(parser.parse_args())
    evaluate_checkpoint(**args)


if __name__ == "__main__":
    main()
