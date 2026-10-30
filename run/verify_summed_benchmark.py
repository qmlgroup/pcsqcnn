"""Audit saved summed-PCS outputs; optionally verify all nine checkpoints on GPU.

This performs no training and never evaluates pre-existing control models.
Prediction arrays independently reproduce the reported shift statistics.
--checkpoints additionally reconstructs the original seeded data, verifies
optimizer resets and checkpoint provenance, and rechecks fixed-test accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, tol=1e-12):
    if not np.allclose(actual, expected, atol=tol, rtol=0):
        raise ValueError(f"Mismatch: {actual} versus {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/summed_pcs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.artifacts_root.resolve()
    source = Path(__file__).resolve().parents[1]
    record = read(root / "implementation.json")
    for name, digest in record["source_sha256"].items():
        assert sha(source/name) == digest, name
    summary = read(root / "summary.json")
    assert summary["seeds"] == [0, 1, 2]
    checks = {"passed": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "verifier_sha256": sha(Path(__file__)), "source_files_verified": len(record["source_sha256"]),
              "history_records_verified": 0, "predictions_verified": [], "checkpoints_verified": []}
    final_results = {}
    expected_shifts = ({(r, c) for r in range(4) for c in range(4)}
                       | {(k, 0) for k in range(32)} | {(0, k) for k in range(32)}
                       | {(12, 12), (16, 20), (23, 23), (31, 31)})
    for total in (2000, 4000, 6000):
        for seed in range(3):
            result = read(root/f"local_summed_e{total}/result_seed{seed}.json")
            history = result["training_history"]
            assert result["seed"] == seed and result["epoch_offset"] == total-2000
            assert history["train_epoch"] == list(range(1, 2001))
            assert history["test_epoch"] == list(range(10, 2001, 10))
            for split in ("train", "test"):
                close(result["final_summaries"][split]["metrics"]["accuracy"], history[f"{split}_metrics"]["accuracy"][-1])
            final_results[total, seed] = result
            checks["history_records_verified"] += 1
    for total in (2000, 6000):
        for split in ("train", "test"):
            values = [final_results[total,s]["final_summaries"][split]["metrics"]["accuracy"] for s in range(3)]
            close(summary["training"][str(total)][split]["values"], values)
            close(summary["training"][str(total)][split]["mean"], statistics.mean(values))
            close(summary["training"][str(total)][split]["sample_std"], statistics.stdev(values))
    cell_accuracies, cell_agreements, diagnostic_runs = [], [], []
    last200 = [float(np.mean(final_results[6000,s]["training_history"]["test_metrics"]["accuracy"][-20:])) for s in range(3)]
    for seed in range(3):
        run = read(root/f"diagnostics/local_summed_seed{seed}.json")
        npz = root/"diagnostics"/run["predictions_file"]
        assert sha(npz) == run["predictions_sha256"]
        assert sha(source/"run/evaluate_summed_translation.py") == run["evaluator_sha256"]
        with np.load(npz, allow_pickle=False) as saved:
            shifts, predictions, labels = saved["shifts"], saved["predictions"], saved["labels"]
        assert predictions.shape == (76, 10000) and labels.shape == (10000,)
        lookup = {tuple(map(int,s)): p for s,p in zip(shifts,predictions)}
        assert set(lookup) == expected_shifts
        for row in run["accuracy"]["shifts"]:
            pred = lookup[tuple(row["shift"])]
            close(row["accuracy"], np.mean(pred==labels))
            close(row["agreement_with_origin"], np.mean(pred==lookup[0,0]))
        cells = np.array([[np.mean(lookup[r,c]==labels) for c in range(4)] for r in range(4)])
        agreements = np.array([[np.mean(lookup[r,c]==lookup[0,0]) for c in range(4)] for r in range(4)])
        close(cells, run["accuracy"]["cell_accuracy"])
        close(agreements, run["accuracy"]["cell_agreement_with_origin"])
        flat = [lookup[r,c] for r in range(4) for c in range(4)]
        close([[np.mean(a==b) for b in flat] for a in flat], run["accuracy"]["cell_pairwise_agreement"])
        close(cells.mean(), run["accuracy"]["cell_accuracy_mean"])
        close(cells.min(), run["accuracy"]["cell_accuracy_min"])
        close(cells.max(), run["accuracy"]["cell_accuracy_max"])
        close(cells.max()-cells.min(), run["accuracy"]["cell_accuracy_range"])
        wrapping = [pred for (r,c),pred in lookup.items() if (r==0 and 9<=c<=23) or (c==0 and 9<=r<=23)]
        close(np.mean([np.mean(p==labels) for p in wrapping]), run["accuracy"]["wrapping_axis_accuracy_mean"])
        close(np.mean([np.mean(p==lookup[0,0]) for p in wrapping]), run["accuracy"]["wrapping_axis_agreement_mean"])
        periodic_changes = [int(np.sum(pred!=lookup[r%4,c%4])) for (r,c),pred in lookup.items()]
        # This includes all 76 explicitly computed shifts, including corners.
        assert max(periodic_changes) == 0
        pairs = run["periodicity"]["pairs"]
        assert len(pairs) == 96 and run["periodicity"]["n_images"] == 200
        assert {tuple(p["residue"]) for p in pairs} == {(r,c) for r in range(4) for c in range(4)}
        assert run["periodicity"]["readout_invariance_l1_max"] < 1e-6
        assert run["periodicity"]["prediction_agreement_min"] == 1
        for key in ("readout_invariance_l1_max", "logit_linf_max"):
            close(max(p[key] for p in pairs), run["periodicity"][key])
        cell_accuracies.append(cells)
        cell_agreements.append(agreements)
        diagnostic_runs.append(run)
        checks["predictions_verified"].append({"seed": seed, "images": 10000, "shifts": 76,
            "predictions_sha256": sha(npz), "checkpoint_sha256": run["checkpoint_sha256"],
            "dataset_sha256": run["dataset_sha256"], "changed_labels_for_equal_residue": max(periodic_changes)})
    for key, saved_summary in summary["translation"].items():
        values = [r["accuracy"][key] for r in diagnostic_runs]
        close(saved_summary["values"], values)
        close(saved_summary["mean"], statistics.mean(values))
        close(saved_summary["sample_std"], statistics.stdev(values))
    close(summary["periodicity"]["readout_l1_max"], max(r["periodicity"]["readout_invariance_l1_max"] for r in diagnostic_runs))
    close(summary["periodicity"]["logit_linf_max"], max(r["periodicity"]["logit_linf_max"] for r in diagnostic_runs))
    mean_cell = np.mean(cell_accuracies, axis=0)
    mean_agreement = np.mean(cell_agreements, axis=0)
    off_origin = mean_agreement.ravel()[1:]
    checks["derived_statistics"] = {
        "last200_test_mean": statistics.mean(last200), "last200_test_sample_std": statistics.stdev(last200),
        "last200_test_by_seed": last200,
        "mean_cell_accuracy_min": float(mean_cell.min()), "mean_cell_accuracy_max": float(mean_cell.max()),
        "mean_cell_accuracy_range": float(np.ptp(mean_cell)),
        "nonzero_cell_agreement_min": float(off_origin.min()), "nonzero_cell_agreement_max": float(off_origin.max()),
    }
    if args.checkpoints:
        import torch
        from qcnn.automation import _fingerprint_image_dataset
        from qcnn.data import prepare_mnist_splits
        from qcnn.serialization import load_model_checkpoint
        from evaluate_summed_translation import data_hash
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        implementation_hash = hashlib.sha256((json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()).hexdigest()
        centered = prepare_mnist_splits(root=args.data_root, samples_per_class=1000, image_size=32,
                                       scaled_image_size=16, max_offset=0, seed=0, download=False).test
        centered_hash = data_hash(centered.images, centered.labels)
        assert all(r["dataset_sha256"] == centered_hash for r in diagnostic_runs)
        for seed in range(3):
            splits = prepare_mnist_splits(root=args.data_root, samples_per_class=1000, image_size=32,
                                         scaled_image_size=16, max_offset=8, seed=seed, download=False)
            fingerprints = {"train": _fingerprint_image_dataset(splits.train), "test": _fingerprint_image_dataset(splits.test)}
            assert len(splits.train) == len(splits.test) == 10000
            for total in (2000, 4000, 6000):
                directory = root/f"local_summed_e{total}"
                path = directory/f"checkpoint_final_seed{seed}.pt"
                loaded = load_model_checkpoint(path, map_location=args.device)
                state = loaded.training_state
                assert state["epoch"] == 2000 and state["data_fingerprints"] == fingerprints
                assert state["implementation_sha256"] == implementation_hash
                close(list(state["optimizer_state"]["state"].values())[0]["step"].item(), 80000)
                first = load_model_checkpoint(directory/f"checkpoint_epoch1_seed{seed}.pt", map_location="cpu")
                for value in first.training_state["optimizer_state"]["state"].values():
                    close(value["step"].item(), 40)
                model = loaded.model.eval()
                assert sum(p.numel() for p in model.multiplexers.parameters()) == 288
                assert sum(p.numel() for p in model.classifier.parameters()) == 50
                correct = 0
                with torch.no_grad():
                    for images, labels in zip(splits.test.images.split(1600), splits.test.labels.split(1600)):
                        predicted = model(images.to(args.device)).argmax(1).cpu()
                        correct += int((predicted==labels).sum())
                accuracy = correct/10000
                recorded = final_results[total,seed]["final_summaries"]["test"]["metrics"]["accuracy"]
                close(accuracy, recorded, 1e-7)
                if total == 6000:
                    assert sha(path) == checks["predictions_verified"][seed]["checkpoint_sha256"]
                checks["checkpoints_verified"].append({"seed": seed, "total_epochs": total,
                    "checkpoint_sha256": sha(path), "data_fingerprints": fingerprints,
                    "recomputed_test_accuracy": accuracy, "recorded_test_accuracy": recorded,
                    "optimizer_steps_first_epoch": 40, "optimizer_steps_last_epoch": 80000})
                print(f"verified seed={seed} epochs={total} accuracy={accuracy:.4f}", flush=True)
    checks["passed"] = True
    output = args.output or root/("verification.json" if args.checkpoints else "saved_output_verification.json")
    output.write_text(json.dumps(checks,indent=2,allow_nan=False)+"\n")
    print(output)


if __name__ == "__main__":
    main()
