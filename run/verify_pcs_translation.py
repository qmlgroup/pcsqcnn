"""Verify portable PCS histories, per-image arrays, and plotted statistics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark_pcs_translation import MODELS, read_plain
from evaluate_pcs_translation import summarize_arrays
from evaluate_summed_translation import file_hash, write_json
from pcs_translation_bundle import DEFAULT_BUNDLE, DEFAULT_CACHE, DEFAULT_WEIGHTS, reconstruct_arrays


def verify(root, models, seeds, *, data_root=Path("data"), device="auto",
           weights_root=DEFAULT_WEIGHTS, cache_root=DEFAULT_CACHE):
    root = Path(root)
    index = read_plain(root/"runs.json")
    checks = {"passed": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "models": list(models), "seeds": list(seeds), "histories": [], "maps": [],
              "verifier_sha256": file_hash(Path(__file__))}
    write_json(root/"verification.json", checks)
    dataset_hashes = set()
    for name in models:
        for seed in seeds:
            entries = index["runs"][name][str(seed)]
            for total in range(2000, MODELS[name][2]+1, 2000):
                entry = entries[str(total)]
                record_file = root/entry["record_file"]
                if file_hash(record_file) != entry["record_sha256"]:
                    raise ValueError(f"Changed training record: {record_file}")
                record = read_plain(record_file)
                history = record["training_history"]
                if record["seed"] != seed or history["train_epoch"] != list(range(1, 2001)):
                    raise ValueError(f"Incomplete training history: {record_file}")
                if history["test_epoch"] != list(range(10, 2001, 10)):
                    raise ValueError(f"Unexpected test history: {record_file}")
                for split in ("train", "test"):
                    a = record["final_summaries"][split]["metrics"]["accuracy"]
                    b = history[f"{split}_metrics"]["accuracy"][-1]
                    if not np.isclose(a, b, atol=1e-7, rtol=0):
                        raise ValueError(f"Final result disagrees with history: {record_file}")
                checks["histories"].append({"model": name, "seed": seed, "epoch": total,
                                            "checkpoint_sha256": entry["checkpoint_sha256"]})
            output = root/"diagnostics"/f"{name}_seed{seed}.json"
            result = read_plain(output)
            if result["identity"]["checkpoint_sha256"] != entries[str(MODELS[name][2])]["checkpoint_sha256"]:
                raise ValueError("Diagnostic uses a different checkpoint")
            dataset_hashes.add(result["identity"]["dataset_sha256"])
            if not result["direct_validation"]["passed"]:
                raise ValueError("Direct reconstruction validation did not pass")
            direct = result["direct_validation"]
            if (result["identity"]["evaluation_dtype"] != "torch.float64"
                    or direct["normalization_linf_max"] >= 1e-10
                    or direct["readout_l1_max"] >= 1e-10
                    or direct["logit_linf_max"] >= 1e-8
                    or direct["changed_labels_max"] != 0):
                raise ValueError("Double-precision direct validation exceeds tolerances")
            arrays_path = output.parent/result["arrays_file"]
            arrays_record = result
            regenerated = not arrays_path.is_file()
            if regenerated:
                arrays_path, arrays_record = reconstruct_arrays(
                    root, name, seed, result, data_root=data_root, device=device,
                    weights=weights_root, cache=cache_root)
                direct = arrays_record["direct_validation"]
                if (not direct["passed"] or direct["normalization_linf_max"] >= 1e-10
                        or direct["readout_l1_max"] >= 1e-10 or direct["logit_linf_max"] >= 1e-8
                        or direct["changed_labels_max"] != 0):
                    raise ValueError("Recomputed direct validation exceeds tolerances")
            if file_hash(arrays_path) != arrays_record["arrays_sha256"]:
                raise ValueError(f"Changed arrays: {arrays_path}")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                expected = 4 if MODELS[name][1] else 32
                if arrays["predictions"].shape != (expected, expected, 10000):
                    raise ValueError("Incomplete prediction grid")
                q = arrays["cell_readouts"]
                spatial = (1, 1) if MODELS[name][1] else (8, 8)
                if q.shape != (4, 4, 10000, *spatial, 4):
                    raise ValueError("Unexpected feature dimensions")
                if q.dtype != np.float64 or not np.isfinite(q).all() or not np.allclose(q.sum(axis=(3, 4, 5)), 1, atol=1e-10, rtol=0):
                    raise ValueError("Invalid saved probabilities")
                if np.any(arrays["predictions"] > 9):
                    raise ValueError("Invalid predicted class")
                recalculated = summarize_arrays(arrays["predictions"], arrays["labels"], q)
            differences = {}
            for metric, values in recalculated.items():
                differences[metric] = float(np.max(np.abs(np.asarray(values)-result["metrics"][metric])))
                if not np.allclose(values, result["metrics"][metric], atol=1e-12, rtol=0):
                    raise ValueError(f"Published {metric} differs from arrays: max error {differences[metric]}")
                if not np.allclose(values, arrays_record["metrics"][metric], atol=1e-12, rtol=0):
                    raise ValueError(f"Recomputed {metric} differs from arrays")
            checks["maps"].append({"model": name, "seed": seed, "grid": [expected, expected],
                                   "arrays_sha256": arrays_record["arrays_sha256"],
                                   "source": "final checkpoint" if regenerated else "archived arrays",
                                   "metric_max_abs_difference": differences})
    if len(dataset_hashes) != 1:
        raise ValueError("The models were evaluated on different centered test data")
    checks.update(passed=True, dataset_sha256=dataset_hashes.pop())
    write_json(root/"verification.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    print(verify(args.output_root, args.models, args.seeds, data_root=args.data_root,
                 device=args.device, weights_root=args.weights_root, cache_root=args.cache_root)["passed"])


if __name__ == "__main__":
    main()
