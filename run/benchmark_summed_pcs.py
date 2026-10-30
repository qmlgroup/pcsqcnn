"""Run the local PCS summed-readout benchmark on translated MNIST.

Three seeds, 3x2000 epochs, fresh Adam at epochs 2000 and 4000, FRQI,
10,000 training images and the unchanged 10,000-image test split. Each stage
can resume exactly after interruption, including optimizer and RNG state.
After training, run explicit fundamental-cell and wrapping diagnostics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import torch

from qcnn.automation import run_mnist_auto_training
from qcnn.data import prepare_mnist_splits
from qcnn.model_spec import instantiate_model
from prepare_local_pcsqcnn_translated_mnist_data import (
    LocalPcsqcnnTranslatedMnistDefaults, iter_local_pcsqcnn_training_configs,
)
from evaluate_summed_translation import evaluate_checkpoint, periodicity, write_json


ROOT = Path(__file__).resolve().parents[1]


def configurations(data_root: Path, artifacts_root: Path, device: str):
    defaults = LocalPcsqcnnTranslatedMnistDefaults(
        data_root=data_root, artifacts_root=artifacts_root, device=device,
        download=False, seed_count=3,
    )
    base, = iter_local_pcsqcnn_training_configs(defaults, runs=["pcsqcnn_local_fq2_ql3_reduced_train8"])
    stages = []
    for total in (2000, 4000, 6000):
        initial = (None if not stages else str(artifacts_root / stages[-1].output.directory_name
                                               / "checkpoint_final_seed{seed}.pt"))
        stages.append(replace(base,
            training=replace(base.training, init_checkpoint=initial,
                             snapshot_epochs=(1, *range(100, 2001, 100))),
            output=replace(base.output, directory_name=f"local_summed_e{total}")))
    return stages


def provenance():
    files = sorted((ROOT / "src/qcnn").rglob("*.py"))
    files += [ROOT / name for name in ("pyproject.toml", "uv.lock",
              "run/prepare_local_pcsqcnn_translated_mnist_data.py",
              "run/benchmark_summed_pcs.py", "run/evaluate_summed_translation.py")]
    return {
        "purpose": "local PCS summed readout; original translated-MNIST protocol; 3x2000 epochs",
        "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/summed_pcs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = configurations(args.data_root.resolve(), args.artifacts_root.resolve(), args.device)
    if args.dry_run:
        print(json.dumps([asdict(c) for c in configs], indent=2, default=str))
        return
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not allowed")
    root = args.artifacts_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = provenance()
    record_path = root / "implementation.json"
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError("Benchmark sources or software changed; use a separate output directory")
    write_json(record_path, record)
    write_json(root / "protocol.json", {
        "configs": json.loads(json.dumps([asdict(c) for c in configs], default=str)),
        "threads": args.threads,
        "note": "Original fixed test accuracy is distinct from centered cyclic scans",
    })
    state = {"updated_utc": now(), "status": "starting", "pid": os.getpid()}
    def status(**values):
        state.update(values, updated_utc=now())
        write_json(root / "status.json", state)

    try:
        structural = root / "random_weight_periodicity.json"
        if not structural.exists():
            status(status="checking_random_weight_symmetry")
            test = prepare_mnist_splits(root=args.data_root, image_size=32, scaled_image_size=16,
                                       max_offset=0, samples_per_class=1000, seed=0, download=False).test
            torch.manual_seed(1234)
            model = instantiate_model(configs[0].model).eval().to(args.device)
            diagnostics = {}
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float64, 1e-10)):
                model = model.to(dtype=dtype)
                diagnostics[str(dtype)] = periodicity(model, test.images[:200].to(args.device, dtype=dtype))
                d = diagnostics[str(dtype)]
                if max(d["readout_invariance_l1_max"], d["logit_linf_max"], d["normalization_linf_max"]) > tolerance:
                    raise RuntimeError(f"Structural symmetry check failed: {dtype}: {d}")
            write_json(structural, {"initialization_seed": 1234, "n_images": 200,
                                   "diagnostics": diagnostics, "tolerances": {"float32": 1e-4, "float64": 1e-10}})
            del model, test
            gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()

        for stage, config in enumerate(configs):
            directory = root / config.output.directory_name
            start = time.monotonic()
            def report(seed, epoch, train_summary, test_summary, history):
                if epoch == 1 or test_summary is not None:
                    current = {"seed": seed, "stage": stage + 1, "epoch": epoch,
                               "total_epoch": stage * 2000 + epoch,
                               "train_accuracy": train_summary.metrics["accuracy"],
                               "test_accuracy": None if test_summary is None else test_summary.metrics["accuracy"],
                               "stage_elapsed_seconds": time.monotonic()-start}
                    status(status="training", **current)
                    write_json(directory / f"history_seed{seed}.json", {"seed": seed, "epoch_offset": stage*2000,
                                                                        "training_history": asdict(history)})
                    print(json.dumps(current), flush=True)
            status(status="training", stage=stage+1)
            result = run_mnist_auto_training(config, resume=directory.exists(), save_training_state=True,
                                             implementation=record, seed_epoch_end_callback=report)
            for run in result.runs:
                write_json(directory / f"result_seed{run.seed}.json", {
                    "seed": run.seed, "device": args.device, "resolved_config": run.resolved_config,
                    "epoch_offset": stage*2000, "training_history": asdict(run.history),
                    "final_summaries": {"train": asdict(run.final_train_summary), "test": asdict(run.final_test_summary)},
                })
            del result
            gc.collect()
            if args.device == "cuda": torch.cuda.empty_cache()

        evaluations = []
        for seed in range(3):
            status(status="evaluating", seed=seed)
            checkpoint = root / configs[-1].output.directory_name / f"checkpoint_final_seed{seed}.pt"
            output = root / "diagnostics" / f"local_summed_seed{seed}.json"
            if output.exists():
                saved = json.loads(output.read_text())
                if saved["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                    raise ValueError("Diagnostic checkpoint changed")
            else:
                saved = evaluate_checkpoint(checkpoint, output, data_root=args.data_root, device=args.device)
            evaluations.append(saved)
        summary = {"seeds": [0, 1, 2], "training": {}, "translation": {}}
        for total in (2000, 6000):
            values = [json.loads((root / f"local_summed_e{total}" / f"result_seed{s}.json").read_text())
                      for s in range(3)]
            summary["training"][str(total)] = {}
            for split in ("train", "test"):
                scores = [v["final_summaries"][split]["metrics"]["accuracy"] for v in values]
                summary["training"][str(total)][split] = {"values": scores, "mean": statistics.mean(scores),
                                                         "sample_std": statistics.stdev(scores)}
        for metric in ("cell_accuracy_mean", "cell_accuracy_min", "cell_accuracy_max", "cell_accuracy_range",
                       "wrapping_axis_accuracy_mean", "wrapping_axis_agreement_mean"):
            values = [d["accuracy"][metric] for d in evaluations]
            summary["translation"][metric] = {"values": values, "mean": statistics.mean(values),
                                               "sample_std": statistics.stdev(values)}
        summary["periodicity"] = {
            "readout_l1_max": max(d["periodicity"]["readout_invariance_l1_max"] for d in evaluations),
            "logit_linf_max": max(d["periodicity"]["logit_linf_max"] for d in evaluations),
            "prediction_agreement_min": min(d["periodicity"]["prediction_agreement_min"] for d in evaluations),
        }
        write_json(root / "summary.json", summary)
        status(status="complete")
    except BaseException as error:
        status(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
