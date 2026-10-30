"""Independently check final MFA artifacts against their saved MNIST test setup."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from qcnn.automation import _fingerprint_image_dataset
from qcnn.serialization import (
    load_auto_training_run,
    reconstruct_run_runner_and_test_loader,
    reconstruct_saved_mnist_splits_from_run,
)


def verify_run(
    run_dir: Path,
    *,
    seed: int,
    data_root: Path,
    device: str,
) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["seeds"] != [seed]:
        raise ValueError(f"Unexpected seed in {run_dir / 'manifest.json'}")
    expected_epochs = int(manifest["resolved_config"]["training"]["num_epochs"])
    loaded = load_auto_training_run(run_dir, seed, map_location="cpu")
    if loaded.model_spec.class_name != "MFAClassifier" or loaded.loss_name != "MFAProbabilityNLLLoss":
        raise ValueError(f"Unexpected MFA model or objective in {run_dir}")
    if loaded.training_history.train_epoch != list(range(1, expected_epochs + 1)):
        raise ValueError(f"Incomplete training history in {run_dir}")
    if loaded.training_history.test_epoch[-1] != expected_epochs:
        raise ValueError(f"Missing final test evaluation in {run_dir}")
    if sum(parameter.numel() for parameter in loaded.model.parameters()) != 105:
        raise ValueError(f"Unexpected trainable parameter count in {run_dir}")

    splits = reconstruct_saved_mnist_splits_from_run(
        loaded, root=data_root, download=False
    )
    result_payload = torch.load(loaded.result_path, map_location="cpu", weights_only=False)
    saved_fingerprints = result_payload["data_fingerprints"]
    actual_fingerprints = {
        "train": _fingerprint_image_dataset(splits.train),
        "test": _fingerprint_image_dataset(splits.test),
    }
    if actual_fingerprints != saved_fingerprints:
        raise ValueError(f"Prepared MNIST data differ from the saved run in {run_dir}")

    context = reconstruct_run_runner_and_test_loader(loaded, splits, device=device)
    recomputed = context.runner.evaluate_loader(context.test_loader)
    saved = loaded.final_summaries["test"]
    comparisons = {
        "test_loss": (float(recomputed.loss), float(saved["loss"])),
        "test_accuracy": (
            float(recomputed.metrics["accuracy"]),
            float(saved["metrics"]["accuracy"]),
        ),
    }
    for name, (actual, expected) in comparisons.items():
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, rel_tol=1e-7, abs_tol=1e-6
        ):
            raise ValueError(
                f"{name} does not reproduce for {run_dir}: "
                f"recomputed={actual}, saved={expected}"
            )
    return {
        "run_dir": str(run_dir),
        "seed": seed,
        "epochs": expected_epochs,
        "device": device,
        "training_device": loaded.device,
        "data_fingerprints": actual_fingerprints,
        "saved_test_loss": comparisons["test_loss"][1],
        "recomputed_test_loss": comparisons["test_loss"][0],
        "saved_test_accuracy": comparisons["test_accuracy"][1],
        "recomputed_test_accuracy": comparisons["test_accuracy"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path)
    target.add_argument("--root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.run_dir is not None:
        if args.seed is None:
            parser.error("--seed is required with --run-dir")
        runs = [(args.run_dir, args.seed)]
    else:
        if args.seed is not None:
            parser.error("--seed is only valid with --run-dir")
        runs = [
            (args.root / f"full_{regime}_seed{seed}", seed)
            for regime in ("translated",)
            for seed in range(3)
        ]
    results = [
        verify_run(path, seed=seed, data_root=args.data_root, device=args.device)
        for path, seed in runs
    ]
    report = {"verified_runs": len(results), "runs": results}
    if args.root is not None:
        output = args.root / "verification.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(output)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
