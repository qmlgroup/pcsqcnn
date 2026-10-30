"""Summarize completed MFA seeds and export their learning curves as CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import torch


def _epoch_seconds(path: Path, seed: int, expected_epochs: list[int]) -> float | None:
    timing_path = path / "epoch_timing.jsonl"
    if not timing_path.is_file():
        return None
    last_duration: dict[int, float] = {}
    for line in timing_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["seed"] == seed:
            last_duration[int(row["epoch"])] = float(row["elapsed_seconds"])
    if set(last_duration) != set(expected_epochs):
        return None
    return sum(last_duration.values())


def _aggregate(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot aggregate an empty list of seed results.")
    return {
        "count": len(values),
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _runtime_with_manifest_end(path: Path, manifest_path: Path) -> tuple[dict | None, dict | None]:
    runtime_path = path / "runtime.json"
    if not runtime_path.is_file():
        return None, None
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    marker_path = path / "runtime_end_from_manifest.json"
    if not marker_path.is_file():
        return runtime, None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {"method": "manifest_mtime_after_paused_orchestrator"}:
        raise ValueError(f"Unrecognized runtime correction: {marker_path}")
    raw_path = path / "runtime_orchestrator_raw.json"
    raw = (
        json.loads(raw_path.read_text(encoding="utf-8"))
        if raw_path.is_file() else runtime
    )
    start = float(raw["start_unix_time"])
    end = manifest_path.stat().st_mtime
    if end <= start:
        raise ValueError(f"Manifest predates run start: {manifest_path}")
    if not raw_path.is_file():
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    corrected = {
        "start_unix_time": start,
        "end_unix_time": end,
        "wall_seconds": end - start,
        "end_time_source": "manifest_mtime_after_paused_orchestrator",
    }
    if runtime != corrected:
        runtime_path.write_text(json.dumps(corrected, indent=2) + "\n", encoding="utf-8")
    return corrected, raw


def summarize(root: Path, *, require_complete: bool = True) -> tuple[dict, list[dict]]:
    regimes: dict[str, dict] = {}
    curves: list[dict] = []
    for regime in ("translated",):
        runs: list[dict] = []
        for seed in range(3):
            path = root / f"full_{regime}_seed{seed}"
            manifest_path = path / "manifest.json"
            result_path = path / f"result_seed{seed}.pt"
            checkpoint_path = path / f"checkpoint_final_seed{seed}.pt"
            if not (manifest_path.is_file() and result_path.is_file() and checkpoint_path.is_file()):
                if require_complete:
                    raise FileNotFoundError(f"MFA result is incomplete: {path}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["seeds"] != [seed]:
                raise ValueError(f"Unexpected seed order in {manifest_path}")
            payload = torch.load(result_path, map_location="cpu", weights_only=False)
            history = payload["training_history"]
            if history.train_epoch != list(range(1, 2001)):
                raise ValueError(f"Training history is not 2000 epochs: {result_path}")
            if history.test_epoch[-1] != 2000:
                raise ValueError(f"Final test epoch is missing: {result_path}")
            if payload["loss_name"] != "MFAProbabilityNLLLoss":
                raise ValueError(f"MFA objective changed unexpectedly: {result_path}")
            final = payload["final_summaries"]
            runtime, raw_runtime = _runtime_with_manifest_end(path, manifest_path)
            prior_runtime_path = path / "prior_runtime.json"
            prior_runtime = (
                json.loads(prior_runtime_path.read_text(encoding="utf-8"))
                if prior_runtime_path.is_file()
                else None
            )
            current_wall_seconds = None if runtime is None else float(runtime["wall_seconds"])
            prior_wall_seconds = (
                None if prior_runtime is None else float(prior_runtime["wall_seconds_estimate"])
            )
            if prior_wall_seconds is not None and prior_wall_seconds <= 0:
                raise ValueError(f"Invalid prior runtime estimate: {prior_runtime_path}")
            runs.append(
                {
                    "seed": seed,
                    "training_device": payload["device"],
                    "train_accuracy": float(final["train"]["metrics"]["accuracy"]),
                    "test_accuracy": float(final["test"]["metrics"]["accuracy"]),
                    "train_loss": float(final["train"]["loss"]),
                    "test_loss": float(final["test"]["loss"]),
                    "epoch_wall_seconds": _epoch_seconds(path, seed, history.train_epoch),
                    "run_wall_seconds": (
                        None
                        if current_wall_seconds is None
                        else current_wall_seconds + (prior_wall_seconds or 0.0)
                    ),
                    "run_wall_seconds_estimated": prior_wall_seconds is not None,
                    "run_wall_seconds_current_segment": current_wall_seconds,
                    "run_wall_seconds_prior_estimate": prior_wall_seconds,
                    "run_wall_seconds_pause_adjusted": raw_runtime is not None,
                    "run_wall_seconds_original_orchestrator": (
                        None if raw_runtime is None else float(raw_runtime["wall_seconds"])
                    ),
                    "parameter_stats": payload["parameter_stats_line"],
                    "data_fingerprints": payload["data_fingerprints"],
                    "result_path": str(result_path),
                }
            )
            for phase in ("train", "test"):
                epochs = getattr(history, f"{phase}_epoch")
                losses = getattr(history, f"{phase}_loss")
                accuracies = getattr(history, f"{phase}_metrics")["accuracy"]
                curves.extend(
                    {
                        "regime": regime,
                        "seed": seed,
                        "phase": phase,
                        "epoch": epoch,
                        "loss": loss,
                        "accuracy": accuracy,
                    }
                    for epoch, loss, accuracy in zip(epochs, losses, accuracies, strict=True)
                )
        if runs:
            regimes[regime] = {
                "runs": runs,
                "test_accuracy": _aggregate([run["test_accuracy"] for run in runs]),
                "test_loss": _aggregate([run["test_loss"] for run in runs]),
                "train_accuracy": _aggregate([run["train_accuracy"] for run in runs]),
            }
    return {
        "source": "MFA independently implemented ten-class baseline",
        "regimes": regimes,
    }, curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/mfa"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary, curves = summarize(args.root, require_complete=not args.allow_partial)
    if not summary["regimes"]:
        raise ValueError("No completed MFA runs found.")
    summary_path = args.root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    curve_path = args.root / "learning_curves.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("regime", "seed", "phase", "epoch", "loss", "accuracy")
        )
        writer.writeheader()
        writer.writerows(curves)
    print(summary_path)
    print(curve_path)


if __name__ == "__main__":
    main()
