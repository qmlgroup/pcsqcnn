"""Assemble the matched PCS and MFA MNIST benchmarks without mixing loss scales."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from qcnn.json_artifacts import load_payload


PCS_RUNS = {
    "translated": "pcsqcnn_architecture_sweep_full_readout/fq2_ql3",
}
DATASETS = {
    "translated": {"samples_per_class": 1000, "image_size": 32,
                   "scaled_image_size": 16, "max_offset": 8, "train_batch_size": 256,
                   "test_batch_size": 1600},
}
# Fingerprints of independently reconstructed PCS and MFA image/label tensors.
EXPECTED_FINGERPRINTS = {
    "translated": {
        0: {
            "train": "2dc81f91e5d32f0fc60bd3d8b569cb774acacb56b689e8ce2ae97e75b7bed778",
            "test": "f0c2c3b42a21e2713451029e20c7c33c0b5092f5e8e64f97a60d9c7ea78e4d60",
        },
        1: {
            "train": "4fcd6d9c37d05486d266a5e8cf3149562653bf5b106f7be75c7751c25eec0311",
            "test": "1ee6f8ab7a1dc684d72140da6233627afbc6fdb52f1b1e7b04c5729ca79b155e",
        },
        2: {
            "train": "ceba0929286cd28e56de98de253d0903b9fd823071307f3000947229c15dc0d1",
            "test": "b50a25a595c12be7f3433f0079a81ff612f4e58b9fecbe53771d340eeb39ad3a",
        },
    },
}
CSV_FIELDS = ("model", "regime", "seed", "phase", "epoch", "accuracy", "loss")


def _stats(values: list[float]) -> dict[str, float | int]:
    return {"count": len(values), "mean": mean(values),
            "standard_deviation": stdev(values) if len(values) > 1 else 0.0}


def _parameter_counts(line: str) -> dict[str, int]:
    tokens = line.split()
    totals = {}
    for kind, label in (("Q", "quantum"), ("C", "classical")):
        values = [int(token[1:]) for token in tokens
                  if token.startswith(kind) and token[1:].isdigit()]
        if len(values) != 1:
            raise ValueError(f"Missing or repeated {kind} parameter total: {line}")
        leaf_total = sum(int(token[1:]) for token in tokens
                         if token.startswith(kind.lower()) and token[1:].isdigit())
        if leaf_total != values[0]:
            raise ValueError(f"Inconsistent {kind} parameter total: {line}")
        totals[label] = values[0]
    return {**totals, "total": totals["quantum"] + totals["classical"]}


def _check_protocol(config: dict, regime: str) -> None:
    dataset = config["dataset"]
    for key, value in DATASETS[regime].items():
        if dataset[key] != value:
            raise ValueError(f"{regime}: expected dataset.{key}={value!r}, got {dataset[key]!r}")
    if (config["training"]["num_epochs"] != 2000
            or config["training"]["test_evaluation_interval_epochs"] != 10):
        raise ValueError(f"{regime}: expected 2000 epochs and test every 10 epochs")


def _history_rows(history, model: str, regime: str, seed: int) -> list[dict]:
    if history.train_epoch != list(range(1, 2001)):
        raise ValueError(f"{model} {regime} seed {seed}: incomplete train history")
    if history.test_epoch != list(range(10, 2001, 10)):
        raise ValueError(f"{model} {regime} seed {seed}: incomplete test history")
    rows = []
    for phase in ("train", "test"):
        epochs = getattr(history, f"{phase}_epoch")
        losses = getattr(history, f"{phase}_loss")
        accuracies = getattr(history, f"{phase}_metrics")["accuracy"]
        if not len(epochs) == len(losses) == len(accuracies):
            raise ValueError(f"{model} {regime} seed {seed}: history length mismatch")
        rows.extend({"model": model, "regime": regime, "seed": seed,
                     "phase": phase, "epoch": epoch, "accuracy": float(accuracy),
                     "loss": float(loss)}
                    for epoch, accuracy, loss in zip(epochs, accuracies, losses, strict=True))
    return rows


def load_pcs(root: Path) -> tuple[dict, list[dict]]:
    summary = {}
    curves = []
    for regime, relative_path in PCS_RUNS.items():
        run_dir = root / relative_path
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["seeds"] != [0, 1, 2]:
            raise ValueError(f"Unexpected PCS seeds: {run_dir}")
        config = manifest["resolved_config"]
        _check_protocol(config, regime)
        if config["model"]["class_name"] != "PCSQCNN":
            raise ValueError(f"Not the main PCS architecture: {run_dir}")
        runs = []
        for seed in range(3):
            result_path = run_dir / f"result_seed{seed}.pt"
            payload = load_payload(result_path)
            result_config = dict(payload["resolved_config"])
            saved_seed = result_config.pop("seed", None)
            if payload["seed"] != seed or saved_seed != seed or result_config != config:
                raise ValueError(f"PCS manifest/result mismatch: {result_path}")
            if payload["loss_name"] != "CrossEntropyLoss":
                raise ValueError(f"Unexpected PCS objective: {result_path}")
            history = payload["training_history"]
            curves.extend(_history_rows(history, "PCS-QCNN", regime, seed))
            final = payload["final_summaries"]
            runs.append({"seed": seed,
                         "train_accuracy": float(final["train"]["metrics"]["accuracy"]),
                         "test_accuracy": float(final["test"]["metrics"]["accuracy"]),
                         "train_loss": float(final["train"]["loss"]),
                         "test_loss": float(final["test"]["loss"]),
                         "loss_name": payload["loss_name"],
                         "parameter_stats": payload["parameter_stats_line"],
                         "parameter_counts": _parameter_counts(payload["parameter_stats_line"]),
                         "result_path": str(result_path)})
        if any(run["parameter_counts"] != runs[0]["parameter_counts"] for run in runs):
            raise ValueError(f"PCS parameter counts differ across seeds: {regime}")
        summary[regime] = {"source_run": relative_path, "runs": runs,
                           "test_accuracy": _stats([run["test_accuracy"] for run in runs]),
                           "parameter_counts": runs[0]["parameter_counts"]}
    return summary, curves


def load_mfa(root: Path) -> tuple[dict, list[dict]]:
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification["verified_runs"] != 3 or len(verification["runs"]) != 3:
        raise ValueError("The translated MFA three-run verification is incomplete")
    summary = json.loads((root / "benchmark_summary.json").read_text(encoding="utf-8"))
    regimes = summary["regimes"]
    verified = {(Path(row["run_dir"]).name, row["seed"]): row
                for row in verification["runs"]}
    for regime in PCS_RUNS:
        if regime not in regimes:
            raise ValueError(f"Missing MFA regime: {regime}")
        runs = regimes[regime]["runs"]
        if [row["seed"] for row in runs] != [0, 1, 2]:
            raise ValueError(f"Missing MFA seeds for {regime}")
        if abs(regimes[regime]["test_accuracy"]["mean"]
               - mean(row["test_accuracy"] for row in runs)) > 1e-9:
            raise ValueError(f"MFA aggregate/seed mismatch: {regime}")
        for row in runs:
            row["parameter_counts"] = _parameter_counts(row["parameter_stats"])
            if row["parameter_counts"] != {"quantum": 105, "classical": 0, "total": 105}:
                raise ValueError(f"Unexpected MFA parameter count: {regime} seed {row['seed']}")
            key = (f"full_{regime}_seed{row['seed']}", row["seed"])
            check = verified.get(key)
            if check is None or check["epochs"] != 2000:
                raise ValueError(f"Unverified MFA run: {key}")
            if abs(check["saved_test_accuracy"] - row["test_accuracy"]) > 1e-6:
                raise ValueError(f"MFA verification/summary mismatch: {key}")
            if row.get("data_fingerprints") != EXPECTED_FINGERPRINTS[regime][row["seed"]]:
                raise ValueError(f"MFA data do not match the PCS reference: {key}")
    curves = []
    with (root / "learning_curves.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            curves.append({"model": "MFA", "regime": row["regime"],
                           "seed": int(row["seed"]), "phase": row["phase"],
                           "epoch": int(row["epoch"]),
                           "accuracy": float(row["accuracy"]), "loss": float(row["loss"])})
    for regime in PCS_RUNS:
        for seed in range(3):
            seed_rows = [row for row in curves if row["regime"] == regime and row["seed"] == seed]
            train_epochs = sorted(row["epoch"] for row in seed_rows if row["phase"] == "train")
            test_rows = sorted((row for row in seed_rows if row["phase"] == "test"),
                               key=lambda row: row["epoch"])
            if train_epochs != list(range(1, 2001)) or [row["epoch"] for row in test_rows] != list(range(10, 2001, 10)):
                raise ValueError(f"Incomplete MFA learning curve: {regime} seed {seed}")
            if abs(test_rows[-1]["accuracy"] - regimes[regime]["runs"][seed]["test_accuracy"]) > 1e-6:
                raise ValueError(f"MFA curve/summary mismatch: {regime} seed {seed}")
    return regimes, curves


def write_report(output: Path, pcs: dict, pcs_curves: list[dict],
                 mfa: dict | None = None, mfa_curves: list[dict] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [*pcs_curves, *(mfa_curves or [])]
    with (output / "comparison_learning_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {"pcs": pcs, "mfa": mfa,
              "note": "Compare accuracy on matched data. PCS and MFA losses use different objectives and are not directly comparable."}
    if mfa is not None:
        report["mean_accuracy_difference_mfa_minus_pcs"] = {
            regime: (mfa[regime]["test_accuracy"]["mean"]
                     - pcs[regime]["test_accuracy"]["mean"])
            for regime in PCS_RUNS
        }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# PCS-QCNN and MFA benchmark", "",
             "Final test accuracy on the same translated-MNIST images. Values are",
             "mean ± sample standard deviation across seeds 0, 1, 2.",
             "The final column is the difference of means in percentage points.", "",
             "| Regime | PCS-QCNN | MFA | MFA − PCS |", "| --- | ---: | ---: | ---: |"]
    for regime in PCS_RUNS:
        p = pcs[regime]["test_accuracy"]
        p_text = f"{100*p['mean']:.2f} ± {100*p['standard_deviation']:.2f}%"
        if mfa is None:
            m_text = delta_text = "pending"
        else:
            m = mfa[regime]["test_accuracy"]
            d = report["mean_accuracy_difference_mfa_minus_pcs"][regime]
            m_text = f"{100*m['mean']:.2f} ± {100*m['standard_deviation']:.2f}%"
            delta_text = f"{100*d:+.2f} pp"
        lines.append(f"| {regime} | {p_text} | {m_text} | {delta_text} |")
    lines.append("")
    for regime in PCS_RUNS:
        p = pcs[regime]["parameter_counts"]
        text = (f"{regime.capitalize()} trainable parameters: PCS-QCNN "
                f"{p['quantum']:,} quantum + {p['classical']:,} classical")
        if mfa is not None:
            q = mfa[regime]["runs"][0]["parameter_counts"]
            text += f"; MFA {q['quantum']:,} quantum + {q['classical']:,} classical"
        lines.append(text + ".")
    lines.extend(["", "The translated PCS reference is the full-readout `fq2_ql3` model,",
                  "not the `pcsqcnn_no_qft_spc1000` ablation. The two models use the",
                  "same MNIST image/label split and geometry within each seed.",
                  "The models' input encodings, parameter counts, and objectives differ, so",
                  "accuracy differences alone cannot isolate quantum symmetry.",
                  "Their loss values are not directly comparable.",
                  "MFA wall times are reported in `benchmark_summary.json`; matching",
                  "PCS wall times are not available in the saved artifacts.",
                  "Three seeds do not support a strong significance claim.", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcs-artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--mfa-root", type=Path, default=Path("artifacts/mfa"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mfa/comparison"))
    parser.add_argument("--pcs-only", action="store_true")
    args = parser.parse_args()
    pcs, pcs_curves = load_pcs(args.pcs_artifacts_root)
    mfa, mfa_curves = (None, None) if args.pcs_only else load_mfa(args.mfa_root)
    write_report(args.output_dir, pcs, pcs_curves, mfa, mfa_curves)
    print(args.output_dir / "comparison.md")


if __name__ == "__main__":
    main()
