"""Export compact tables and plots after the three summed-PCS runs finish."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/summed_pcs"))
    args = parser.parse_args()
    root = args.artifacts_root
    state = json.loads((root / "status.json").read_text())
    if state["status"] != "complete":
        raise RuntimeError("Final plots require all three training runs and diagnostics to complete")
    summary = json.loads((root / "summary.json").read_text())
    runs = [json.loads((root / "diagnostics" / f"local_summed_seed{s}.json").read_text()) for s in range(3)]
    out = root / "plots"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})

    def save(fig, name):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def csv_table(name, header, rows):
        with (out / name).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(axes, ("cell_accuracy", "cell_agreement_with_origin"),
                                ("Classification accuracy (%)", "Agreement with (0, 0) (%)")):
        values = 100 * np.array([r["accuracy"][key] for r in runs])
        mean, std = values.mean(0), values.std(0, ddof=1)
        im = axis.imshow(mean, origin="lower", vmin=0, vmax=100, cmap="viridis")
        for row in range(4):
            for col in range(4):
                axis.text(col, row, f"{mean[row,col]:.1f}\n±{std[row,col]:.1f}",
                          ha="center", va="center", color="white" if mean[row,col]<55 else "black", fontsize=8)
        axis.set(xlabel="Column shift (pixels)", ylabel="Row shift (pixels)", title=title,
                 xticks=range(4), yticks=range(4))
        fig.colorbar(im, ax=axis, shrink=.8)
        csv_table(f"{key}.csv", ["row_shift", "column_shift", "mean_percent", "sample_std_percent"],
                  [(r, c, mean[r,c], std[r,c]) for r in range(4) for c in range(4)])
    save(fig, "summed_fundamental_cell")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    rows = []
    for direction, color in (("row", "tab:blue"), ("column", "tab:orange")):
        for axis, metric, label in zip(axes, ("accuracy", "agreement_with_origin"),
                                       ("Accuracy (%)", "Agreement with (0, 0) (%)")):
            values = []
            for run in runs:
                lookup = {tuple(v["shift"]): v for v in run["accuracy"]["shifts"]}
                values.append([lookup[(k, 0) if direction=="row" else (0, k)][metric] for k in range(32)])
            values = 100*np.array(values)
            mean, std = values.mean(0), values.std(0, ddof=1)
            axis.plot(range(32), mean, color=color, label=direction)
            axis.fill_between(range(32), mean-std, mean+std, color=color, alpha=.15)
            axis.scatter(range(0,32,4), mean[::4], color=color, s=20)
            axis.set(ylabel=label, ylim=(0, 103))
            rows.extend((direction, metric, k, mean[k], std[k]) for k in range(32))
    for axis in axes:
        axis.axvspan(8.5, 23.5, color="grey", alpha=.12, label="Boundary crossing")
        axis.grid(alpha=.2)
    axes[0].legend(loc="best")
    axes[1].set(xlabel="Cyclic shift (pixels)", xticks=range(0,32,4))
    save(fig, "summed_cyclic_shifts")
    csv_table("cyclic_shifts.csv", ["axis", "metric", "shift", "mean_percent", "sample_std_percent"], rows)

    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for split, color in (("train", "tab:blue"), ("test", "tab:orange")):
        curves = []
        for seed in range(3):
            epochs, scores = [], []
            for stage, total in enumerate((2000, 4000, 6000)):
                history = json.loads((root / f"local_summed_e{total}" / f"result_seed{seed}.json").read_text())["training_history"]
                epochs.extend(stage*2000 + n for n in history[f"{split}_epoch"])
                scores.extend(history[f"{split}_metrics"]["accuracy"])
            curves.append(scores)
        curves = 100*np.array(curves)
        mean, std = curves.mean(0), curves.std(0, ddof=1)
        axis.plot(epochs, mean, label=split, color=color, linewidth=.8)
        axis.fill_between(epochs, mean-std, mean+std, color=color, alpha=.12)
    for epoch in (2000, 4000): axis.axvline(epoch, color="grey", linestyle=":", linewidth=.8)
    axis.set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0,100))
    axis.legend()
    axis.grid(alpha=.2)
    save(fig, "summed_convergence")

    table = []
    for epoch in (2000, 6000):
        for split in ("train", "test"):
            value = summary["training"][str(epoch)][split]
            table.append((epoch, split, 100*value["mean"], 100*value["sample_std"]))
    csv_table("training_accuracy.csv", ["epoch", "split", "mean_percent", "sample_std_percent"], table)
    final = summary["training"]["6000"]["test"]
    p = summary["periodicity"]
    cell = 100*np.array([r["accuracy"]["cell_accuracy"] for r in runs]).mean(0)
    translation = summary["translation"]
    cell_range = translation["cell_accuracy_range"]
    wrapping = translation["wrapping_axis_accuracy_mean"]
    last200 = np.array([
        json.loads((root/"local_summed_e6000"/f"result_seed{s}.json").read_text())
        ["training_history"]["test_metrics"]["accuracy"][-20:] for s in range(3)
    ]).mean(1)*100
    report = ["# Compact PCS with summed readout", "",
              "FRQI, radius-one convolutions, 288 quantum + 50 head parameters, seeds 0/1/2.", "",
              f"Final fixed-test accuracy at 6000 epochs: {100*final['mean']:.2f} ± {100*final['sample_std']:.2f}% (sample SD).", "",
              "| Epochs | Train (%) | Test (%) |", "| --- | --- | --- |",
              *[f"| {epoch} | {100*summary['training'][str(epoch)]['train']['mean']:.2f} ± {100*summary['training'][str(epoch)]['train']['sample_std']:.2f} | {100*summary['training'][str(epoch)]['test']['mean']:.2f} ± {100*summary['training'][str(epoch)]['test']['sample_std']:.2f} |" for epoch in (2000,6000)], "",
              "Training accuracy is accumulated during the epoch; test accuracy uses fixed final weights.",
              f"Mean test accuracy over the last 200 epochs: {last200.mean():.2f} ± {last200.std(ddof=1):.2f}% (SD across the three seed means).",
              "The histories still contain sharp fluctuations; no best-epoch selection is used.", "",
              f"Across the 16 positions, seed-averaged accuracy ranges from {cell.min():.2f}% to {cell.max():.2f}%.",
              f"The within-seed range across positions is {100*cell_range['mean']:.2f} ± {100*cell_range['sample_std']:.2f} percentage points.",
              f"Mean accuracy for boundary-crossing axis shifts (9..23 pixels): {100*wrapping['mean']:.2f} ± {100*wrapping['sample_std']:.2f}%.", "",
              f"Maximum readout L1 periodicity error: {p['readout_l1_max']:.6g}.",
              f"Maximum logit error: {p['logit_linf_max']:.6g}.",
              f"Minimum label agreement across tested coarse translations: {100*p['prediction_agreement_min']:.6f}%.", "",
              "Readout/logit residuals use the first 200 test digits: all 16 residues and six additional coarse translations per residue.",
              "Translation plots evaluate the same 10000 centered test digits at each displacement.",
              "These are distinct from the original fixed randomly translated test split.",
              "The 4x4 cells and wrapping predictions were computed explicitly, without orbit averaging.",
              "Translations by multiples of four preserve summed readout; arbitrary one-pixel invariance is not assumed.",
              "Agreement with the origin for a nonzero residue is distinct from agreement between equal residues.",
              "Boundary-crossing and non-crossing means weight residues differently; periodicity should be assessed at matching residues.",
              "Convergence curves are unsmoothed seed means with sample-standard-deviation bands.",
              "Vertical dotted lines mark the planned Adam restarts at epochs 2000 and 4000.", ""]
    verification = root/"verification.json"
    if verification.exists():
        audit = json.loads(verification.read_text())
        if not audit["passed"]: raise ValueError("Saved verification did not pass")
        maximum_changes = max(r["changed_labels_for_equal_residue"] for r in audit["predictions_verified"])
        report += [f"Saved predictions were checked for all 76 distinct shifts and all 10000 digits per seed: maximum {maximum_changes} changed labels between matching residues.",
                   f"Checkpoint/data/optimizer audits and independent fixed-test re-evaluations passed for {len(audit['checkpoints_verified'])} saved stage checkpoints.", ""]
    (out / "report.md").write_text("\n".join(report))
    inventory = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    inventory["plotter_source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "SHA256SUMS.json").write_text(json.dumps(inventory, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
