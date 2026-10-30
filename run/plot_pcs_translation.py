"""Regenerate PCS heatmaps and convergence figures from portable saved data."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark_pcs_translation import MODELS, read_plain
from evaluate_summed_translation import file_hash, write_json
from qcnn.matplotlib_config import MANUSCRIPT_TEXT_WIDTH_IN, configure_manuscript_plot_fonts


NAMES = {"general_full": "General PCS, full", "general_summed": "General PCS, summed",
         "local_full": "Local PCS, full", "local_summed": "Local PCS, summed"}
METRICS = (("accuracy", "Accuracy (%)", "viridis"),
           ("changed_labels", "Changed labels (%)", "magma"),
           ("aligned_feature_tv", "Aligned feature TV (%)", "magma"))


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return values.mean(0), (values.std(0, ddof=1) if len(values)>1 else np.zeros_like(values[0]))


def map_summary(runs):
    """Average images/positions within each seed, then report sample SD over seeds."""
    result = {}
    for metric, _, _ in METRICS:
        values = np.asarray([r["metrics"][metric] for r in runs])*100
        avg = values.mean(0)
        result[metric] = {"mean_map_min_percent": float(avg.min()),
                          "mean_map_max_percent": float(avg.max())}
        subsets = {"whole_grid": values.reshape(len(runs), -1),
                   "origin": values[:, 0, 0, None]}
        if values.shape[1:] == (32, 32):
            offsets = np.arange(32)
            bounded = (offsets <= 8) | (offsets >= 24)
            mask = bounded[:, None] & bounded[None, :]
            subsets.update(bounded=values[:, mask], wrapping=values[:, ~mask],
                           coarse_subgroup=values[:, ::4, ::4].reshape(len(runs), -1))
        for key, subset in subsets.items():
            per_seed = subset.mean(1)
            mean, std = mean_std(per_seed)
            result[metric][key] = {"mean_percent": float(mean), "sample_std_percent": float(std),
                                   "values_percent": per_seed.tolist()}
    return result


def plot_heatmaps(runs, names, tv_limit, destination):
    """Render one or two model rows at the manuscript's final text width."""
    from matplotlib.patches import Rectangle

    headings = ("Accuracy", "Changed class", "Feature TV")
    rows = len(names)
    with plt.rc_context():
        configure_manuscript_plot_fonts(matplotlib)
        fig, axes = plt.subplots(
            rows, 3, squeeze=False,
            figsize=(MANUSCRIPT_TEXT_WIDTH_IN, 2.3 if rows == 1 else 4.5),
            gridspec_kw={"left": .105, "right": .97,
                         "bottom": .31 if rows == 1 else .19,
                         "top": .80 if rows == 1 else .88,
                         "wspace": .38, "hspace": .62})
        images = []
        for row, name in enumerate(names):
            for col, (metric, _, cmap) in enumerate(METRICS):
                ax = axes[row, col]
                avg, _ = mean_std([r["metrics"][metric] for r in runs[name]])
                avg = 100 * avg
                size = avg.shape[0]
                cell = size == 4
                display = avg if cell else np.roll(avg, (16, 16), axis=(0, 1))
                extent = (-.5, 3.5, -.5, 3.5) if cell else (-16.5, 15.5, -16.5, 15.5)
                upper = tv_limit if metric == "aligned_feature_tv" else 100
                img = ax.imshow(display, origin="lower", extent=extent,
                                vmin=0, vmax=upper, cmap=cmap, interpolation="nearest")
                if row == 0:
                    images.append(img)
                label = f"({chr(97 + 3*row + col)})"
                ax.set_title(f"{label} {headings[col]}" if row == 0 else label,
                             loc="left", fontweight="bold", pad=4)
                ax.set_xlabel(r"$\Delta x\,\mathrm{mod}\,4$" if cell else r"$\Delta x$ (pixels)", labelpad=2)
                ax.set_ylabel(r"$\Delta y\,\mathrm{mod}\,4$" if cell else r"$\Delta y$ (pixels)", labelpad=2)
                ax.tick_params(length=2, width=.5, pad=2)
                for spine in ax.spines.values():
                    spine.set_linewidth(.5)
                if cell:
                    ax.set_xticks(range(4)); ax.set_yticks(range(4))
                    ax.set_xticks(np.arange(.5, 3.5), minor=True)
                    ax.set_yticks(np.arange(.5, 3.5), minor=True)
                    ax.tick_params(which="minor", length=0)
                    ax.grid(which="minor", color="white", alpha=.25, linewidth=.3)
                    for y in range(4):
                        for x in range(4):
                            rgb = img.cmap(img.norm(avg[y, x]))[:3]
                            luminance = np.dot(rgb, [.2126, .7152, .0722])
                            precision = 2 if metric == "aligned_feature_tv" and avg[y, x] < 1 else 1
                            ax.text(x, y, f"{avg[y,x]:.{precision}f}",
                                    ha="center", va="center",
                                    color="black" if luminance > .55 else "white")
                else:
                    ax.set_xticks([-16, -8, 0, 8, 15])
                    ax.set_yticks([-16, -8, 0, 8, 15])
                    ax.add_patch(Rectangle((-8.5, -8.5), 17, 17, fill=False,
                                           edgecolor="white", linewidth=.7))
        fig.canvas.draw()
        for row, name in enumerate(names):
            pos = axes[row, 0].get_position()
            title = "General PCS" if name.startswith("general") else "Radius-one PCS"
            fig.text(.105, pos.y1 + (.12 if rows == 1 else .045), title,
                     fontweight="bold", ha="left", va="bottom")
        for col, img in enumerate(images):
            pos = axes[-1, col].get_position()
            bar_y, bar_h = (.12, .035) if rows == 1 else (.080, .018)
            cax = fig.add_axes([pos.x0, bar_y, pos.width, bar_h])
            bar = fig.colorbar(img, cax=cax, orientation="horizontal")
            bar.set_ticks([0, 50, 100] if col < 2 else np.linspace(0, tv_limit, 4))
            bar.ax.tick_params(length=2, width=.5, pad=2)
            bar.outline.set_linewidth(.5)
            bar.set_label("%", labelpad=0)
        fig.savefig(destination.with_suffix(".pdf"))
        fig.savefig(destination.with_suffix(".png"), dpi=220)
        plt.close(fig)


def plot(root, models, seeds, *, figures_dir=None):
    root = Path(root)
    out = root/"plots"
    out.mkdir(parents=True, exist_ok=True)
    index = read_plain(root/"runs.json")
    rows, training_rows, summary = [], [], {}
    # Common limits make the same metric comparable across the four models.
    runs = {name: [read_plain(root/"diagnostics"/f"{name}_seed{s}.json") for s in seeds]
            for name in models}
    tv_limit = max(1e-6, max(float(np.max(mean_std([r["metrics"]["aligned_feature_tv"] for r in runs[n]])[0]))
                             for n in models)*100)
    # A shared, rounded scale preserves comparisons between full and summed readout.
    tv_limit = max(3., 3. * np.ceil(tv_limit / 3.))
    for readout in ("full", "summed"):
        pair = [f"general_{readout}", f"local_{readout}"]
        if all(name in runs for name in pair):
            stem = out / f"pcs_readout_{readout}"
            plot_heatmaps(runs, pair, tv_limit, stem)
            if figures_dir is not None:
                target = Path(figures_dir)
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stem.with_suffix(".pdf"), target / stem.with_suffix(".pdf").name)
    for name in models:
        plot_heatmaps(runs, [name], tv_limit, out / f"{name}_translation")
        for metric, _, _ in METRICS:
            avg, std = mean_std([r["metrics"][metric] for r in runs[name]])
            avg, std = avg * 100, std * 100
            for y in range(avg.shape[0]):
                for x in range(avg.shape[1]):
                    rows.append([name, metric, y, x, avg[y,x], std[y,x], len(seeds)])
        final = {}
        for total in range(2000, MODELS[name][2]+1, 2000):
            records = [read_plain(root/index["runs"][name][str(seed)][str(total)]["record_file"]) for seed in seeds]
            final[str(total)] = {}
            for split in ("train", "test"):
                values = [r["final_summaries"][split]["metrics"]["accuracy"]*100 for r in records]
                avg, std = mean_std(values)
                final[str(total)][split] = {"mean_percent": float(avg), "sample_std_percent": float(std),
                                            "values_percent": values}
                for seed, record in zip(seeds, records):
                    h = record["training_history"]
                    training_rows.extend([name, seed, epoch+total-2000, split, value*100]
                                         for epoch, value in zip(h[f"{split}_epoch"], h[f"{split}_metrics"]["accuracy"]))
        summary[name] = {"training": final, "parameters": runs[name][0]["parameters"],
            "translation": map_summary(runs[name]),
            "readout_l1_max": max(r["direct_validation"]["readout_l1_max"] for r in runs[name]),
            "logit_linf_max": max(r["direct_validation"]["logit_linf_max"] for r in runs[name]),
            "direct_changed_labels_max": max(r["direct_validation"]["changed_labels_max"] for r in runs[name])}
        if all("precision_audit" in r for r in runs[name]):
            summary[name]["precision_audit"] = {
                "normalization_linf_max": max(r["direct_validation"]["normalization_linf_max"] for r in runs[name]),
                "original_normalization_linf_max": max(r["precision_audit"]["original_normalization_linf_max"] for r in runs[name]),
                "changed_prediction_percent_max": max(r["precision_audit"]["changed_prediction_fraction"]*100 for r in runs[name])}

    for file, header, values in (("translation_maps.csv", ["model", "metric", "delta_y_mod32", "delta_x_mod32",
                                "mean_percent", "sample_std_percent", "seeds"], rows),
                               ("training_accuracy.csv", ["model", "seed", "epoch", "split", "accuracy_percent"], training_rows)):
        with (out/file).open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n"); writer.writerow(header); writer.writerows(values)
    with plt.rc_context():
        configure_manuscript_plot_fonts(matplotlib)
        fig, axes = plt.subplots(1, 2, figsize=(MANUSCRIPT_TEXT_WIDTH_IN, 2.6), layout="constrained")
        for ax, split in zip(axes, ("train", "test")):
            for name in models:
                values = [[r[4] for r in training_rows if r[0] == name and r[1] == seed and r[3] == split]
                          for seed in seeds]
                epochs = [r[2] for r in training_rows if r[0] == name and r[1] == seeds[0] and r[3] == split]
                avg, std = mean_std(values)
                line, = ax.plot(epochs, avg, label=NAMES[name], linewidth=1)
                ax.fill_between(epochs, avg-std, avg+std, color=line.get_color(), alpha=.15)
            if any(MODELS[n][2]>2000 for n in models):
                for epoch in (2000, 4000):
                    ax.axvline(epoch, linestyle=":", color="gray", linewidth=.7)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
            ax.set_title(f"({'a' if split=='train' else 'b'}) {split.capitalize()}", loc="left", fontweight="bold")
        axes[1].legend(loc="lower right", labelspacing=0.3, borderpad=0.4)
        fig.savefig(out/"pcs_convergence.pdf"); fig.savefig(out/"pcs_convergence.png", dpi=180); plt.close(fig)
    if figures_dir is not None:
        target = Path(figures_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out/"pcs_convergence.pdf", target/"pcs_readout_convergence.pdf")
    write_json(out/"summary.json", summary)
    report = ["# PCS translation benchmark", "",
        "The same 10000 centered test digits are used at every displacement and for every model.",
        "Heatmap rows are delta_y; columns are delta_x. Full maps use offsets -16..15, modulo 32.",
        "The white square marks the bounded offsets [-8,8]^2; outside it the canvas wraps.",
        "Cell annotations are seed means; sample standard deviations are retained in translation_maps.csv.",
        "Changed predictions are measured against each model's own unshifted predictions.",
        "Feature TV averages per-image distances after undoing the known coarse spatial permutation.",
        "Only this feature metric is aligned; classification always uses the unaligned readout.",
        "16 quantum readouts reconstruct the full grid. Separate direct simulations validate this method.",
        "The summed classifier is invariant under translations divisible by four, not every pixel shift.",
        "Summed readout compresses 256 features to four and is a symmetry demonstration, not an optimized invariant classifier.",
        "Histories use the original fixed randomly translated test split, distinct from the centered shift scans.",
        "Training accuracy is accumulated within an epoch; test accuracy uses the final weights of that epoch.",
        "Local runs restart Adam at epochs 2000 and 4000. All curves are unsmoothed.", "",
        "| Model | Epochs | Parameters | Fixed-test accuracy (%) |",
        "| --- | ---: | ---: | ---: |"]
    for name in models:
        for epoch, splits in summary[name]["training"].items():
            test = splits["test"]
            report.append(f"| {NAMES[name]} | {epoch} | {summary[name]['parameters']} | "
                          f"{test['mean_percent']:.2f} ± {test['sample_std_percent']:.2f} |")
    report.extend(["", "## Translation diagnostics", "",
        "Ranges below are extrema of the seed-mean map, not extrema over individual seeds.",
        "Full-map bounded averages use 289 offsets in [-8,8]^2; wrapping averages use the other 735 offsets.",
        "The coarse subgroup consists of the 64 offsets divisible by four (including the origin).",
        "For summed readout, every whole-grid mean averages the 16 residues equally.",
        "Means and sample SD of spatial averages are computed after averaging within each seed.", "",
        "| Model | Accuracy range (%) | Max changed predictions (%) | Max aligned feature TV (%) |",
        "| --- | ---: | ---: | ---: |"])
    for name in models:
        m = summary[name]["translation"]
        report.append(f"| {NAMES[name]} | {m['accuracy']['mean_map_min_percent']:.2f}–"
                      f"{m['accuracy']['mean_map_max_percent']:.2f} | "
                      f"{m['changed_labels']['mean_map_max_percent']:.2f} | "
                      f"{m['aligned_feature_tv']['mean_map_max_percent']:.4f} |")
    report.extend(["", "Fixed weights are evaluated in double precision without renormalization.",
        "Per-seed direct residuals and native-precision prediction audits are retained in the diagnostic JSON files.",
        "The local full model is a separate repetition; it does not replace the earlier QCS-comparison cohort."])
    (out/"report.md").write_text("\n".join(report)+"\n")
    write_json(out/"SHA256SUMS.json", {p.name: file_hash(p) for p in sorted(out.iterdir())
                                      if p.is_file() and p.name != "SHA256SUMS.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/pcs_translation"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--figures-dir", type=Path, help="Also export the combined heatmaps and convergence PDF here")
    args = parser.parse_args()
    plot(args.output_root, args.models, args.seeds, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()
