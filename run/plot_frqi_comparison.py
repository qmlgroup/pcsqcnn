"""Plot saved FRQI learning histories for the compact-PCS comparison.

No model is instantiated and no evaluation or training is run. Select models
explicitly with --models to change the selection; missing or inconsistent histories
are errors, never replaced by reported endpoints or offset-accuracy matrices.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qcnn import load_saved_training_histories
from qcnn.matplotlib_config import MANUSCRIPT_TEXT_WIDTH_IN, configure_manuscript_plot_fonts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "frqi_comparison_convergence"
DEFAULT_MODELS = ("local_pcs", "hkp_frqi", "sp_qcnn", "pcs_general", "address_qcs")


@dataclass(frozen=True)
class Model:
    label: str
    color: str
    seeds: tuple[int, ...]
    stages: tuple[tuple[str, int], ...]


MODELS = {
    "local_pcs": Model("Local PCS", "#0072B2", (0, 1, 2), (
        ("local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8", 2000),
        ("local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e4000_warm", 2000),
        ("local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8_e6000_warm", 2000),
    )),
    "hkp_frqi": Model("HKP x1", "#D55E00", (0, 1, 2), (
        ("encoding_swap/hur_su4_l1_frqi_train8", 2000),
        ("encoding_swap/hur_su4_l1_frqi_train8_e6000_warm", 4000),
    )),
    "sp_qcnn": Model("sp-QCNN", "#009E73", (0, 1), (
        ("sp_qcnn_translated_mnist/sp_qcnn_train8", 2000),
    )),
    "pcs_general": Model("PCS (general)", "#555555", (0, 1, 2), (
        ("pcsqcnn_architecture_sweep_full_readout/fq2_ql3", 2000),
    )),
    "address_qcs": Model("Address-QCS, K=8 (tuned)", "#CC79A7", (0, 1, 2), (
        ("qcsqcnn_translated_mnist_tuned/fq2_ql3_sl8", 2000),
    )),
}


def load_histories(root: Path, model: Model) -> list[SimpleNamespace]:
    """Join restart stages by seed identity and retain their actual epochs."""
    joined = {seed: SimpleNamespace(train_epoch=[], test_epoch=[],
              train_metrics={"accuracy": []}, test_metrics={"accuracy": []})
              for seed in model.seeds}
    offset = 0
    for relative, duration in model.stages:
        directory = root / relative
        manifest = json.loads((directory / "manifest.json").read_text())
        seeds = manifest["seeds"]
        if len(seeds) != len(model.seeds) or set(seeds) != set(model.seeds):
            raise ValueError(f"Unexpected seeds in {directory}: {seeds}")
        histories = load_saved_training_histories(directory)
        for seed, history in zip(seeds, histories, strict=True):
            if history.train_epoch != list(range(1, duration + 1)):
                raise ValueError(f"Incomplete training stage: {directory}, seed {seed}")
            if (not history.test_epoch or history.test_epoch[-1] != duration
                    or history.test_epoch != sorted(set(history.test_epoch))
                    or history.test_epoch[0] < 1):
                raise ValueError(f"Invalid test epochs: {directory}, seed {seed}")
            for split in ("train", "test"):
                epochs = getattr(history, split + "_epoch")
                values = np.asarray(getattr(history, split + "_metrics")["accuracy"])
                if (values.shape != (len(epochs),) or not np.isfinite(values).all()
                        or (values < 0).any() or (values > 1).any()):
                    raise ValueError(f"Invalid {split} accuracy: {directory}, seed {seed}")
                getattr(joined[seed], split + "_epoch").extend(epoch + offset for epoch in epochs)
                getattr(joined[seed], split + "_metrics")["accuracy"].extend(values.tolist())
        offset += duration
    return [joined[seed] for seed in model.seeds]


def summarize(histories, split):
    epochs = getattr(histories[0], split + "_epoch")
    if any(getattr(history, split + "_epoch") != epochs for history in histories):
        raise ValueError(f"Seed histories have different {split} evaluation epochs")
    samples = 100 * np.asarray([getattr(h, split + "_metrics")["accuracy"] for h in histories])
    return np.asarray(epochs), samples.mean(axis=0), samples.std(axis=0, ddof=1)


def render(histories, output_dir: Path):
    import matplotlib
    configure_manuscript_plot_fonts(matplotlib)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(MANUSCRIPT_TEXT_WIDTH_IN, 2.8), sharex=True, sharey=True)
    for ax, split, panel in zip(axes, ("train", "test"), ("a", "b"), strict=True):
        for key, runs in histories.items():
            model = MODELS[key]
            epochs, mean, std = summarize(runs, split)
            ax.fill_between(epochs, mean - std, mean + std, color=model.color, alpha=0.13, linewidth=0)
            ax.plot(epochs, mean, color=model.color, linewidth=1, label=model.label)
            ax.plot(epochs[-1], mean[-1], marker="o", markersize=3.5, color=model.color)
        ax.set_title(f"({panel}) {split.capitalize()} accuracy", fontweight="bold", loc="left")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.18)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(runs[0].train_epoch[-1] for runs in histories.values()))
        ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis="y", labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, handlelength=1.4, columnspacing=1.1, labelspacing=0.3)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.82))
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"{OUTPUT_STEM}.{suffix}"
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figs")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=DEFAULT_MODELS,
                        help="Models to plot (default: %(default)s)")
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error("Each model may be selected only once")
    histories = {key: load_histories(args.artifacts_root, MODELS[key]) for key in args.models}
    render(histories, args.output_dir)
    for key, runs in histories.items():
        epochs, mean, std = summarize(runs, "test")
        print(f"{key}: epoch {epochs[-1]}, test {mean[-1]:.2f} +/- {std[-1]:.2f}%")


if __name__ == "__main__":
    main()
