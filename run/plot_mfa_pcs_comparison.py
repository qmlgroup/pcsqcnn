"""Plot matched PCS and MFA train/test accuracy curves from the benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REGIMES = {
    "translated": "Translated MNIST (10k training images)",
}
MODELS = {"PCS-QCNN": "#1f77b4", "MFA": "#d55e00"}


def load_curves(path: Path) -> dict[tuple[str, str, str], dict[int, dict[int, float]]]:
    curves: dict[tuple[str, str, str], dict[int, dict[int, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["regime"], row["model"], row["phase"])
            if key[0] not in REGIMES or key[1] not in MODELS or key[2] not in ("train", "test"):
                raise ValueError(f"Unexpected benchmark series: {key}")
            seed, epoch = int(row["seed"]), int(row["epoch"])
            if seed not in (0, 1, 2):
                raise ValueError(f"Unexpected benchmark seed: {seed}")
            accuracy = float(row["accuracy"])
            if not 0 <= accuracy <= 1:
                raise ValueError(f"Accuracy is outside [0, 1]: {accuracy}")
            by_epoch = curves[key].setdefault(epoch, {})
            if seed in by_epoch:
                raise ValueError(f"Repeated benchmark point: {key}, seed {seed}, epoch {epoch}")
            by_epoch[seed] = accuracy
    expected_keys = {(regime, model, phase)
                     for regime in REGIMES for model in MODELS
                     for phase in ("train", "test")}
    if set(curves) != expected_keys:
        raise ValueError(f"Incomplete benchmark series: {expected_keys - set(curves)}")
    for key, epochs in curves.items():
        expected_epochs = set(range(1, 2001)) if key[2] == "train" else set(range(10, 2001, 10))
        if set(epochs) != expected_epochs:
            raise ValueError(f"Incomplete epoch grid for {key}")
        if any(set(values) != {0, 1, 2} for values in epochs.values()):
            raise ValueError(f"Missing seed in {key}")
    return curves


def render(curves: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from qcnn.matplotlib_config import MANUSCRIPT_TEXT_WIDTH_IN, configure_manuscript_plot_fonts

    configure_manuscript_plot_fonts(matplotlib)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(0.88 * MANUSCRIPT_TEXT_WIDTH_IN, 2.3))
    for regime in REGIMES:
        for model, color in MODELS.items():
            for phase in ("train", "test"):
                series = curves[(regime, model, phase)]
                epochs = sorted(series)
                averages = [mean(series[epoch].values()) for epoch in epochs]
                if phase == "test":
                    deviations = [stdev(series[epoch].values()) for epoch in epochs]
                    axis.plot(epochs, averages, color=color, linewidth=1.0,
                              label=f"{model} test")
                    axis.fill_between(
                        epochs,
                        [max(0, value - deviation) for value, deviation in zip(averages, deviations, strict=True)],
                        [min(1, value + deviation) for value, deviation in zip(averages, deviations, strict=True)],
                        color=color, alpha=0.15, linewidth=0,
                    )
                else:
                    axis.plot(epochs, averages, color=color, linewidth=1.0,
                              linestyle="--", alpha=0.7, label=f"{model} train")
        axis.set(xlabel="Epoch", xlim=(0, 2000), ylim=(0, 1))
        axis.grid(alpha=0.2)
        axis.legend(loc="center right", ncols=2, frameon=False, columnspacing=0.8)
    axis.set_ylabel("Accuracy")
    fig.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path,
                        default=Path("artifacts/mfa/comparison/comparison_learning_curves.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/mfa/comparison/comparison_learning_curves.pdf"))
    args = parser.parse_args()
    render(load_curves(args.curves), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
