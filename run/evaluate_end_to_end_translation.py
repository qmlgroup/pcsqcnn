"""Evaluate the end-to-end translation behavior of the trained translated-MNIST classifiers.

The final checkpoints of every model in ``MODELS`` (seeds 0, 1, 2) are
re-evaluated, without retraining, on the full 10,000-image MNIST test split.
Every digit is resized to 16 x 16 and centred on the zero-filled 32 x 32
canvas (rows and columns 8-23); a shift is a cyclic roll of the canvas by
``(a, b)`` pixels along image dimensions 1 and 2. For ``|a|, |b| <= 8`` no
occupied pixel crosses the boundary, so these are exactly the bounded
translations of the training data. Per model and seed the script reports:

- ``grid_acc`` and ``grid_cons``: the test accuracy and the fraction of
  predicted labels equal to the zero-offset prediction for each of the
  17 x 17 bounded offsets ``(a, b)`` (``grid[a + 8][b + 8]``), with their means
  (``*_lattice_mean`` over the offsets that are multiples of ``GRID_CELL``,
  without the zero offset for the agreement, ``*_offlattice_mean`` over the
  others);
- ``cyclic_acc``: the test accuracy under a cyclic shift by ``k = 0, ..., 31``
  pixels along one axis, averaged over the two axes; ``wrapping_acc_mean`` is
  its mean over ``k = 9, ..., 23``, where part of the digit wraps across the
  boundary;
- readout errors on the first ``READOUT_DIGITS`` test digits, for the shifts
  in ``LATTICE_SHIFTS``. For the lattice models (PCS-QCNN and RBC-QCNN)
  ``readout_l1_max`` is the largest per-image L1 distance between the readout
  of the shifted digit and the unshifted readout rolled by the shift divided
  by the coarse cell (Proposition on coarse-lattice equivariance). The HKP
  readout carries no lattice, so ``readout_l1_max`` compares it with the
  unshifted readout directly and ``readout_l1_relabel_max`` is the largest
  per-image L1 distance between the sorted outcome probabilities, a lower
  bound on the readout error of every fixed relabelling of the outcomes;
  ``shift(1,0)_vs_unshifted`` is the mean per-image L1 change of the readout
  under a one-pixel shift.

The results are written to
``artifacts/translation_generalization/end_to_end_translation.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import statistics

import torch

from qcnn.data import prepare_mnist_splits
from qcnn.serialization import load_model_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_RELATIVE_PATH = "translation_generalization/end_to_end_translation.json"
CANVAS = 32
MAX_OFFSET = 8
GRID_CELL = 4
WRAPPING_SHIFTS = range(MAX_OFFSET + 1, CANVAS - MAX_OFFSET)
LATTICE_SHIFTS = ((4, 0), (0, 4), (4, 4), (8, -4))
READOUT_DIGITS = 200
SEEDS = (0, 1, 2)
BATCH_SIZE = 2500

# name -> (run directory relative to the artifacts root, model family)
MODELS: dict[str, tuple[str, str]] = {
    "CNN": ("classical_cnn/16on32", "classical"),
    "MLP": ("classical_mlp/16on32", "classical"),
    "PCS-QCNN": ("pcsqcnn_architecture_sweep_full_readout/fq2_ql3", "lattice"),
    "RBC-QCNN": ("pcsqcnn_no_qft_spc1000", "lattice"),
    "HKP-x1-amplitude": ("hur_qcnn_translated_mnist/su4_l1", "hur"),
    "HKP-x1-FRQI": ("encoding_swap/hur_su4_l1_frqi_train8", "hur"),
    "HKP-x8-amplitude": ("hur_qcnn_translated_mnist/ch8_su4_l1", "hur"),
}


def readout(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the readout the head sees: ``[B, H, W, F]`` for lattice models, ``[B, outcomes]`` for HKP."""

    probabilities = model.exact_quantum_readout_probabilities(images)
    if probabilities.ndim == 3:  # HKP: [B, channels, outcomes], averaged over channels before the head
        return probabilities.flatten(1) / probabilities.shape[1]
    return probabilities


def readout_errors(model: torch.nn.Module, images: torch.Tensor, *, family: str) -> dict[str, dict[str, float]]:
    base = readout(model, images)
    exact: dict[str, float] = {}
    relabel: dict[str, float] = {}
    for shift in LATTICE_SHIFTS:
        moved = readout(model, torch.roll(images, shifts=shift, dims=(1, 2)))
        name = f"shift({shift[0]},{shift[1]})"
        if family == "lattice":
            cell = images.shape[1] // base.shape[1]
            expected = torch.roll(base, shifts=(shift[0] // cell, shift[1] // cell), dims=(1, 2))
        else:
            expected = base
            relabel[name] = float((moved.sort(dim=1).values - base.sort(dim=1).values).abs().sum(dim=1).max())
        exact[name] = float((moved - expected).abs().flatten(1).sum(dim=1).max())
    moved = readout(model, torch.roll(images, shifts=1, dims=1))
    exact["shift(1,0)_vs_unshifted"] = float((moved - base).abs().flatten(1).sum(dim=1).mean())
    return {"readout_l1_max": exact, **({"readout_l1_relabel_max": relabel} if relabel else {})}


def predictions(model: torch.nn.Module, images: torch.Tensor, shift: tuple[int, int]) -> torch.Tensor:
    moved = torch.roll(images, shifts=shift, dims=(1, 2))
    return torch.cat([model(batch).argmax(dim=1) for batch in moved.split(BATCH_SIZE)])


def evaluate_seed(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    cache: dict[tuple[int, int], torch.Tensor] = {}

    def predicted(shift: tuple[int, int]) -> torch.Tensor:
        key = (shift[0] % CANVAS, shift[1] % CANVAS)
        if key not in cache:
            cache[key] = predictions(model, images, key)
        return cache[key]

    def accuracy(shift: tuple[int, int]) -> float:
        return float((predicted(shift) == labels).float().mean())

    offsets = range(-MAX_OFFSET, MAX_OFFSET + 1)
    reference = predicted((0, 0))
    grid_acc = [[accuracy((a, b)) for b in offsets] for a in offsets]
    grid_cons = [[float((predicted((a, b)) == reference).float().mean()) for b in offsets] for a in offsets]
    cyclic_acc = [(accuracy((k, 0)) + accuracy((0, k))) / 2 for k in range(CANVAS)]

    def on_lattice(a: int, b: int) -> bool:
        return a % GRID_CELL == 0 and b % GRID_CELL == 0

    cells = [(a, b) for a in offsets for b in offsets]
    lattice = [cell for cell in cells if on_lattice(*cell)]
    offlattice = [cell for cell in cells if not on_lattice(*cell)]

    def mean(grid: list[list[float]], selection: list[tuple[int, int]]) -> float:
        return statistics.fmean(grid[a + MAX_OFFSET][b + MAX_OFFSET] for a, b in selection)

    flat_acc = [value for row in grid_acc for value in row]
    return {
        "grid_acc": grid_acc,
        "grid_cons": grid_cons,
        "grid_acc_mean": statistics.fmean(flat_acc),
        "grid_acc_min": min(flat_acc),
        "grid_acc_max": max(flat_acc),
        "grid_acc_center": grid_acc[MAX_OFFSET][MAX_OFFSET],
        "grid_cons_mean": mean(grid_cons, cells),
        "grid_acc_lattice_mean": mean(grid_acc, lattice),
        "grid_acc_offlattice_mean": mean(grid_acc, offlattice),
        "grid_cons_lattice_mean": mean(grid_cons, [cell for cell in lattice if cell != (0, 0)]),
        "grid_cons_offlattice_mean": mean(grid_cons, offlattice),
        "cyclic_acc": cyclic_acc,
        "cyclic_acc_mean": statistics.fmean(cyclic_acc),
        "wrapping_acc_mean": statistics.fmean(cyclic_acc[k] for k in WRAPPING_SHIFTS),
    }


def summarize(runs: list[dict[str, object]]) -> dict[str, float]:
    """Mean and sample standard deviation over seeds of the table entries, in percent; the worst offset is a mean."""

    def spread(key: str) -> tuple[float, float]:
        values = [100 * float(run[key]) for run in runs]
        return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0

    summary: dict[str, float] = {}
    for key in ("grid_acc_mean", "grid_cons_mean", "grid_cons_lattice_mean", "wrapping_acc_mean"):
        summary[key], summary[f"{key}_std"] = spread(key)
    summary["grid_acc_min_mean"] = statistics.fmean(100 * float(run["grid_acc_min"]) for run in runs)
    for key in ("readout_l1_max", "readout_l1_relabel_max"):
        if key in runs[0]:
            summary[key] = max(value for run in runs for name, value in run[key].items() if name.startswith("shift(") and "vs" not in name)
    if "readout_l1_max" in runs[0]:
        summary["readout_l1_shift1_mean"] = statistics.fmean(run["readout_l1_max"]["shift(1,0)_vs_unshifted"] for run in runs)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end translation behavior of trained classifiers.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--rebuild", action="store_true", help="Accepted for build-script symmetry; always recomputes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Exact float32 matmuls: the readout errors are compared with float32 rounding.
    torch.set_float32_matmul_precision("highest")
    splits = prepare_mnist_splits(
        root=args.data_root,
        samples_per_class=1000,
        image_size=CANVAS,
        scaled_image_size=16,
        max_offset=0,
        seed=0,
        download=False,
    )
    images = splits.test.images.to(device)
    labels = splits.test.labels.to(device)

    output_path = args.artifacts_root / OUTPUT_RELATIVE_PATH
    results: dict[str, object] = {"n_images": len(labels), "models": {}, "summary": {}}
    if output_path.is_file():  # keep the models that are not re-evaluated
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        results["models"] = {name: runs for name, runs in previous["models"].items() if name not in args.models}
        results["summary"] = {name: value for name, value in previous["summary"].items() if name not in args.models}
    with torch.no_grad():
        for name in args.models:
            directory, family = MODELS[name]
            runs = []
            for seed in SEEDS:
                path = args.artifacts_root / directory / f"checkpoint_final_seed{seed}.pt"
                model = load_model_checkpoint(path, map_location=device).model.eval()
                run: dict[str, object] = {"seed": seed, "directory": directory, "family": family}
                if family != "classical":
                    run.update(readout_errors(model, images[:READOUT_DIGITS], family=family))
                run.update(evaluate_seed(model, images, labels))
                runs.append(run)
                print(f"{name} seed {seed}: offsets {100 * run['grid_acc_mean']:.2f}%, wrapping {100 * run['wrapping_acc_mean']:.2f}%", flush=True)
            results["models"][name] = runs
            results["summary"][name] = summarize(runs)
            print(f"{name}: {results['summary'][name]}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
