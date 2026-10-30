"""Check readout equivariance and re-align shifted readouts of trained models.

Two diagnostics on the final checkpoints of the translation-generalization and
encoding-swap runs, evaluated on the first ``digits`` standard test digits
centred on the 32 x 32 canvas (no digit crosses the canvas edge for offsets in
``[-8, 8]``):

- equivariance residual: for a column shift by ``s`` pixels, the smallest
  over all cyclic rolls of the PCS-QCNN readout grid of the largest per-image
  L1 distance between the readout of the shifted image and the rolled readout
  of the unshifted image. For the Hur-QCNN, whose outcomes carry no grid, the
  script reports the largest per-image L1 distance between the sorted outcome
  probabilities, a lower bound on the residual of every relabelling of the
  outcomes;
- invariance residual of the sp-QCNN, whose orbit readout is designed to be
  invariant under cyclic rotations of the address bits: the largest per-image
  L1 distance between the readout of the shifted image and that of the
  unshifted image, and the same distance for the image whose row index bits
  are rotated by one position (``bit_rotation``), the symmetry of the model;
- address-bit rotation residuals of the PCS-QCNN and Hur-QCNN models, the same
  row-bit rotation (the qubit cyclic shift ``S`` of the row register):
  ``bit_rotation`` is the largest per-image L1 change of the readout, and
  ``bit_rotation_relabel`` the largest per-image L1 distance between the
  sorted outcome probabilities, a lower bound on the residual of every
  relabelling of the outcomes, so a value well above round-off rules out
  equivariance under ``S`` up to any fixed permutation of the readout;
- re-aligned accuracy of the full-readout PCS-QCNN: the readout of a digit
  shifted by ``(dy, dx)`` is rolled back by ``(round(dy / 4), round(dx / 4))``
  grid cells before the unchanged linear head, which leaves a residual shift
  inside ``[-2, 2]^2``. The script reports the accuracy inside and outside the
  ``[-2, 2]^2`` window, with and without re-alignment.

The results are written to
``artifacts/translation_generalization/readout_alignment.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch

from qcnn.data import prepare_mnist_splits
from qcnn.serialization import load_model_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_RELATIVE_PATH = "translation_generalization/readout_alignment.json"
MAX_OFFSET = 8
TRAIN_WINDOW = 2
GRID_CELL = 4
SHIFTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
DEFAULT_DIGITS = 2000

# key -> (run directory relative to the artifacts root, model family)
EQUIVARIANCE_RUNS: dict[str, tuple[str, str]] = {
    "pcs_frqi": ("pcsqcnn_architecture_sweep_full_readout/fq2_ql3", "pcs"),
    "pcs_amplitude": ("encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8", "pcs"),
    "pcs_local": ("local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8", "pcs"),
    "hur1_amplitude": ("hur_qcnn_translated_mnist/su4_l1", "hur"),
    "hur1_frqi": ("encoding_swap/hur_su4_l1_frqi_train8", "hur"),
    "sp_qcnn": ("sp_qcnn_translated_mnist/sp_qcnn_train8", "sp"),
}
ALIGNMENT_RUNS: dict[str, str] = {
    "pcs_frqi_train2": "translation_generalization/pcsqcnn_fq2_ql3_train2",
    "pcs_amplitude_train2": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train2",
    "pcs_frqi_train8": "pcsqcnn_architecture_sweep_full_readout/fq2_ql3",
    "pcs_amplitude_train8": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8",
}


def _readout(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    probabilities = model.exact_quantum_readout_probabilities(images)
    return probabilities[:, 0] if probabilities.ndim == 3 else probabilities  # Hur: first channel


def equivariance_residual(model: torch.nn.Module, images: torch.Tensor, *, family: str, shift: int) -> float:
    base = _readout(model, images)
    moved = _readout(model, torch.roll(images, shifts=shift, dims=2))
    if family == "hur":
        return float((moved.sort(dim=1).values - base.sort(dim=1).values).abs().sum(dim=1).max())
    if family == "sp":
        return float((moved - base).abs().flatten(1).sum(dim=1).max())
    rolls = range(base.shape[1])
    return min(
        float((moved - torch.roll(base, shifts=(rx, ry), dims=(1, 2))).abs().sum(dim=(1, 2, 3)).max())
        for rx in rolls
        for ry in rolls
    )


def rotate_row_bits(images: torch.Tensor) -> torch.Tensor:
    """Move every pixel row ``r`` to the row whose index bits are those of ``r`` rotated by one position."""

    bits = images.shape[1].bit_length() - 1
    mask = (1 << bits) - 1
    rotated_rows = [((row << 1) | (row >> (bits - 1))) & mask for row in range(images.shape[1])]
    rotated = torch.empty_like(images)
    rotated[:, rotated_rows, :] = images
    return rotated


def bit_rotation_residual(model: torch.nn.Module, images: torch.Tensor) -> float:
    """Largest per-image L1 change of the readout when the row index bits are rotated by one position."""

    base = _readout(model, images)
    return float((_readout(model, rotate_row_bits(images)) - base).abs().flatten(1).sum(dim=1).max())


def bit_rotation_relabel_residual(model: torch.nn.Module, images: torch.Tensor) -> float:
    """Lower bound, over all relabellings of the outcomes, of the readout change under the row-bit rotation."""

    base = _readout(model, images).flatten(1)
    rotated = _readout(model, rotate_row_bits(images)).flatten(1)
    return float((rotated.sort(dim=1).values - base.sort(dim=1).values).abs().sum(dim=1).max())


def aligned_accuracy(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    sums = {"raw_inside": 0.0, "raw_outside": 0.0, "aligned_inside": 0.0, "aligned_outside": 0.0}
    counts = {"inside": 0, "outside": 0}
    for dy in range(-MAX_OFFSET, MAX_OFFSET + 1):
        for dx in range(-MAX_OFFSET, MAX_OFFSET + 1):
            readout = model.exact_quantum_readout_probabilities(torch.roll(images, shifts=(dy, dx), dims=(1, 2)))
            aligned = torch.roll(readout, shifts=(-round(dy / GRID_CELL), -round(dx / GRID_CELL)), dims=(1, 2))
            region = "inside" if max(abs(dy), abs(dx)) <= TRAIN_WINDOW else "outside"
            counts[region] += 1
            for name, values in (("raw", readout), ("aligned", aligned)):
                predictions = model.classify_readout_histogram(values).argmax(dim=1)
                sums[f"{name}_{region}"] += float((predictions == labels).float().mean())
    return {key: value / counts[key.split("_")[1]] for key, value in sums.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check readout equivariance and re-align shifted readouts.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS)
    parser.add_argument("--rebuild", action="store_true", help="Accepted for build-script symmetry; always recomputes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Exact float32 matmuls: the residuals are compared with float32 rounding.
    torch.set_float32_matmul_precision("highest")
    splits = prepare_mnist_splits(
        root=args.data_root,
        samples_per_class=1000,
        image_size=32,
        scaled_image_size=16,
        max_offset=0,
        seed=0,
        download=False,
    )
    images = splits.test.images[: args.digits].to(device)
    labels = splits.test.labels[: args.digits].to(device)

    def load(relative_directory: str, seed: int) -> torch.nn.Module:
        path = args.artifacts_root / relative_directory / f"checkpoint_final_seed{seed}.pt"
        return load_model_checkpoint(path, map_location=device).model.eval()

    results: dict[str, object] = {"digits": args.digits, "equivariance": {}, "alignment": {}}
    with torch.no_grad():
        for key, (directory, family) in EQUIVARIANCE_RUNS.items():
            model = load(directory, 0)
            residuals = {str(shift): equivariance_residual(model, images, family=family, shift=shift) for shift in SHIFTS}
            residuals["bit_rotation"] = bit_rotation_residual(model, images)
            if family != "sp":
                residuals["bit_rotation_relabel"] = bit_rotation_relabel_residual(model, images)
            results["equivariance"][key] = {"directory": directory, "seed": 0, "family": family, "residual": residuals}
            print(f"{key}: {residuals}", flush=True)
        for key, directory in ALIGNMENT_RUNS.items():
            per_seed = [aligned_accuracy(load(directory, seed), images, labels) for seed in SEEDS]
            results["alignment"][key] = {"directory": directory, "seeds": list(SEEDS), "accuracy": per_seed}
            mean = {name: sum(seed[name] for seed in per_seed) / len(per_seed) for name in per_seed[0]}
            print(f"{key}: " + ", ".join(f"{name} {100 * value:.2f}%" for name, value in mean.items()), flush=True)
    output_path = args.artifacts_root / OUTPUT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
