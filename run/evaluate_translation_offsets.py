"""Measure the test accuracy of trained models at every translation offset.

Every model is evaluated on centred translated-MNIST test digits (16 x 16 on a
32 x 32 canvas) shifted by each offset ``(dy, dx)`` in ``[-8, 8]^2``. A digit
centred on the canvas stays inside it for all these offsets, so every shifted
image is a valid benchmark image. The script evaluates the final checkpoint of
every seed of the runs in ``RUNS`` that exist and writes
``artifacts/translation_generalization/offset_accuracy.pt``:

- ``offsets``: the offsets ``-8..8`` of both axes;
- ``digits``: the number of test digits (the first ``digits`` of the standard
  test split);
- ``runs[key]``: ``{"directory", "seeds", "accuracy"}`` with ``accuracy`` of
  shape ``[seeds, 17, 17]`` indexed by ``(dy + 8, dx + 8)``.

Each run is evaluated with the float32 matmul precision recorded in its
manifest. Runs already in the file are kept unless ``--rebuild`` is passed.
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
OUTPUT_RELATIVE_PATH = "translation_generalization/offset_accuracy.pt"
MAX_OFFSET = 8
DEFAULT_DIGITS = 2000
EVALUATION_BATCH_SIZE = 1600

# key -> run directory relative to the artifacts root
RUNS: dict[str, str] = {
    "pcs_train2": "translation_generalization/pcsqcnn_fq2_ql3_train2",
    "pcs_reduced_train2": "translation_generalization/pcsqcnn_fq2_ql3_reduced_train2",
    "hur8_train2": "translation_generalization/hur_ch8_su4_l1_train2",
    "pcs_train8": "pcsqcnn_architecture_sweep_full_readout/fq2_ql3",
    "pcs_reduced_train8": "translation_generalization/pcsqcnn_fq2_ql3_reduced_train8",
    "hur8_train8": "hur_qcnn_translated_mnist/ch8_su4_l1",
    "hur1_train2": "encoding_swap/hur_su4_l1_train2",
    "hur1_train8": "hur_qcnn_translated_mnist/su4_l1",
    "hur1_frqi_train2": "encoding_swap/hur_su4_l1_frqi_train2",
    "hur1_frqi_train8": "encoding_swap/hur_su4_l1_frqi_train8",
    "pcs_amplitude_train2": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train2",
    "pcs_amplitude_train8": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_train8",
    "sp_qcnn_train2": "sp_qcnn_translated_mnist/sp_qcnn_train2",
    "sp_qcnn_train8": "sp_qcnn_translated_mnist/sp_qcnn_train8",
    "pcs_local_train2": "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train2",
    "pcs_local_train8": "local_pcsqcnn_translated_mnist/pcsqcnn_local_fq2_ql3_train8",
    "pcs_amplitude_reduced_train2": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train2",
    "pcs_amplitude_reduced_train8": "encoding_swap/pcsqcnn_fq2_ql3_amplitude_reduced_train8",
}


def offset_accuracy(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the accuracy ``[17, 17]`` of ``model`` on ``images`` rolled by every offset."""

    offsets = range(-MAX_OFFSET, MAX_OFFSET + 1)
    accuracy = torch.zeros(len(offsets), len(offsets), dtype=torch.float64)
    with torch.no_grad():
        for row, dy in enumerate(offsets):
            for column, dx in enumerate(offsets):
                shifted = torch.roll(images, shifts=(dy, dx), dims=(1, 2))
                correct = sum(
                    int((model(batch).argmax(dim=1) == batch_labels).sum())
                    for batch, batch_labels in zip(
                        shifted.split(EVALUATION_BATCH_SIZE), labels.split(EVALUATION_BATCH_SIZE), strict=True
                    )
                )
                accuracy[row, column] = correct / len(labels)
    return accuracy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models at every translation offset.")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--digits", type=int, default=DEFAULT_DIGITS, help="Number of test digits to shift.")
    parser.add_argument("--rebuild", action="store_true", help="Re-evaluate runs that are already stored.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    output_path = args.artifacts_root / OUTPUT_RELATIVE_PATH
    stored = torch.load(output_path, weights_only=False) if output_path.is_file() else None
    if stored is not None and stored["digits"] != args.digits:
        stored = None
    results = {
        "offsets": list(range(-MAX_OFFSET, MAX_OFFSET + 1)),
        "digits": args.digits,
        "runs": {} if stored is None else dict(stored["runs"]),
    }

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

    for key, relative_directory in RUNS.items():
        if key in results["runs"] and not args.rebuild:
            continue
        directory = args.artifacts_root / relative_directory
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            print(f"Skipping {key}: {manifest_path} does not exist.", flush=True)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seeds = [int(run["seed"]) for run in manifest["runs"]]
        # Evaluate with the matmul precision the run was trained with.
        torch.set_float32_matmul_precision(manifest["resolved_config"]["training"].get("torch_matmul_precision") or "high")
        accuracy = []
        for run in manifest["runs"]:
            model = load_model_checkpoint(directory / run["checkpoint_final"], map_location=device).model.eval()
            accuracy.append(offset_accuracy(model, images, labels))
            print(f"{key} seed {run['seed']}: mean accuracy {float(accuracy[-1].mean()):.4f}", flush=True)
        results["runs"][key] = {"directory": relative_directory, "seeds": seeds, "accuracy": torch.stack(accuracy)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
