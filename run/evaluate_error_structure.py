from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from qcnn.article_figures import load_saved_run_seeds, resolve_saved_run_seed
from qcnn import (
    collect_error_analysis_payload,
    load_auto_training_run,
    reconstruct_run_runner_and_test_loader,
    reconstruct_saved_mnist_splits_from_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SWEEP_DIRECTORY_NAME = "pcsqcnn_image_size_sweep"
DEFAULT_SCALED_IMAGE_SIZE = 16
DEFAULT_IMAGE_SIZE = 16


def load_manifest_seeds(run_directory: str | Path) -> list[int]:
    return load_saved_run_seeds(run_directory)


def resolve_error_structure_seed(
    run_directory: str | Path,
    *,
    seed: int | None,
) -> int:
    return resolve_saved_run_seed(run_directory, seed=seed)


def build_figure_5b_run_directory(
    artifacts_root: str | Path,
    *,
    scaled_image_size: int | None = None,
    image_size: int,
) -> Path:
    resolved_scaled_image_size = image_size if scaled_image_size is None else scaled_image_size
    if resolved_scaled_image_size <= 0:
        raise ValueError(f"scaled_image_size must be positive, got {resolved_scaled_image_size}.")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    return (
        Path(artifacts_root)
        / DEFAULT_SWEEP_DIRECTORY_NAME
        / f"{resolved_scaled_image_size}on{image_size}"
    )


def build_error_structure_output_path(
    run_directory: str | Path,
    *,
    seed: int,
) -> Path:
    return Path(run_directory) / f"error_structure_seed{seed}.pt"


def extract_error_structure(
    *,
    run_directory: str | Path,
    root: str | Path,
    seed: int,
    device: str | None = None,
    download: bool = True,
) -> dict[str, Any]:
    loaded_run = load_auto_training_run(run_directory, seed=seed, map_location="cpu")
    restored_splits = reconstruct_saved_mnist_splits_from_run(
        loaded_run,
        root=root,
        download=download,
    )
    evaluation_context = reconstruct_run_runner_and_test_loader(
        loaded_run,
        restored_splits,
        device=device,
    )
    payload = collect_error_analysis_payload(
        evaluation_context.runner,
        evaluation_context.test_loader,
        max_examples=None,
    )
    dataset_config = loaded_run.saved_mnist_test_config()
    return {
        "seed": seed,
        "image_size": int(dataset_config["image_size"]),
        "class_labels": list(payload.class_labels),
        "confusion_matrix": payload.confusion_matrix.clone(),
        "misclassified_images": payload.misclassified_images.clone(),
        "misclassified_true_labels": payload.misclassified_true_labels.clone(),
        "misclassified_predicted_labels": payload.misclassified_predicted_labels.clone(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Figure S4 error-structure payload from a saved Figure 5b run.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_image_size_sweep run subdirectories.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MNIST data root used to reconstruct the saved test split.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Figure 5b canvas size selecting the source run directory.",
    )
    parser.add_argument(
        "--scaled-image-size",
        type=int,
        default=DEFAULT_SCALED_IMAGE_SIZE,
        help="Figure 5b resized digit size selecting the source run directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional saved seed to extract. Defaults to the first seed recorded in manifest.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional execution device passed to runner reconstruction.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute the saved payload even if error_structure_seed*.pt already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    run_directory = build_figure_5b_run_directory(
        args.artifacts_root.expanduser().resolve(),
        scaled_image_size=args.scaled_image_size,
        image_size=args.image_size,
    )
    seed = resolve_error_structure_seed(run_directory, seed=args.seed)
    output_path = build_error_structure_output_path(run_directory, seed=seed)

    if args.rebuild or not output_path.is_file():
        payload = extract_error_structure(
            run_directory=run_directory,
            root=args.data_root.expanduser().resolve(),
            seed=seed,
            device=args.device,
        )
        torch.save(payload, output_path)

    print(output_path)


if __name__ == "__main__":
    main()
