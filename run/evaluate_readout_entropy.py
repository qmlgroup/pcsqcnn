from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import torch

from qcnn.article_training import (
    DEFAULT_S2_ENTROPY_SHOT_BUDGETS,
    DEFAULT_S2_REFERENCE_EPOCHS,
    build_canonical_reference_run_directory_name,
)
from qcnn.article_figures import resolve_saved_run_seed
from qcnn import evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test
from qcnn.script_tasks import ManifestTaskContext, ManifestTaskSpec, run_manifest_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_REFERENCE_DIRECTORY_NAME = build_canonical_reference_run_directory_name()
DEFAULT_OUTPUT_FILENAME = "readout_entropy.pt"
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = DEFAULT_S2_REFERENCE_EPOCHS
DEFAULT_SHOT_BUDGETS: tuple[int, ...] = DEFAULT_S2_ENTROPY_SHOT_BUDGETS
DEFAULT_BATCH_SIZE = 128
DEFAULT_TASK_ROOT_DIRECTORY_NAME = "readout_entropy"
DEFAULT_TASK_RESULT_FILENAME = "result.pt"
DEFAULT_TASK_FORMAT_VERSION = 1


def load_single_seed_from_manifest(run_directory: str | Path, *, seed: int | None = None) -> int:
    return resolve_saved_run_seed(run_directory, seed=seed)


def resolve_default_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_task_root_directory(run_directory: str | Path) -> Path:
    return Path(run_directory) / DEFAULT_TASK_ROOT_DIRECTORY_NAME


def build_task_directory(run_directory: str | Path, *, epoch: int, shot_budget: int) -> Path:
    return build_task_root_directory(run_directory) / f"epoch{epoch:03d}_shots{shot_budget}"


def build_task_manifest(
    *,
    seed: int,
    epoch: int,
    shot_budget: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "task_type": "readout_entropy",
        "format_version": DEFAULT_TASK_FORMAT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epoch": epoch,
        "shot_budget": shot_budget,
        "batch_size": batch_size,
        "result_filename": DEFAULT_TASK_RESULT_FILENAME,
    }


def save_task_result(
    output_directory: str | Path,
    *,
    payload: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    resolved_directory = Path(output_directory)
    resolved_directory.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved_directory / DEFAULT_TASK_RESULT_FILENAME)
    (resolved_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_task_result(output_directory: str | Path) -> dict[str, Any]:
    payload = torch.load(
        Path(output_directory) / DEFAULT_TASK_RESULT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Figure S2b task result must deserialize to a mapping.")
    return dict(payload)


def is_compatible_task_cache(
    output_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budget: int,
    batch_size: int,
) -> bool:
    resolved_directory = Path(output_directory)
    manifest_path = resolved_directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(manifest, dict):
        return False
    if manifest.get("task_type") != "readout_entropy":
        return False
    if manifest.get("format_version") != DEFAULT_TASK_FORMAT_VERSION:
        return False
    if manifest.get("seed") != seed or manifest.get("epoch") != epoch:
        return False
    if manifest.get("shot_budget") != shot_budget:
        return False
    if manifest.get("batch_size") != batch_size:
        return False
    return (resolved_directory / DEFAULT_TASK_RESULT_FILENAME).is_file()


def clear_incompatible_task_caches(
    *,
    run_directory: Path,
    seed: int,
    epochs: Sequence[int],
    shot_budgets: Sequence[int],
    batch_size: int,
) -> None:
    for epoch in epochs:
        for shot_budget in shot_budgets:
            task_directory = build_task_directory(run_directory, epoch=epoch, shot_budget=shot_budget)
            if not task_directory.exists():
                continue
            if not is_compatible_task_cache(
                task_directory,
                seed=seed,
                epoch=epoch,
                shot_budget=shot_budget,
                batch_size=batch_size,
            ):
                shutil.rmtree(task_directory)


def build_task_spec(
    *,
    run_directory: Path,
    root: Path,
    seed: int,
    epoch: int,
    shot_budget: int,
    batch_size: int,
    device: str,
    download: bool,
) -> ManifestTaskSpec:
    output_directory = build_task_directory(run_directory, epoch=epoch, shot_budget=shot_budget)

    def run(task_context: ManifestTaskContext) -> None:
        def on_batch_progress(completed_batches: int, total_batches: int) -> None:
            if completed_batches == 1:
                task_context.show_primary_progress(
                    description=f"Batches ({epoch}, {shot_budget} shots)",
                    total=total_batches,
                    completed=0,
                )
            task_context.update_primary_progress(
                description=f"Batches ({epoch}, {shot_budget} shots)",
                total=total_batches,
                completed=completed_batches,
            )

        evaluation = evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test(
            run_directory,
            seed=seed,
            epoch=epoch,
            shot_budget=shot_budget,
            root=root,
            device=device,
            batch_size=batch_size,
            download=download,
            batch_progress_callback=on_batch_progress,
        )
        save_task_result(
            output_directory,
            payload={
                "seed": seed,
                "epoch": epoch,
                "shot_budget": shot_budget,
                "entropy": evaluation.entropy.clone(),
            },
            manifest=build_task_manifest(
                seed=seed,
                epoch=epoch,
                shot_budget=shot_budget,
                batch_size=batch_size,
            ),
        )

    return ManifestTaskSpec(
        name=f"epoch {epoch}, {shot_budget} shots",
        output_directory=output_directory,
        run=run,
    )


def assemble_readout_entropy_payload(
    *,
    run_directory: str | Path,
    seed: int,
    epochs: Sequence[int],
    shot_budgets: Sequence[int],
) -> dict[str, Any]:
    resolved_directory = Path(run_directory)
    evaluations: list[dict[str, Any]] = []

    for epoch in epochs:
        for shot_budget in shot_budgets:
            task_payload = load_task_result(
                build_task_directory(resolved_directory, epoch=epoch, shot_budget=shot_budget)
            )
            entropy = task_payload.get("entropy")
            if not isinstance(entropy, torch.Tensor):
                raise ValueError(
                    f"Cached Figure S2b task for epoch={epoch}, shot_budget={shot_budget} "
                    "is missing an entropy tensor."
                )
            evaluations.append(
                {
                    "epoch": epoch,
                    "shot_budget": shot_budget,
                    "entropy": entropy.clone(),
                }
            )

    return {
        "seed": seed,
        "epochs": list(epochs),
        "shot_budgets": list(shot_budgets),
        "evaluations": evaluations,
    }


def evaluate_readout_entropy(
    *,
    run_directory: str | Path,
    root: str | Path,
    epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
    shot_budgets: Sequence[int] = DEFAULT_SHOT_BUDGETS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = None,
    device: str | None = None,
    download: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    resolved_directory = Path(run_directory)
    seed = load_single_seed_from_manifest(resolved_directory, seed=seed)
    resolved_device = resolve_default_device(device)
    clear_incompatible_task_caches(
        run_directory=resolved_directory,
        seed=seed,
        epochs=epochs,
        shot_budgets=shot_budgets,
        batch_size=batch_size,
    )
    task_specs = tuple(
        build_task_spec(
            run_directory=resolved_directory,
            root=Path(root),
            seed=seed,
            epoch=epoch,
            shot_budget=shot_budget,
            batch_size=batch_size,
            device=resolved_device,
            download=download,
        )
        for epoch in epochs
        for shot_budget in shot_budgets
    )
    run_manifest_tasks(task_specs, rebuild=rebuild)
    return assemble_readout_entropy_payload(
        run_directory=resolved_directory,
        seed=seed,
        epochs=epochs,
        shot_budgets=shot_budgets,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Figure S2b sampled-readout entropy for the canonical reference run.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Directory containing the pcsqcnn_image_size_sweep artifacts.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MNIST data root used to reconstruct the saved test split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device for reevaluation. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Test batch size used during entropy extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional saved seed to reevaluate. Defaults to the first seed recorded in manifest.json.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild diagnostic task caches even if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_directory = (
        args.artifacts_root.expanduser().resolve()
        / DEFAULT_REFERENCE_DIRECTORY_NAME
    )
    payload = evaluate_readout_entropy(
        run_directory=run_directory,
        root=args.data_root.expanduser().resolve(),
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        rebuild=args.rebuild,
    )
    output_path = run_directory / DEFAULT_OUTPUT_FILENAME
    torch.save(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
