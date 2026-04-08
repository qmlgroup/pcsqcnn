from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import torch

from qcnn.article_training import build_canonical_reference_run_directory_name
from qcnn.article_figures import resolve_saved_run_seed
from qcnn import evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test
from qcnn.script_tasks import ManifestTaskContext, ManifestTaskSpec, run_manifest_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_FILENAME = "readout_landscape.pt"

# Canonical scientific defaults for the Figure S3 / supplementary Figure S5
# readout-landscape diagnostic.
#
# The canonical `16on16` source run, its dense snapshot schedule, and the fixed
# Figure 5b architecture are produced upstream by
# `run/prepare_pcsqcnn_image_size_sweep_data.py`. This script intentionally
# does not re-declare `feature_qubits` or `quantum_layers`, because changing
# them requires recomputing that upstream run family.
DEFAULT_REFERENCE_DIRECTORY_NAME = build_canonical_reference_run_directory_name()
DEFAULT_REFERENCE_SEED: int | None = None
DEFAULT_REFERENCE_EPOCHS: tuple[int, ...] = (10, 100, 800)
DEFAULT_ALPHA_BETA_POINTS = 81
DEFAULT_AXIS_LIMIT = 3.0
DEFAULT_SIGMA_SHOT_BUDGET = 128

# Runtime/performance defaults. These resource knobs affect how the evaluation
# is executed, but are not intended to change the scientific target.
DEFAULT_SAMPLE_BATCH_SIZE = 128
DEFAULT_GRID_CHUNK_SIZE = 512

# Artifact/cache contract defaults. These values control the on-disk cache
# layout and compatibility checks; they are not scientific configuration.
DEFAULT_BASIS_MODE = "sample_local_pca_exact_readout_shot_noise_covariance"
DEFAULT_TASK_ROOT_DIRECTORY_NAME = "readout_landscape"
DEFAULT_TASK_RESULT_FILENAME = "result.pt"
DEFAULT_TASK_FORMAT_VERSION = 6


def load_single_seed_from_manifest(run_directory: str | Path, *, seed: int | None = None) -> int:
    return resolve_saved_run_seed(run_directory, seed=seed)


def resolve_default_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_task_root_directory(run_directory: str | Path) -> Path:
    return Path(run_directory) / DEFAULT_TASK_ROOT_DIRECTORY_NAME


def build_task_directory(run_directory: str | Path, *, epoch: int) -> Path:
    return build_task_root_directory(run_directory) / f"epoch{epoch:03d}"


def build_task_manifest(
    *,
    seed: int,
    epoch: int,
    alpha_beta_points: int,
    axis_limit: float,
    sample_batch_size: int,
    grid_chunk_size: int,
    sigma_shot_budget: int,
    basis_mode: str,
    total_samples: int,
) -> dict[str, Any]:
    return {
        "task_type": "readout_landscape",
        "format_version": DEFAULT_TASK_FORMAT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epoch": epoch,
        "alpha_beta_points": alpha_beta_points,
        "axis_limit": axis_limit,
        "sample_batch_size": sample_batch_size,
        "grid_chunk_size": grid_chunk_size,
        "sigma_shot_budget": sigma_shot_budget,
        "basis_mode": basis_mode,
        "total_samples": total_samples,
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
    payload = torch.load(Path(output_directory) / DEFAULT_TASK_RESULT_FILENAME, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Figure S3 task result must deserialize to a mapping.")
    return dict(payload)


def is_compatible_task_cache(
    output_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    alpha_beta_points: int,
    axis_limit: float,
    sample_batch_size: int,
    grid_chunk_size: int,
    sigma_shot_budget: int,
    basis_mode: str,
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
    if manifest.get("task_type") != "readout_landscape":
        return False
    if manifest.get("format_version") != DEFAULT_TASK_FORMAT_VERSION:
        return False
    if manifest.get("seed") != seed or manifest.get("epoch") != epoch:
        return False
    if manifest.get("alpha_beta_points") != alpha_beta_points:
        return False
    if manifest.get("axis_limit") != axis_limit:
        return False
    if manifest.get("sample_batch_size") != sample_batch_size:
        return False
    if manifest.get("grid_chunk_size") != grid_chunk_size:
        return False
    if manifest.get("sigma_shot_budget") != sigma_shot_budget:
        return False
    if manifest.get("basis_mode") != basis_mode:
        return False

    result_path = resolved_directory / DEFAULT_TASK_RESULT_FILENAME
    if not result_path.is_file():
        return False
    try:
        payload = load_task_result(resolved_directory)
    except Exception:
        return False

    valid_count = payload.get("valid_count")
    valid_fraction = payload.get("valid_fraction")
    mean_loss = payload.get("mean_loss")
    pc1_sigma = payload.get("pc1_sigma")
    pc2_sigma = payload.get("pc2_sigma")
    eligible_sample_count = payload.get("eligible_sample_count")
    total_samples = payload.get("total_samples")
    if not isinstance(valid_count, torch.Tensor) or valid_count.shape != (alpha_beta_points, alpha_beta_points):
        return False
    if not isinstance(valid_fraction, torch.Tensor) or valid_fraction.shape != (alpha_beta_points, alpha_beta_points):
        return False
    if not isinstance(mean_loss, torch.Tensor) or mean_loss.shape != (alpha_beta_points, alpha_beta_points):
        return False
    if not isinstance(pc1_sigma, torch.Tensor) or pc1_sigma.shape != (alpha_beta_points,):
        return False
    if not isinstance(pc2_sigma, torch.Tensor) or pc2_sigma.shape != (alpha_beta_points,):
        return False
    if not isinstance(total_samples, int) or total_samples <= 0:
        return False
    if not isinstance(eligible_sample_count, int) or eligible_sample_count < 0 or eligible_sample_count > total_samples:
        return False
    if not torch.isfinite(valid_fraction).all():
        return False
    if not ((valid_fraction >= 0.0).all() and (valid_fraction <= 1.0).all()):
        return False
    if not torch.allclose(
        valid_fraction,
        valid_count.to(dtype=valid_fraction.dtype) / float(total_samples),
        atol=1e-6,
        rtol=1e-6,
    ):
        return False
    return True


def cleanup_incompatible_task_caches(
    *,
    run_directory: Path,
    seed: int,
    epochs: Sequence[int],
    alpha_beta_points: int,
    axis_limit: float,
    sample_batch_size: int,
    grid_chunk_size: int,
    sigma_shot_budget: int,
    basis_mode: str,
) -> None:
    for epoch in epochs:
        task_directory = build_task_directory(run_directory, epoch=epoch)
        if not task_directory.exists():
            continue
        if not is_compatible_task_cache(
            task_directory,
            seed=seed,
            epoch=epoch,
            alpha_beta_points=alpha_beta_points,
            axis_limit=axis_limit,
            sample_batch_size=sample_batch_size,
            grid_chunk_size=grid_chunk_size,
            sigma_shot_budget=sigma_shot_budget,
            basis_mode=basis_mode,
        ):
            shutil.rmtree(task_directory)


def build_task_spec(
    *,
    run_directory: Path,
    root: Path,
    seed: int,
    device: str,
    alpha_beta_points: int,
    axis_limit: float,
    sample_batch_size: int,
    grid_chunk_size: int,
    sigma_shot_budget: int,
    basis_mode: str,
    epoch: int,
    download: bool,
) -> ManifestTaskSpec:
    output_directory = build_task_directory(run_directory, epoch=epoch)

    def run(task_context: ManifestTaskContext) -> None:
        def on_batch_progress(completed_batches: int, total_batches: int) -> None:
            if completed_batches == 1:
                task_context.show_primary_progress(
                    description=f"Batches (epoch {epoch})",
                    total=total_batches,
                    completed=0,
                )
            task_context.update_primary_progress(
                description=f"Batches (epoch {epoch})",
                total=total_batches,
                completed=completed_batches,
            )

        evaluation = evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
            run_directory,
            seed=seed,
            epoch=epoch,
            root=root,
            device=device,
            alpha_beta_points=alpha_beta_points,
            axis_limit=axis_limit,
            sample_batch_size=sample_batch_size,
            grid_chunk_size=grid_chunk_size,
            sigma_shot_budget=sigma_shot_budget,
            download=download,
            batch_progress_callback=on_batch_progress,
        )
        save_task_result(
            output_directory,
            payload={
                "epoch": epoch,
                "pc1_sigma": evaluation.alpha.clone(),
                "pc2_sigma": evaluation.beta.clone(),
                "mean_loss": evaluation.mean_loss.clone(),
                "valid_count": evaluation.valid_count.clone(),
                "valid_fraction": evaluation.valid_fraction.clone(),
                "total_samples": evaluation.total_samples,
                "eligible_sample_count": evaluation.eligible_sample_count,
            },
            manifest=build_task_manifest(
                seed=seed,
                epoch=epoch,
                alpha_beta_points=alpha_beta_points,
                axis_limit=axis_limit,
                sample_batch_size=sample_batch_size,
                grid_chunk_size=grid_chunk_size,
                sigma_shot_budget=sigma_shot_budget,
                basis_mode=basis_mode,
                total_samples=evaluation.total_samples,
            ),
        )

    return ManifestTaskSpec(
        name=f"epoch {epoch}",
        output_directory=output_directory,
        run=run,
    )


def assemble_readout_landscape_payload(
    *,
    run_directory: str | Path,
    seed: int,
    epochs: Sequence[int],
    alpha_beta_points: int,
    sigma_shot_budget: int,
    basis_mode: str,
) -> dict[str, Any]:
    resolved_directory = Path(run_directory)
    pc1_sigma: torch.Tensor | None = None
    pc2_sigma: torch.Tensor | None = None
    total_samples: int | None = None
    evaluations: list[dict[str, Any]] = []

    for epoch in epochs:
        task_payload = load_task_result(build_task_directory(resolved_directory, epoch=epoch))
        task_pc1_sigma = task_payload.get("pc1_sigma")
        task_pc2_sigma = task_payload.get("pc2_sigma")
        task_mean_loss = task_payload.get("mean_loss")
        task_valid_count = task_payload.get("valid_count")
        task_valid_fraction = task_payload.get("valid_fraction")
        task_total_samples = task_payload.get("total_samples")
        task_eligible_sample_count = task_payload.get("eligible_sample_count")
        if not isinstance(task_pc1_sigma, torch.Tensor) or task_pc1_sigma.ndim != 1 or task_pc1_sigma.numel() != alpha_beta_points:
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing a valid pc1_sigma grid.")
        if not isinstance(task_pc2_sigma, torch.Tensor) or task_pc2_sigma.ndim != 1 or task_pc2_sigma.numel() != alpha_beta_points:
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing a valid pc2_sigma grid.")
        if not isinstance(task_mean_loss, torch.Tensor) or task_mean_loss.shape != (alpha_beta_points, alpha_beta_points):
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing a valid mean_loss tensor.")
        if not isinstance(task_valid_count, torch.Tensor) or task_valid_count.shape != (alpha_beta_points, alpha_beta_points):
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing a valid valid_count tensor.")
        if not isinstance(task_valid_fraction, torch.Tensor) or task_valid_fraction.shape != (alpha_beta_points, alpha_beta_points):
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing a valid valid_fraction tensor.")
        if not isinstance(task_total_samples, int) or task_total_samples <= 0:
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing total_samples.")
        if (
            not isinstance(task_eligible_sample_count, int)
            or task_eligible_sample_count < 0
            or task_eligible_sample_count > task_total_samples
        ):
            raise ValueError(f"Cached Figure S3 task for epoch={epoch} is missing eligible_sample_count.")

        if pc1_sigma is None:
            pc1_sigma = task_pc1_sigma.clone()
            pc2_sigma = task_pc2_sigma.clone()
            total_samples = task_total_samples
        else:
            if not torch.equal(pc1_sigma, task_pc1_sigma):
                raise ValueError("All Figure S3 tasks must use the same pc1_sigma grid.")
            if not torch.equal(pc2_sigma, task_pc2_sigma):
                raise ValueError("All Figure S3 tasks must use the same pc2_sigma grid.")
            if total_samples != task_total_samples:
                raise ValueError("All Figure S3 tasks must use the same test split size.")

        evaluations.append(
            {
                "epoch": epoch,
                "mean_loss": task_mean_loss.clone(),
                "valid_count": task_valid_count.clone(),
                "valid_fraction": task_valid_fraction.clone(),
                "eligible_sample_count": task_eligible_sample_count,
            }
        )

    if pc1_sigma is None or pc2_sigma is None or total_samples is None:
        raise ValueError("At least one epoch must be requested for Figure S3.")

    return {
        "seed": seed,
        "epochs": list(epochs),
        "pc1_sigma": pc1_sigma,
        "pc2_sigma": pc2_sigma,
        "sigma_shot_budget": sigma_shot_budget,
        "basis_mode": basis_mode,
        "total_samples": total_samples,
        "evaluations": evaluations,
    }


def evaluate_readout_landscape(
    *,
    run_directory: str | Path,
    root: str | Path,
    epochs: Sequence[int] = DEFAULT_REFERENCE_EPOCHS,
    seed: int | None = DEFAULT_REFERENCE_SEED,
    device: str | None = None,
    alpha_beta_points: int = DEFAULT_ALPHA_BETA_POINTS,
    axis_limit: float = DEFAULT_AXIS_LIMIT,
    sample_batch_size: int = DEFAULT_SAMPLE_BATCH_SIZE,
    grid_chunk_size: int = DEFAULT_GRID_CHUNK_SIZE,
    sigma_shot_budget: int = DEFAULT_SIGMA_SHOT_BUDGET,
    basis_mode: str = DEFAULT_BASIS_MODE,
    download: bool = True,
    rebuild: bool = False,
) -> dict[str, Any]:
    if not epochs:
        raise ValueError("epochs must not be empty.")
    if axis_limit <= 0.0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}.")
    if sigma_shot_budget <= 0:
        raise ValueError(f"sigma_shot_budget must be positive, got {sigma_shot_budget}.")

    resolved_directory = Path(run_directory)
    seed = load_single_seed_from_manifest(resolved_directory, seed=seed)
    resolved_device = resolve_default_device(device)
    cleanup_incompatible_task_caches(
        run_directory=resolved_directory,
        seed=seed,
        epochs=epochs,
        alpha_beta_points=alpha_beta_points,
        axis_limit=axis_limit,
        sample_batch_size=sample_batch_size,
        grid_chunk_size=grid_chunk_size,
        sigma_shot_budget=sigma_shot_budget,
        basis_mode=basis_mode,
    )
    task_specs = tuple(
        build_task_spec(
            run_directory=resolved_directory,
            root=Path(root),
            seed=seed,
            device=resolved_device,
            alpha_beta_points=alpha_beta_points,
            axis_limit=axis_limit,
            sample_batch_size=sample_batch_size,
            grid_chunk_size=grid_chunk_size,
            sigma_shot_budget=sigma_shot_budget,
            basis_mode=basis_mode,
            epoch=epoch,
            download=download,
        )
        for epoch in epochs
    )
    run_manifest_tasks(task_specs, rebuild=rebuild)
    return assemble_readout_landscape_payload(
        run_directory=resolved_directory,
        seed=seed,
        epochs=epochs,
        alpha_beta_points=alpha_beta_points,
        sigma_shot_budget=sigma_shot_budget,
        basis_mode=basis_mode,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Figure S3 readout-probability landscapes for the reference PCS-QCNN run.",
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
        help="Optional execution device for reevaluation; defaults to 'cuda' when available, otherwise 'cpu'.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=DEFAULT_SAMPLE_BATCH_SIZE,
        help="Number of test samples processed per exact-quantum batch.",
    )
    parser.add_argument(
        "--grid-chunk-size",
        type=int,
        default=DEFAULT_GRID_CHUNK_SIZE,
        help="Number of (alpha, beta) grid points processed per GPU chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_REFERENCE_SEED,
        help="Optional saved seed to reevaluate. Defaults to the first seed recorded in manifest.json.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild per-epoch Figure S3 caches even if their manifest.json already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_directory = args.artifacts_root.expanduser().resolve() / DEFAULT_REFERENCE_DIRECTORY_NAME
    payload = evaluate_readout_landscape(
        run_directory=run_directory,
        root=args.data_root.expanduser().resolve(),
        epochs=DEFAULT_REFERENCE_EPOCHS,
        seed=args.seed,
        device=args.device,
        alpha_beta_points=DEFAULT_ALPHA_BETA_POINTS,
        axis_limit=DEFAULT_AXIS_LIMIT,
        sample_batch_size=args.sample_batch_size,
        grid_chunk_size=args.grid_chunk_size,
        sigma_shot_budget=DEFAULT_SIGMA_SHOT_BUDGET,
        rebuild=args.rebuild,
    )
    output_path = run_directory / DEFAULT_OUTPUT_FILENAME
    torch.save(payload, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
