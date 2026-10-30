"""Run ONE compact PCS or QCS/literature model on the main translated-MNIST task.

Default: FRQI sp-QCNN, depths (10,10,1), seeds 0/1/2, 2000 epochs.
--dry-run only prints the configuration. No architecture sweep is implicit.
Short verification runs must use their own --directory-name and --epochs.
The adapted Hur circuits share gates but do not guarantee exact QCS equivariance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import torch

from qcnn.automation import (
    AutoTrainingConfig, MnistDatasetConfig, OptimizerConfig, OutputConfig,
    SeedConfig, TrainingConfig, run_mnist_auto_training,
)
from qcnn.model_spec import ModelSpec


MODELS = ("sp_qcnn", "pcs_local", "hur_x1", "hur_x8", "hur_l4", "hur_l16")


def make_config(
    *, model: str = "sp_qcnn", data_root: Path = Path("data"),
    artifacts_root: Path = Path("artifacts/qcs_baselines"), directory_name: str | None = None,
    epochs: int = 2000, seed_count: int = 3, base_seed: int = 0,
    samples_per_class: int = 1000, train_batch_size: int = 256,
    test_batch_size: int = 1600, device: str | None = None,
) -> AutoTrainingConfig:
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}.")
    if model == "sp_qcnn":
        spec = ModelSpec("qcnn.baselines.sp_qcnn", "SpQCNN", {
            "image_size": 32, "num_classes": 10, "depth": 10,
            "brightness_range": (0.0, 3.141592653589793),
            "angle_init_scale": 6.283185307179586, "dtype": "float32",
        })
    elif model == "pcs_local":
        spec = ModelSpec("qcnn.hybrid", "PCSQCNN", {
            "image_size": 32, "num_classes": 10, "feature_qubits": 2,
            "quantum_layers": 3, "multiplexer_kind": "local_3x3",
            "brightness_range": (0.0, 3.141592653589793),
            "multiplexer_init_scale": 6.283185307179586,
            "shot_budget": None, "reduce_readout_to_feature_distribution": False,
            "use_reduced_fourier_junction": True,
        })
    else:
        spec = ModelSpec("qcnn.hur", "HurQCNN", {
            "image_size": 32, "num_classes": 10,
            "channels": 8 if model == "hur_x8" else 1,
            "filters": {"hur_l4": 4, "hur_l16": 16}.get(model, 1),
            "output_qubits": 4, "angle_init_scale": 6.283185307179586,
        })
    return AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=data_root, image_size=32, scaled_image_size=16, max_offset=8,
            samples_per_class=samples_per_class, train_batch_size=train_batch_size,
            test_batch_size=test_batch_size, num_workers=0, pin_memory=False,
        ),
        model=spec,
        optimizer=OptimizerConfig(kind="adam", learning_rate=0.03, weight_decay=0.0),
        training=TrainingConfig(
            num_epochs=epochs, device=device, test_evaluation_interval_epochs=10,
            # Reduced-precision CUDA products introduce nonunitary blocks and
            # spatial leakage in the local PCS kernel. Keep full float32 math.
            torch_matmul_precision="highest",
            snapshot_epochs=tuple(sorted({1, epochs, *range(25, epochs + 1, 25)})),
        ),
        seeds=SeedConfig(base_seed=base_seed, seed_count=seed_count),
        output=OutputConfig(root=artifacts_root, directory_name=directory_name, use_timestamp_dir=directory_name is None),
    )


def implementation_record(model: str) -> dict:
    """Capture source bytes once, before any seed starts training."""
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "src/qcnn").rglob("*.py"))
    source_files += [root / name for name in ("pyproject.toml", "uv.lock", "run/prepare_qcs_baseline_data.py")]
    return {
        "model": model,
        "purpose": "main translated-MNIST benchmark configuration; overrides recorded in manifest",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
        "sources": {
            "sp_qcnn": ["https://doi.org/10.1103/PhysRevResearch.6.023042",
                        "https://doi.org/10.1007/s42484-026-00397-2"],
            "pcs_local": ["https://arxiv.org/abs/2604.06094",
                          "https://doi.org/10.1007/s11005-022-01578-3"],
        }.get(model, ["https://doi.org/10.1007/s42484-021-00061-x", "https://github.com/takh04/QCNN"]),
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="sp_qcnn")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/qcs_baselines"))
    parser.add_argument("--directory-name")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1600)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true",
                        help="Resume all configured seeds in --directory-name; completed seeds skip training.")
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args(argv))
    resume, dry_run = args.pop("resume"), args.pop("dry_run")
    config = make_config(**args)
    print(json.dumps(asdict(config), indent=2, default=str), flush=True)
    if dry_run:
        return

    def report(seed, epoch, train_summary, test_summary, history):
        del history
        if test_summary is not None:
            print(f"seed={seed} epoch={epoch} train={train_summary.metrics['accuracy']:.4f} "
                  f"test={test_summary.metrics['accuracy']:.4f}", flush=True)

    provenance = implementation_record(args["model"])
    result = run_mnist_auto_training(
        config, resume=resume, save_training_state=True, implementation=provenance,
        seed_epoch_end_callback=report,
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
