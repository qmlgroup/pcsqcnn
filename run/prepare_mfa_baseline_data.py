"""Train the MFA ten-class baseline on the article's MNIST protocols.

The translated regime reproduces the Figure 2 data preparation: 1000 images
per class, 16-pixel digits on a 32-pixel canvas, offsets up to 8, 256-image
train batches, and 2000 epochs. The centered regime matches the PCS image-size
sweep's full MNIST training split, with 32-pixel digits, no shifts, and
512-image train batches. Both use three seeds by default and fit PCA anew on
each seed's training images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from qcnn.automation import (
    AutoTrainingConfig,
    MnistDatasetConfig,
    OptimizerConfig,
    OutputConfig,
    SeedConfig,
    TrainingConfig,
    run_mnist_auto_training,
)
from qcnn.model_spec import ModelSpec


def make_config(
    *,
    regime: str,
    data_root: Path,
    artifacts_root: Path,
    directory_name: str | None = None,
    samples_per_class: int | None = None,
    epochs: int = 2000,
    seed_count: int = 3,
    base_seed: int = 0,
    train_batch_size: int | None = None,
    test_batch_size: int = 1600,
    device: str | None = None,
    test_interval: int = 10,
) -> AutoTrainingConfig:
    if regime not in {"translated", "centered"}:
        raise ValueError("regime must be 'translated' or 'centered'.")
    translated = regime == "translated"
    resolved_samples_per_class = (
        1000 if translated and samples_per_class is None else samples_per_class
    )
    resolved_train_batch_size = (
        (256 if translated else 512)
        if train_batch_size is None
        else train_batch_size
    )
    return AutoTrainingConfig(
        dataset=MnistDatasetConfig(
            root=data_root,
            samples_per_class=resolved_samples_per_class,
            image_size=32,
            scaled_image_size=16 if translated else 32,
            max_offset=8 if translated else 0,
            train_batch_size=resolved_train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=0,
            pin_memory=False,
            download=True,
        ),
        model=ModelSpec(
            module="qcnn.baselines.mfa",
            class_name="MFAClassifier",
            constructor_kwargs={"image_size": 32, "num_classes": 10},
        ),
        optimizer=OptimizerConfig(kind="mfa_adam", learning_rate=0.01),
        training=TrainingConfig(
            num_epochs=epochs,
            device=device,
            test_evaluation_interval_epochs=test_interval,
            snapshot_epochs=tuple(
                epoch
                for epoch in sorted({1, 10, *range(25, epochs + 1, 25)})
                if epoch <= epochs
            ),
        ),
        seeds=SeedConfig(base_seed=base_seed, seed_count=seed_count),
        output=OutputConfig(
            root=artifacts_root,
            directory_name=directory_name,
            use_timestamp_dir=directory_name is None,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("translated", "centered"), default="translated")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/mfa"))
    parser.add_argument("--directory-name")
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--test-batch-size", type=int, default=1600)
    parser.add_argument("--test-interval", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = make_config(
        regime=args.regime,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        directory_name=args.directory_name,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        seed_count=args.seed_count,
        base_seed=args.base_seed,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        test_interval=args.test_interval,
        device=args.device,
    )
    print(
        f"MFA {args.regime}: {args.seed_count} seed(s), {args.epochs} epoch(s), "
        f"training images per class={config.dataset.samples_per_class or 'full split'}, "
        f"device={args.device or 'auto'}",
        flush=True,
    )
    epoch_started = time.perf_counter()
    timing_path = (
        Path(config.output.root) / config.output.directory_name / "epoch_timing.jsonl"
        if config.output.directory_name is not None
        else None
    )

    def report_epoch(seed, epoch, train_summary, test_summary, history) -> None:
        nonlocal epoch_started
        del history
        elapsed = time.perf_counter() - epoch_started
        epoch_started = time.perf_counter()
        if timing_path is not None:
            with timing_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "elapsed_seconds": elapsed,
                            "completed_unix_time": time.time(),
                        }
                    )
                    + "\n"
                )
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            test_accuracy = (
                f"{test_summary.metrics['accuracy']:.4f}" if test_summary is not None else "pending"
            )
            print(
                f"seed={seed} epoch={epoch}/{args.epochs} "
                f"train_accuracy={train_summary.metrics['accuracy']:.4f} "
                f"test_accuracy={test_accuracy} elapsed={elapsed:.1f}s",
                flush=True,
            )

    result = run_mnist_auto_training(
        config,
        resume=args.resume,
        save_training_state=True,
        seed_epoch_end_callback=report_epoch,
    )
    print(result.manifest_path)


if __name__ == "__main__":
    main()
