"""Automatic MNIST training orchestration for importable qcnn-compatible models."""

from __future__ import annotations

import io
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from qcnn.data import PreparedMnistSplits, prepare_mnist_splits
from qcnn.model import EvaluationSummary, ImageClassifierRunner, ProgressFactory, TrainingHistory
from qcnn.model_stats import format_trainable_parameter_stats_line
from qcnn.model_spec import ModelSpec, instantiate_model
from qcnn.serialization import load_model_checkpoint, save_model_checkpoint
from qcnn.script_tasks import ManifestTaskContext, ManifestTaskSpec, run_manifest_tasks
from qcnn.statistics import StatisticCollector


@dataclass(frozen=True)
class MnistDatasetConfig:
    """Dataset-preparation and loader settings for automated MNIST training.

    Attributes:
        root: Directory used by ``prepare_mnist_splits`` for MNIST files.
        samples_per_class: Number of balanced training samples to keep per
            class. When ``None``, the full standard MNIST train split is used.
        image_size: Final square canvas side length after preprocessing.
        scaled_image_size: Optional intermediate side length used by the
            bilinear resize step before placing digits on the final canvas.
            When ``None``, ``image_size`` is used.
        max_offset: Maximum integer translation magnitude applied
            independently along both axes after centering the resized image on
            the final canvas.
        test_max_offset: Optional translation magnitude of the test split
            only; ``None`` reuses ``max_offset``.
        train_batch_size: Batch size used by the training loader.
        test_batch_size: Batch size used by the evaluation loader.
        num_workers: Worker count passed into both loaders.
        pin_memory: Whether to enable pinned host memory for both loaders.
        download: Forwarded to ``prepare_mnist_splits``.
    """

    root: str | Path
    samples_per_class: int | None = 1000
    image_size: int = 16
    scaled_image_size: int | None = None
    max_offset: int = 0
    test_max_offset: int | None = None
    train_batch_size: int = 64
    test_batch_size: int = 16000
    num_workers: int = 0
    pin_memory: bool = False
    download: bool = True


def _default_model_spec() -> ModelSpec:
    return ModelSpec(
        module="qcnn.hybrid",
        class_name="PCSQCNN",
        constructor_kwargs={
            "image_size": 16,
            "num_classes": 10,
            "feature_qubits": 1,
            "quantum_layers": 1,
            "brightness_range": (0.0, math.pi),
            "shot_budget": None,
            "use_reduced_fourier_junction": True,
            "multiplexer_init_scale": 2.0 * math.pi,
        },
    )


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for automated training.

    Attributes:
        kind: Optimizer identifier. Supported values are ``"adam"`` and
            ``"sgd"``.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight-decay coefficient.
        momentum: Momentum used by SGD. Ignored by Adam.
    """

    kind: str = "adam"
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    momentum: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings for automated runs.

    Attributes:
        num_epochs: Number of epochs to train.
        set_to_none: Forwarded to ``ImageClassifierRunner.fit`` zeroing.
        snapshot_epochs: Explicit epochs that trigger checkpoint snapshots.
        device: Optional execution device for the training run.
        torch_matmul_precision: Optional precision mode forwarded to Torch.
        test_requires_grad: Whether epoch-end test evaluation should preserve
            the autograd graph for gradient-based collectors.
        test_evaluation_interval_epochs: Frequency of epoch-end test
            evaluation. The final epoch is always evaluated.
        init_checkpoint: Optional model checkpoint whose weights initialize
            every seed; ``{seed}`` in the path is replaced by the seed. Its
            model specification must equal the configured one. Only the
            weights are loaded: the optimizer starts fresh (warm restart),
            and epochs are counted from one again.
    """

    num_epochs: int = 100
    set_to_none: bool = True
    snapshot_epochs: tuple[int, ...] = ()
    device: str | torch.device | None = None
    torch_matmul_precision: str | None = "high"
    test_requires_grad: bool = False
    test_evaluation_interval_epochs: int = 10
    init_checkpoint: str | None = None


@dataclass(frozen=True)
class SeedConfig:
    """Sequential seed range used for repeated runs.

    Attributes:
        base_seed: First seed used by the automated run.
        seed_count: Number of sequential seeds to execute.
    """

    base_seed: int = 0
    seed_count: int = 1


@dataclass(frozen=True)
class OutputConfig:
    """Output-directory naming and root settings for automated runs.

    Attributes:
        root: Parent directory under which one dedicated run directory is
            created.
        directory_name: Explicit output directory name. When omitted, the
            directory name is generated automatically.
        use_timestamp_dir: Whether automatic names should use a timestamp.
    """

    root: str | Path = "artifacts"
    directory_name: str | None = None
    use_timestamp_dir: bool = True


@dataclass(frozen=True)
class AutoTrainingConfig:
    """Top-level configuration for ``run_mnist_auto_training(...)``.

    Attributes:
        dataset: MNIST preprocessing and loader settings.
        model: Importable model specification.
        optimizer: Optimizer settings.
        training: Training-loop settings.
        seeds: Sequential seed range for repeated runs.
        output: Output-directory naming settings.
    """

    dataset: MnistDatasetConfig
    model: ModelSpec = field(default_factory=_default_model_spec)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass(frozen=True)
class SeedTrainingArtifacts:
    """Filesystem artifacts created for one seed run.

    Attributes:
        result_path: Path to the per-seed ``result_seed*.pt`` payload.
        final_checkpoint_path: Path to the unconditional final checkpoint.
        snapshot_paths: Mapping from epoch number to snapshot checkpoint path.
    """

    result_path: Path
    final_checkpoint_path: Path
    snapshot_paths: dict[int, Path]


@dataclass(frozen=True)
class SeedTrainingRun:
    """In-memory result bundle for one automated seed run.

    Attributes:
        seed: Executed seed for this run.
        runner: Runner used for training and evaluation.
        train_loader: Seeded training loader used for the run.
        test_loader: Evaluation loader used for the run.
        history: Epoch-level training history for the run.
        final_train_summary: Final train summary from the last epoch.
        final_test_summary: Final test summary from the last epoch.
        collector_states: Exported state from all loss/metric collectors.
        artifacts: Files written for this seed.
        resolved_config: Plain resolved config bundle including the seed.
    """

    seed: int
    runner: ImageClassifierRunner
    train_loader: DataLoader
    test_loader: DataLoader
    history: TrainingHistory
    final_train_summary: EvaluationSummary
    final_test_summary: EvaluationSummary
    collector_states: dict[str, Any]
    artifacts: SeedTrainingArtifacts
    resolved_config: dict[str, Any]


@dataclass(frozen=True)
class AutoTrainingResult:
    """Return value of ``run_mnist_auto_training(...)``.

    Attributes:
        config: Original high-level auto-training config.
        output_directory: Dedicated directory created for this function call.
        manifest_path: Path to the generated ``manifest.json`` file.
        device: Resolved execution device used by the run.
        seeds: Concrete sequential seed list that was executed.
        runs: Per-seed in-memory result bundles.
    """

    config: AutoTrainingConfig
    output_directory: Path
    manifest_path: Path
    device: torch.device
    seeds: list[int]
    runs: list[SeedTrainingRun]


SeedLifecycleCallback = Callable[[int, int, int], None]
SeedEpochEndCallback = Callable[[int, int, EvaluationSummary, EvaluationSummary | None, TrainingHistory], None]


def resolve_auto_training_output_directory(config: AutoTrainingConfig) -> Path:
    """Resolve the explicit output directory required by manifest-managed runs."""

    directory_name = config.output.directory_name
    if directory_name is None:
        raise ValueError("Auto-training manifest tasks require output.directory_name to be set explicitly.")
    if not directory_name:
        raise ValueError("output.directory_name must be a non-empty string when provided.")
    return Path(config.output.root).expanduser().resolve() / directory_name


def build_auto_training_manifest_task(training_config: AutoTrainingConfig) -> ManifestTaskSpec:
    """Wrap one named auto-training config as a manifest-managed task."""

    output_directory = resolve_auto_training_output_directory(training_config)
    task_name = training_config.output.directory_name or training_config.model.class_name
    current_num_epochs = training_config.training.num_epochs

    def run(task_context: ManifestTaskContext) -> None:
        current_seed: int | None = None
        current_seed_index: int | None = None
        current_seed_count: int | None = None
        task_epoch_timer_start: float | None = None
        last_test_summary: EvaluationSummary | None = None

        def on_seed_start(seed: int, seed_index: int, seed_count: int) -> None:
            nonlocal current_seed, current_seed_index, current_seed_count, task_epoch_timer_start, last_test_summary
            current_seed = seed
            current_seed_index = seed_index
            current_seed_count = seed_count
            if task_epoch_timer_start is None:
                task_epoch_timer_start = time.perf_counter()
            last_test_summary = None
            task_context.show_primary_progress(
                description=_format_primary_progress_description(),
                total=seed_count,
                completed=seed_index - 1,
            )
            task_context.show_secondary_progress(
                description=_format_secondary_progress_description(),
                total=current_num_epochs,
                completed=0,
            )
            task_context.show_status_line(_format_epoch_status_line())

        def on_seed_end(seed: int, seed_index: int, seed_count: int) -> None:
            nonlocal current_seed, current_seed_index, current_seed_count
            del seed, seed_count
            current_seed = None
            current_seed_index = None
            current_seed_count = None
            task_context.update_primary_progress(
                description=_format_primary_progress_description(0.0),
                completed=seed_index,
            )
            task_context.update_secondary_progress(description=_format_secondary_progress_description(0.0))
            task_context.hide_status_line()
            task_context.hide_secondary_progress()

        def epoch_progress_factory(epoch_range: range) -> Iterable[int]:
            if current_seed is None:
                raise RuntimeError("epoch_progress_factory requires an active seed context.")
            task_context.update_secondary_progress(
                description=_format_secondary_progress_description(),
                total=len(epoch_range),
                completed=0,
            )
            for epoch in epoch_range:
                yield epoch
                task_context.advance_secondary_progress()

        def on_seed_epoch_end(
            seed: int,
            epoch: int,
            train_summary: EvaluationSummary,
            test_summary: EvaluationSummary | None,
            history: TrainingHistory,
        ) -> None:
            nonlocal last_test_summary
            del seed, history
            if (
                current_seed_index is None
                or current_seed_count is None
                or task_epoch_timer_start is None
            ):
                raise RuntimeError("ETA computation requires active seed timing state.")
            if test_summary is not None:
                last_test_summary = test_summary
            completed_total_epochs = (current_seed_index - 1) * current_num_epochs + epoch
            elapsed_task_seconds = time.perf_counter() - task_epoch_timer_start
            avg_epoch_seconds = elapsed_task_seconds / completed_total_epochs
            remaining_current_seed_epochs = current_num_epochs - epoch
            remaining_total_task_epochs = (
                remaining_current_seed_epochs + (current_seed_count - current_seed_index) * current_num_epochs
            )
            task_context.update_primary_progress(
                description=_format_primary_progress_description(avg_epoch_seconds * remaining_total_task_epochs)
            )
            task_context.update_secondary_progress(
                description=_format_secondary_progress_description(avg_epoch_seconds * remaining_current_seed_epochs)
            )
            task_context.update_status_line(
                _format_epoch_status_line(
                    train_summary=train_summary,
                    test_summary=last_test_summary,
                )
            )

        run_mnist_auto_training(
            training_config,
            progress_factory=epoch_progress_factory,
            seed_start_callback=on_seed_start,
            seed_end_callback=on_seed_end,
            seed_epoch_end_callback=on_seed_epoch_end,
        )

    return ManifestTaskSpec(
        name=task_name,
        output_directory=output_directory,
        run=run,
    )


def run_auto_training_manifest_tasks(
    training_configs: Sequence[AutoTrainingConfig],
    *,
    rebuild: bool = False,
) -> None:
    """Run one or more explicit-directory training configs through manifest tasks."""

    tasks = tuple(build_auto_training_manifest_task(training_config) for training_config in training_configs)
    run_manifest_tasks(tasks, rebuild=rebuild)


def run_mnist_auto_training(
    config: AutoTrainingConfig,
    *,
    resume: bool = False,
    save_training_state: bool = False,
    implementation: Mapping[str, Any] | None = None,
    collectors: tuple[StatisticCollector, ...] | list[StatisticCollector] = (),
    progress_factory: ProgressFactory | None = None,
    seed_start_callback: SeedLifecycleCallback | None = None,
    seed_end_callback: SeedLifecycleCallback | None = None,
    seed_epoch_end_callback: SeedEpochEndCallback | None = None,
) -> AutoTrainingResult:
    """Run deterministic automated MNIST training for importable models.

    Args:
        config: High-level dataset/model/training/output configuration.
        resume: Restore each seed from its final checkpoint or latest snapshot
            in an existing explicit output directory. Unstarted seeds train
            normally; completed seeds do not repeat any epochs.
        save_training_state: Include optimizer, RNG, data fingerprints and
            history in checkpoints. Required for ``resume``; MFA enables it.
        implementation: Optional JSON-compatible source/version record. Saved
            before training, checked on resume, and bound to checkpoints by
            its SHA-256 digest. Pass the current record on every invocation.
        collectors: Optional additional metric collectors saved alongside the
            default loss and accuracy collectors.
        progress_factory: Optional epoch-iterator wrapper, for example a
            notebook progress bar.
        seed_start_callback: Optional callback invoked immediately before each
            seed run starts. Receives ``(seed, seed_index, seed_count)`` with a
            1-based ``seed_index``.
        seed_end_callback: Optional callback invoked immediately after each
            seed run completes successfully. Receives ``(seed, seed_index,
            seed_count)`` with a 1-based ``seed_index``.
        seed_epoch_end_callback: Optional callback invoked after each recorded
            epoch of each seed run. Receives ``(seed, epoch, train_summary,
            test_summary, history)`` where ``test_summary`` is ``None`` on
            epochs without test evaluation.

    Returns:
        ``AutoTrainingResult`` describing the created output directory and the
        per-seed in-memory run bundles.

    Notes:
        The function always creates one dedicated output directory, saves
        ``manifest.json`` plus per-seed result/checkpoint files, and runs seeds
        sequentially as ``base_seed .. base_seed + seed_count - 1``.
    """

    snapshot_epochs = _validate_config(config)
    if resume and not save_training_state:
        raise ValueError("Resume requires save_training_state=True.")
    if resume and collectors:
        raise ValueError("Resume currently supports the built-in loss and accuracy collectors only.")
    resolved_device = _resolve_execution_device(config.training.device)
    output_directory = _prepare_output_directory(config.output, resume=resume)
    implementation_sha256 = _prepare_implementation_record(
        output_directory, implementation=implementation, resume=resume,
    )

    if config.training.torch_matmul_precision is not None:
        torch.set_float32_matmul_precision(config.training.torch_matmul_precision)

    seeds = list(range(config.seeds.base_seed, config.seeds.base_seed + config.seeds.seed_count))
    runs: list[SeedTrainingRun] = []
    seed_count = len(seeds)
    for seed_index, seed in enumerate(seeds, start=1):
        if seed_start_callback is not None:
            seed_start_callback(seed, seed_index, seed_count)
        run = _run_single_seed_training(
            config=config,
            seed=seed,
            collectors=tuple(collectors),
            snapshot_epochs=snapshot_epochs,
            device=resolved_device,
            output_directory=output_directory,
            resume=resume,
            save_training_state=save_training_state,
            implementation_sha256=implementation_sha256,
            progress_factory=progress_factory,
            seed_epoch_end_callback=seed_epoch_end_callback,
        )
        runs.append(run)
        if seed_end_callback is not None:
            seed_end_callback(seed, seed_index, seed_count)

    manifest = {
        "output_directory": str(output_directory),
        "device": str(resolved_device),
        "resolved_config": _resolved_global_config(config),
        "seeds": seeds,
        "runs": [_seed_manifest_entry(run, output_directory=output_directory) for run in runs],
    }
    if implementation_sha256 is not None:
        manifest["implementation_sha256"] = implementation_sha256
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return AutoTrainingResult(
        config=config,
        output_directory=output_directory,
        manifest_path=manifest_path,
        device=resolved_device,
        seeds=seeds,
        runs=runs,
    )


def _run_single_seed_training(
    *,
    config: AutoTrainingConfig,
    seed: int,
    collectors: tuple[StatisticCollector, ...],
    snapshot_epochs: tuple[int, ...],
    device: torch.device,
    output_directory: Path,
    resume: bool,
    save_training_state: bool,
    implementation_sha256: str | None,
    progress_factory: ProgressFactory | None,
    seed_epoch_end_callback: SeedEpochEndCallback | None,
) -> SeedTrainingRun:
    _set_global_seeds(seed)
    # A separate test translation is forwarded only when configured.
    test_split_kwargs = (
        {} if config.dataset.test_max_offset is None else {"test_max_offset": config.dataset.test_max_offset}
    )
    splits = prepare_mnist_splits(
        root=config.dataset.root,
        samples_per_class=config.dataset.samples_per_class,
        image_size=config.dataset.image_size,
        scaled_image_size=config.dataset.scaled_image_size,
        max_offset=config.dataset.max_offset,
        seed=seed,
        download=config.dataset.download,
        **test_split_kwargs,
    )
    data_fingerprints = (
        {
            "train": _fingerprint_image_dataset(splits.train),
            "test": _fingerprint_image_dataset(splits.test),
        }
        if save_training_state
        else {}
    )
    if device.type == "cuda":
        splits = splits.to(device)
    loader_num_workers, loader_pin_memory = _loader_settings_for_device(
        device=device,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
    )
    train_loader, test_loader = _build_mnist_loaders(
        splits,
        train_batch_size=config.dataset.train_batch_size,
        test_batch_size=config.dataset.test_batch_size,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        seed=seed,
    )

    resume_state: dict[str, Any] | None = None
    resume_path: Path | None = None
    if resume:
        candidates = sorted(
            output_directory.glob(f"checkpoint_epoch*_seed{seed}.pt"),
            key=lambda path: int(path.name.split("_seed")[0].split("epoch")[1]),
            reverse=True,
        )
        final_path = output_directory / f"checkpoint_final_seed{seed}.pt"
        if final_path.is_file():
            candidates.insert(0, final_path)
        if candidates:
            resume_path = candidates[0]
            loaded = load_model_checkpoint(resume_path, map_location="cpu")
            if loaded.model_spec != config.model:
                raise ValueError("The saved model specification differs from the requested run.")
            resume_state = loaded.training_state
            if resume_state is None:
                raise ValueError(f"Checkpoint has no resumable training state: {resume_path}")
            if not _resume_configs_match(
                resume_state.get("resolved_config"), _resolved_seed_config(config, seed=seed),
            ):
                raise ValueError("The saved training configuration differs from the requested run.")
            if resume_state.get("implementation_sha256") != implementation_sha256:
                raise ValueError("Checkpoint implementation hash differs from the current implementation record.")
            if resume_state.get("data_fingerprints") != data_fingerprints:
                raise ValueError("Prepared MNIST data differ from the saved checkpoint data.")
            model = loaded.model
    if resume_state is None:
        model = instantiate_model(config.model)
        fit_preprocessing = getattr(model, "fit_preprocessing", None)
        if callable(fit_preprocessing):
            fit_preprocessing(splits.train.images, seed=seed)
        # A warm start loads the weights of an earlier run after any preprocessing fit, so they win.
        if config.training.init_checkpoint is not None:
            init_path = Path(config.training.init_checkpoint.format(seed=seed))
            initial = load_model_checkpoint(init_path, map_location="cpu")
            if initial.model_spec != config.model:
                raise ValueError(f"The model specification of {init_path} differs from the requested run.")
            model.load_state_dict(initial.model.state_dict())
    parameter_stats_line = format_trainable_parameter_stats_line(model)
    optimizer = _build_optimizer(config.optimizer, model.parameters())
    runner = ImageClassifierRunner(
        model=model,
        optimizer=optimizer,
        device=device,
        metric_collectors=collectors,
    )

    initial_history: TrainingHistory | None = None
    start_epoch = 1
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        initial_history = TrainingHistory(**resume_state["training_history"])
        start_epoch = int(resume_state["epoch"]) + 1
        _restore_training_rng(resume_state["rng_state"], train_loader=train_loader)

    snapshot_paths: dict[int, Path] = (
        {
            int(path.name.split("_seed")[0].split("epoch")[1]): path
            for path in output_directory.glob(f"checkpoint_epoch*_seed{seed}.pt")
            if resume_path is not None
            and int(path.name.split("_seed")[0].split("epoch")[1]) < start_epoch
        }
        if resume_state is not None
        else {}
    )
    last_train_summary: EvaluationSummary | None = (
        _summary_from_plain(resume_state.get("last_train_summary"))
        if resume_state is not None
        else None
    )
    last_test_summary: EvaluationSummary | None = (
        _summary_from_plain(resume_state.get("last_test_summary"))
        if resume_state is not None
        else None
    )

    def save_epoch_outputs(
        epoch: int,
        train_summary: EvaluationSummary,
        test_summary: EvaluationSummary | None,
        history: TrainingHistory,
    ) -> None:
        nonlocal last_train_summary, last_test_summary
        last_train_summary = train_summary
        if test_summary is not None:
            last_test_summary = test_summary
        if epoch in snapshot_epochs:
            checkpoint_path = output_directory / f"checkpoint_epoch{epoch}_seed{seed}.pt"
            save_model_checkpoint(
                runner.model,
                config.model,
                checkpoint_path,
                training_state=(
                    _capture_training_state(
                        epoch=epoch,
                        config=config,
                        seed=seed,
                        data_fingerprints=data_fingerprints,
                        implementation_sha256=implementation_sha256,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        history=history,
                        last_train_summary=last_train_summary,
                        last_test_summary=last_test_summary,
                    )
                    if save_training_state
                    else None
                ),
            )
            snapshot_paths[epoch] = checkpoint_path
        if seed_epoch_end_callback is not None:
            seed_epoch_end_callback(seed, epoch, train_summary, test_summary, history)

    history = runner.fit(
        train_loader,
        test_loader,
        num_epochs=config.training.num_epochs,
        set_to_none=config.training.set_to_none,
        test_requires_grad=config.training.test_requires_grad,
        test_evaluation_interval_epochs=config.training.test_evaluation_interval_epochs,
        progress_factory=progress_factory,
        epoch_end_callback=save_epoch_outputs,
        start_epoch=start_epoch,
        initial_history=initial_history,
    )
    if last_train_summary is None or last_test_summary is None:
        raise RuntimeError("Automated training finished without recording final epoch summaries.")

    final_checkpoint_path = output_directory / f"checkpoint_final_seed{seed}.pt"
    result_path = output_directory / f"result_seed{seed}.pt"
    completed_artifacts = (
        resume_path == final_checkpoint_path
        and start_epoch == config.training.num_epochs + 1
        and result_path.is_file()
    )
    if not completed_artifacts:
        save_model_checkpoint(
            runner.model,
            config.model,
            final_checkpoint_path,
            training_state=(
                _capture_training_state(
                    epoch=config.training.num_epochs,
                    config=config,
                    seed=seed,
                    data_fingerprints=data_fingerprints,
                    implementation_sha256=implementation_sha256,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    history=history,
                    last_train_summary=last_train_summary,
                    last_test_summary=last_test_summary,
                )
                if save_training_state
                else None
            ),
        )

    collector_states = runner.export_collector_states()
    for collector_name, collector_state in collector_states.items():
        _ensure_torch_serializable(collector_state, label=f"collector {collector_name!r}")

    resolved_config = _resolved_seed_config(config, seed=seed)
    result_payload = {
        "resolved_config": resolved_config,
        "seed": seed,
        "device": str(device),
        "loss_name": _callable_name(runner.loss_collector.loss_fn),
        "parameter_stats_line": parameter_stats_line,
        "final_summaries": {
            "train": _summary_to_plain(last_train_summary),
            "test": _summary_to_plain(last_test_summary),
        },
        "training_history": history,
        "collector_states": collector_states,
        "files": {
            "result": f"result_seed{seed}.pt",
            "checkpoint_final": final_checkpoint_path.name,
            "snapshots": {
                str(epoch): path.name
                for epoch, path in sorted(snapshot_paths.items())
            },
        },
    }
    if save_training_state:
        result_payload["data_fingerprints"] = data_fingerprints
    if implementation_sha256 is not None:
        result_payload["implementation_sha256"] = implementation_sha256
    if not completed_artifacts:
        torch.save(result_payload, result_path)

    artifacts = SeedTrainingArtifacts(
        result_path=result_path,
        final_checkpoint_path=final_checkpoint_path,
        snapshot_paths=dict(snapshot_paths),
    )
    return SeedTrainingRun(
        seed=seed,
        runner=runner,
        train_loader=train_loader,
        test_loader=test_loader,
        history=history,
        final_train_summary=last_train_summary,
        final_test_summary=last_test_summary,
        collector_states=collector_states,
        artifacts=artifacts,
        resolved_config=resolved_config,
    )


def _prepare_implementation_record(
    output_directory: Path, *, implementation: Mapping[str, Any] | None, resume: bool,
) -> str | None:
    """Persist provenance before any training, without replacing its origin."""
    path = output_directory / "implementation.json"
    if implementation is None:
        if path.exists():
            raise ValueError("Resuming this run requires the current implementation record.")
        return None
    serialized = json.dumps(dict(implementation), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if resume:
        if not path.is_file():
            raise ValueError(f"Missing implementation record for resume: {path}")
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != json.loads(serialized):
            raise ValueError("The saved implementation differs from the current source code or software versions.")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resume_configs_match(saved: Any, requested: Mapping[str, Any]) -> bool:
    """Output naming does not change the experiment, including auto names."""
    if not isinstance(saved, Mapping):
        return False
    return {key: value for key, value in saved.items() if key != "output"} == {
        key: value for key, value in requested.items() if key != "output"
    }


def _fingerprint_image_dataset(dataset) -> str:
    """Hash prepared images and labels so resumed runs use identical examples."""

    digest = hashlib.sha256()
    for name, values in (("images", dataset.images), ("labels", dataset.labels)):
        host_values = values.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(host_values.shape).encode("ascii"))
        digest.update(str(host_values.dtype).encode("ascii"))
        digest.update(host_values.tobytes())
    return digest.hexdigest()


def _summary_from_plain(raw: Any) -> EvaluationSummary | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("Saved epoch summary must be a mapping.")
    return EvaluationSummary(loss=float(raw["loss"]), metrics=dict(raw["metrics"]))


def _capture_training_state(
    *,
    epoch: int,
    config: AutoTrainingConfig,
    seed: int,
    data_fingerprints: dict[str, str],
    implementation_sha256: str | None,
    optimizer: Optimizer,
    train_loader: DataLoader,
    history: TrainingHistory,
    last_train_summary: EvaluationSummary,
    last_test_summary: EvaluationSummary | None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    loader_generator = train_loader.generator
    return {
        "epoch": epoch,
        "resolved_config": _resolved_seed_config(config, seed=seed),
        "data_fingerprints": dict(data_fingerprints),
        "implementation_sha256": implementation_sha256,
        "optimizer_state": optimizer.state_dict(),
        "training_history": asdict(history),
        "last_train_summary": _summary_to_plain(last_train_summary),
        "last_test_summary": (
            None if last_test_summary is None else _summary_to_plain(last_test_summary)
        ),
        "rng_state": {
            "python": random.getstate(),
            "numpy": (
                numpy_state[0],
                torch.from_numpy(numpy_state[1].copy()),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            ),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_loader": None if loader_generator is None else loader_generator.get_state(),
        },
    }


def _restore_training_rng(state: Mapping[str, Any], *, train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            numpy_state[1].detach().cpu().numpy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    if train_loader.generator is not None and state["train_loader"] is not None:
        train_loader.generator.set_state(state["train_loader"].cpu())


def _validate_config(config: AutoTrainingConfig) -> tuple[int, ...]:
    if config.seeds.seed_count < 1:
        raise ValueError(f"seed_count must be positive, got {config.seeds.seed_count}.")
    if config.training.num_epochs < 1:
        raise ValueError(f"num_epochs must be positive, got {config.training.num_epochs}.")
    if config.dataset.train_batch_size < 1:
        raise ValueError(f"train_batch_size must be positive, got {config.dataset.train_batch_size}.")
    if config.dataset.test_batch_size < 1:
        raise ValueError(f"test_batch_size must be positive, got {config.dataset.test_batch_size}.")
    if config.training.test_evaluation_interval_epochs < 1:
        raise ValueError(
            "test_evaluation_interval_epochs must be positive, "
            f"got {config.training.test_evaluation_interval_epochs}."
        )
    _ensure_torch_serializable(asdict(config.model), label="model spec")

    snapshot_epochs = tuple(config.training.snapshot_epochs)
    if len(set(snapshot_epochs)) != len(snapshot_epochs):
        raise ValueError("snapshot_epochs must not contain duplicates.")
    out_of_range_epochs = [
        epoch
        for epoch in snapshot_epochs
        if epoch < 1 or epoch > config.training.num_epochs
    ]
    if out_of_range_epochs:
        raise ValueError(
            "snapshot_epochs must fall inside the training range 1..num_epochs, "
            f"got {out_of_range_epochs}."
        )

    if config.optimizer.kind not in {"adam", "sgd", "mfa_adam"}:
        raise ValueError(
            "optimizer.kind must be one of 'adam', 'sgd', or 'mfa_adam', "
            f"got {config.optimizer.kind!r}."
        )

    return tuple(sorted(snapshot_epochs))


def _resolve_execution_device(requested_device: str | torch.device | None) -> torch.device:
    if requested_device is not None:
        resolved_device = torch.device(requested_device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for automated training but is unavailable.")
        return resolved_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_output_directory(config: OutputConfig, *, resume: bool = False) -> Path:
    root = Path(config.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if config.directory_name is not None:
        if not config.directory_name:
            raise ValueError("output.directory_name must be a non-empty string when provided.")
        output_directory = root / config.directory_name
        if output_directory.exists():
            if resume and output_directory.is_dir():
                return output_directory
            raise FileExistsError(f"Output directory already exists: {output_directory}")
        if resume:
            raise FileNotFoundError(f"No output directory to resume: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        output_directory.mkdir(parents=False, exist_ok=False)
        return output_directory

    if resume:
        raise ValueError("Resuming requires an explicit output directory name.")

    if config.use_timestamp_dir:
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        base_name = "run"

    suffix = 0
    while True:
        candidate_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        output_directory = root / candidate_name
        if not output_directory.exists():
            output_directory.mkdir(parents=False, exist_ok=False)
            return output_directory
        suffix += 1


def _build_mnist_loaders(
    splits: PreparedMnistSplits,
    *,
    train_batch_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        splits.train,
        batch_size=train_batch_size,
        shuffle=True,
        generator=train_loader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def _loader_settings_for_device(
    *,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
) -> tuple[int, bool]:
    if device.type == "cuda":
        return 0, False
    return num_workers, pin_memory


def _build_optimizer(config: OptimizerConfig, parameters) -> Optimizer:
    if config.kind == "mfa_adam":
        from qcnn.baselines.mfa_optimizer import MFAAdam

        if config.weight_decay != 0:
            raise ValueError("The published MFA Adam optimizer has no weight decay.")
        return MFAAdam(parameters, lr=config.learning_rate)
    if config.kind == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.kind == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise ValueError(f"Unsupported optimizer kind {config.kind!r}.")


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summary_to_plain(summary: EvaluationSummary) -> dict[str, Any]:
    return {
        "loss": summary.loss,
        "metrics": dict(summary.metrics),
    }


def _callable_name(obj: Any) -> str:
    if hasattr(obj, "__name__") and not isinstance(obj, torch.nn.Module):
        return str(obj.__name__)
    return obj.__class__.__name__


def _seed_manifest_entry(run: SeedTrainingRun, *, output_directory: Path) -> dict[str, Any]:
    return {
        "seed": run.seed,
        "result": run.artifacts.result_path.relative_to(output_directory).as_posix(),
        "checkpoint_final": run.artifacts.final_checkpoint_path.relative_to(output_directory).as_posix(),
        "snapshots": {
            str(epoch): path.relative_to(output_directory).as_posix()
            for epoch, path in sorted(run.artifacts.snapshot_paths.items())
        },
        "final_train_summary": _summary_to_plain(run.final_train_summary),
        "final_test_summary": _summary_to_plain(run.final_test_summary),
    }


def _resolved_global_config(config: AutoTrainingConfig) -> dict[str, Any]:
    plain_config = _to_plain_data(asdict(config))
    plain_config["dataset"] = _resolved_dataset_config(config.dataset)
    # Only warm-started runs record the initial checkpoint, so older manifests keep their form.
    if config.training.init_checkpoint is None:
        del plain_config["training"]["init_checkpoint"]
    plain_config["output"]["directory_name"] = (
        config.output.directory_name
        if config.output.directory_name is not None
        else None
    )
    return plain_config


def _resolved_seed_config(config: AutoTrainingConfig, *, seed: int) -> dict[str, Any]:
    plain_config = _resolved_global_config(config)
    plain_config["seed"] = seed
    return plain_config


def _resolved_dataset_config(config: MnistDatasetConfig) -> dict[str, Any]:
    resolved_scaled_image_size = (
        config.image_size if config.scaled_image_size is None else int(config.scaled_image_size)
    )
    resolved = {
        "root": str(config.root),
        "samples_per_class": config.samples_per_class,
        "image_size": config.image_size,
        "scaled_image_size": resolved_scaled_image_size,
        "max_offset": config.max_offset,
        "train_batch_size": config.train_batch_size,
        "test_batch_size": config.test_batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "download": config.download,
    }
    # Only runs with a separate test translation record it, so older manifests keep their form.
    if config.test_max_offset is not None:
        resolved["test_max_offset"] = config.test_max_offset
    return resolved


def _format_epoch_metric(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{value:.1%}"
    return f"{value:.4f}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    clamped_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(clamped_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_primary_progress_description(remaining_seconds: float | None = None) -> str:
    return f"Seeds ETA {_format_eta(remaining_seconds)}"


def _format_secondary_progress_description(remaining_seconds: float | None = None) -> str:
    return f"Epochs ETA {_format_eta(remaining_seconds)}"


def _format_epoch_status_line(
    *,
    train_summary: EvaluationSummary | None = None,
    test_summary: EvaluationSummary | None = None,
) -> str:
    train_accuracy = None if train_summary is None else train_summary.metrics.get("accuracy")
    test_accuracy = None if test_summary is None else test_summary.metrics.get("accuracy")
    return (
        f"Loss {_format_epoch_metric(None if train_summary is None else train_summary.loss)} "
        f"{_format_epoch_metric(None if test_summary is None else test_summary.loss)} "
        f"Accuracy {_format_epoch_metric(train_accuracy, percent=True)} "
        f"{_format_epoch_metric(test_accuracy, percent=True)}"
    )


def _ensure_torch_serializable(value: Any, *, label: str) -> None:
    try:
        buffer = io.BytesIO()
        torch.save(value, buffer)
    except Exception as exc:  # pragma: no cover - exact exception type is implementation-dependent
        raise ValueError(f"{label} is not torch.save-serializable.") from exc


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_data(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return repr(value)
