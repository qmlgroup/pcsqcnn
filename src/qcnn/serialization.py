"""Serialization helpers for auto-training checkpoints and run results.

The module exposes the checkpoint/result loaders used by automated training,
along with reevaluation helpers built on top of the saved MNIST run metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qcnn.data import PreparedMnistSplits, prepare_mnist_splits
from qcnn.model import EvaluationSummary, ImageClassifierRunner, TrainingHistory
from qcnn.model_spec import ModelSpec, instantiate_model, model_spec_from_mapping, model_spec_to_dict

_CHECKPOINT_TYPE = "qcnn_model_checkpoint"
_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedModelCheckpoint:
    """Reconstructed model and metadata loaded from one checkpoint bundle.

    Attributes:
        model: Reconstructed ``nn.Module`` with checkpoint weights loaded.
        model_spec: Saved model-construction specification from the checkpoint.
        checkpoint_path: Filesystem path of the loaded checkpoint bundle.
    """

    model: nn.Module
    model_spec: ModelSpec
    checkpoint_path: Path


@dataclass(frozen=True)
class LoadedAutoTrainingRun:
    """Reconstructed model plus saved result payload for one auto-training seed.

    Attributes:
        model: Reconstructed ``nn.Module`` with the saved final weights loaded.
        model_spec: Saved model specification from the checkpoint bundle.
        checkpoint_path: Path to the loaded final checkpoint bundle.
        result_path: Path to the loaded ``result_seed*.pt`` payload.
        seed: Seed associated with the loaded run.
        device: Saved device string from the run result payload.
        loss_name: Saved loss-function name from the run result payload.
        resolved_config: Saved plain resolved training config for the run.
        final_summaries: Saved final train/test summaries from the result.
        training_history: Saved epoch-level history from the result payload.
        collector_states: Saved exported collector states from the result.
    """

    model: nn.Module
    model_spec: ModelSpec
    checkpoint_path: Path
    result_path: Path
    seed: int
    device: str
    loss_name: str
    resolved_config: dict[str, Any]
    final_summaries: dict[str, dict[str, Any]]
    training_history: TrainingHistory
    collector_states: dict[str, Any]

    def to_summary_dict(self) -> dict[str, Any]:
        """Build a compact nested summary for notebook-side inspection."""

        summary: dict[str, Any] = {
            "run": {
                "seed": self.seed,
                "checkpoint_path": str(self.checkpoint_path),
                "result_path": str(self.result_path),
                "model_class": self.model.__class__.__name__,
                "device": self.device,
                "loss_name": self.loss_name,
                "training_history_available": self.training_history is not None,
            },
            "model_spec": model_spec_to_dict(self.model_spec),
        }

        dataset_config = self.resolved_config.get("dataset")
        if isinstance(dataset_config, Mapping):
            summary["dataset"] = dict(dataset_config)

        optimizer_config = self.resolved_config.get("optimizer")
        if isinstance(optimizer_config, Mapping):
            summary["optimizer"] = dict(optimizer_config)

        training_config = self.resolved_config.get("training")
        if isinstance(training_config, Mapping):
            summary["training"] = dict(training_config)

        summary["final_summaries"] = dict(self.final_summaries)
        return summary

    def to_markdown(self, *, title: str = "Saved Run Metadata") -> str:
        """Render the run summary as Markdown."""

        return format_nested_mapping_markdown(self.to_summary_dict(), title=title)

    def build_evaluation_notes(self) -> list[str]:
        """Describe evaluation caveats implied by the saved run metadata."""

        notes: list[str] = []
        metric_names = list(self.training_history.test_metrics)
        custom_metrics = [metric_name for metric_name in metric_names if metric_name != "accuracy"]

        if self.loss_name != "CrossEntropyLoss":
            notes.append(
                f"Saved loss is {self.loss_name!r}; reconstructed evaluation uses CrossEntropyLoss."
            )
        if custom_metrics:
            notes.append(
                "Custom saved metrics are not reconstructed; evaluation reports loss and accuracy only."
            )
        return notes

    def saved_mnist_test_config(self) -> dict[str, Any]:
        """Extract the saved MNIST reconstruction config from the run result."""

        dataset_config = self.resolved_config.get("dataset")
        if not isinstance(dataset_config, Mapping):
            raise ValueError("Run result does not contain a valid dataset configuration.")

        missing = object()
        image_size = dataset_config.get("image_size", missing)
        scaled_image_size = dataset_config.get("scaled_image_size", missing)
        max_offset = dataset_config.get("max_offset", missing)
        samples_per_class = dataset_config.get("samples_per_class", missing)
        test_batch_size = dataset_config.get("test_batch_size", missing)
        num_workers = dataset_config.get("num_workers", missing)
        pin_memory = dataset_config.get("pin_memory", missing)
        seed = self.resolved_config.get("seed", missing)
        missing_fields = [
            field_name
            for field_name, field_value in (
                ("image_size", image_size),
                ("scaled_image_size", scaled_image_size),
                ("max_offset", max_offset),
                ("test_batch_size", test_batch_size),
                ("num_workers", num_workers),
                ("pin_memory", pin_memory),
                ("seed", seed),
            )
            if field_value is missing or field_value is None
        ]
        if samples_per_class is missing:
            missing_fields.append("samples_per_class")
        if missing_fields:
            raise ValueError(
                "Run MNIST reconstruction metadata is incomplete. Missing: "
                + ", ".join(missing_fields)
                + "."
            )

        return {
            "dataset_name": "MNIST",
            "image_size": int(image_size),
            "scaled_image_size": int(scaled_image_size),
            "max_offset": int(max_offset),
            "samples_per_class": (
                None if samples_per_class is None else int(samples_per_class)
            ),
            "seed": int(seed),
            "test_loader": {
                "batch_size": int(test_batch_size),
                "num_workers": int(num_workers),
                "pin_memory": bool(pin_memory),
                "drop_last": False,
            },
        }

    def __str__(self) -> str:
        """Render the run as the default Markdown summary string."""

        return self.to_markdown()


@dataclass(frozen=True)
class LoadedAutoTrainingRunEvaluationContext:
    """Runner and loader context rebuilt from a loaded auto-training run.

    Attributes:
        runner: Fresh runner wrapping the loaded model.
        test_loader: Reconstructed MNIST test loader for the saved run.
        device: Device used by the reconstructed runner.
        notes: Human-readable caveats about the reconstructed evaluation flow.
    """

    runner: ImageClassifierRunner
    test_loader: DataLoader
    device: torch.device
    notes: list[str]


@dataclass(frozen=True)
class LoadedAutoTrainingRunTestEvaluation:
    """Result of re-evaluating a loaded auto-training run on its saved MNIST test setup.

    Attributes:
        context: Reconstructed evaluation context used for the test pass.
        summary: Loader-level loss and metrics computed during evaluation.
        report_dict: Nested mapping with recomputed metrics, saved metrics, and
            deltas suitable for programmatic inspection.
        report_markdown: Human-readable Markdown rendering of ``report_dict``.
    """

    context: LoadedAutoTrainingRunEvaluationContext
    summary: EvaluationSummary
    report_dict: dict[str, Any]
    report_markdown: str


@dataclass(frozen=True)
class AutoTrainingSnapshotTestEvaluation:
    """Result of re-evaluating one saved snapshot checkpoint on MNIST test data.

    Attributes:
        checkpoint_path: Snapshot checkpoint used for the reevaluation.
        model_spec: Reconstruction spec with the effective ``shot_budget``.
        epoch: Snapshot epoch that was evaluated.
        shot_budget: Effective finite-shot budget. ``None`` means exact `Inf`.
        summary: Aggregate test loss and accuracy for the full test pass.
        predictions: Integer class predictions for the whole test split.
        targets: Integer ground-truth labels for the whole test split.
    """

    checkpoint_path: Path
    model_spec: ModelSpec
    epoch: int
    shot_budget: int | None
    summary: EvaluationSummary
    predictions: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True)
class AutoTrainingSnapshotEvaluationContext:
    """Reconstructed runner and test loader for one snapshot reevaluation.

    Attributes:
        checkpoint_path: Snapshot checkpoint used for reevaluation.
        model_spec: Effective reconstruction spec after applying any runtime
            overrides such as ``shot_budget``.
        epoch: Snapshot epoch associated with ``checkpoint_path``.
        shot_budget: Effective finite-shot budget. ``None`` means exact `Inf`.
        runner: Fresh runner wrapping the reconstructed snapshot model.
        test_loader: Reconstructed MNIST test loader for the saved run.
        device: Device used by the reconstructed runner.
    """

    checkpoint_path: Path
    model_spec: ModelSpec
    epoch: int
    shot_budget: int | None
    runner: ImageClassifierRunner
    test_loader: DataLoader
    device: torch.device


@dataclass(frozen=True)
class AutoTrainingSnapshotRepeatedEvaluation:
    """Repeated stochastic reevaluation summary for one snapshot checkpoint.

    Attributes:
        context: Snapshot reevaluation context reused across repeated passes.
        mean_loss: One scalar mean test loss per stochastic full-test pass.
        mean_accuracy: One scalar mean test accuracy per stochastic full-test
            pass.
    """

    context: AutoTrainingSnapshotEvaluationContext
    mean_loss: torch.Tensor
    mean_accuracy: torch.Tensor


@dataclass(frozen=True)
class AutoTrainingSnapshotBatchwiseShotBudgetEvaluation:
    """Batchwise finite-shot or exact aggregates for one shot setting."""

    shot_budget: int | None
    num_draws: int
    batch_sizes: torch.Tensor
    batch_loss_sum: torch.Tensor
    batch_correct_count: torch.Tensor


@dataclass(frozen=True)
class AutoTrainingSnapshotBatchwiseLossSamplingEvaluation:
    """Batchwise finite-shot or exact aggregates for one snapshot checkpoint."""

    context: AutoTrainingSnapshotEvaluationContext
    repetitions: int
    batch_size: int
    repetition_block_size: int
    evaluations: list[AutoTrainingSnapshotBatchwiseShotBudgetEvaluation]


@dataclass(frozen=True)
class AutoTrainingSnapshotReadoutLandscapeEvaluation:
    """Loss landscape over perturbed exact readout distributions for one checkpoint."""

    context: AutoTrainingSnapshotEvaluationContext
    alpha: torch.Tensor
    beta: torch.Tensor
    mean_loss: torch.Tensor
    valid_count: torch.Tensor
    valid_fraction: torch.Tensor
    total_samples: int
    eligible_sample_count: int


@dataclass(frozen=True)
class AutoTrainingSnapshotLayerGradientNormsEvaluation:
    """Mean-gradient diagnostics for one saved snapshot checkpoint."""

    context: AutoTrainingSnapshotEvaluationContext
    layer_keys: tuple[str, ...]
    layer_labels: tuple[str, ...]
    gradient_norms: torch.Tensor
    mean_gradients: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AutoTrainingSnapshotReadoutEntropyEvaluation:
    """Per-sample Shannon entropy of sampled readout histograms for one checkpoint."""

    context: AutoTrainingSnapshotEvaluationContext
    shot_budget: int
    entropy: torch.Tensor


def resolve_snapshot_trainable_layer_blocks(
    model: nn.Module,
) -> list[tuple[str, str, tuple[tuple[str, nn.Parameter], ...]]]:
    """Resolve the trainable layer blocks used by Figure S2a.

    The current PCS-QCNN contract groups trainable parameters into quantum
    multiplexer layers followed by one classical classifier block.
    """

    named_parameters = tuple(model.named_parameters())
    groups: list[tuple[str, str, tuple[tuple[str, nn.Parameter], ...]]] = []
    assigned_names: set[str] = set()

    multiplexers = getattr(model, "multiplexers", ())
    for layer_index in range(len(multiplexers)):
        prefix = f"multiplexers.{layer_index}."
        group_parameters = tuple(
            (name, parameter)
            for name, parameter in named_parameters
            if name.startswith(prefix)
        )
        if not group_parameters:
            raise ValueError(
                f"Could not resolve any trainable parameters for Figure S2a block {prefix!r}."
            )
        groups.append(
            (
                f"multiplexers.{layer_index}",
                f"Quantum {layer_index + 1}",
                group_parameters,
            )
        )
        assigned_names.update(name for name, _ in group_parameters)

    classifier_parameters = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("classifier.")
    )
    if classifier_parameters:
        groups.append(("classifier", "Classifier", classifier_parameters))
        assigned_names.update(name for name, _ in classifier_parameters)

    if not groups:
        raise ValueError("Figure S2a requires at least one trainable layer block.")

    unassigned_parameter_names = [
        name
        for name, parameter in named_parameters
        if parameter.requires_grad and name not in assigned_names
    ]
    if unassigned_parameter_names:
        raise ValueError(
            "Figure S2a does not know how to group all trainable parameters. "
            f"Unassigned parameters: {unassigned_parameter_names}."
        )

    return groups


def _flatten_parameter_like_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Expected at least one tensor to flatten.")
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def compute_histogram_shannon_entropy(histograms: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy in nats from flattened or spatial histograms."""

    if histograms.ndim < 2:
        raise ValueError(
            "Histogram entropy expects at least one leading sample axis and one support axis, "
            f"got shape {tuple(histograms.shape)}."
        )
    if not torch.is_floating_point(histograms):
        raise ValueError(
            "Histogram entropy expects a real floating tensor, "
            f"got dtype {histograms.dtype}."
        )
    if not torch.isfinite(histograms).all():
        raise ValueError("Histogram entropy expects finite histogram values.")
    if torch.any(histograms < 0):
        raise ValueError("Histogram entropy expects non-negative histogram values.")

    if histograms.ndim >= 4:
        flattened = histograms.reshape(*histograms.shape[:-3], -1)
    else:
        flattened = histograms.reshape(*histograms.shape[:-1], -1)

    positive = flattened > 0
    safe_probabilities = torch.where(positive, flattened, torch.ones_like(flattened))
    entropy_terms = torch.where(
        positive,
        flattened * safe_probabilities.log(),
        torch.zeros_like(flattened),
    )
    return -entropy_terms.sum(dim=-1)


def format_nested_mapping_markdown(
    data: Mapping[str, Any],
    *,
    title: str | None = None,
    heading_level: int = 2,
    float_precision: int = 6,
    sort_keys: bool = False,
) -> str:
    """Render a nested mapping as readable Markdown.

    Args:
        data: Nested mapping to render.
        title: Optional top-level heading placed above the rendered sections.
        heading_level: Markdown heading level used for ``title``. Top-level
            data sections render one level deeper when ``title`` is present.
        float_precision: Number of digits after the decimal point for float
            leaves.
        sort_keys: Whether to sort mapping keys lexicographically instead of
            preserving insertion order.

    Returns:
        A Markdown string with nested headings for mapping sections, flat bullet
        rows for scalar leaves, inline scalar sequences, and numbered-style
        subheadings for sequences of mappings.
    """

    if heading_level < 1:
        raise ValueError(f"heading_level must be positive, got {heading_level}.")

    lines: list[str] = []
    if title is not None:
        lines.append(f"{'#' * min(heading_level, 6)} {title}")
        lines.append("")

    section_level = heading_level + 1 if title is not None else heading_level
    lines.extend(
        _render_mapping_sections(
            data,
            level=section_level,
            float_precision=float_precision,
            sort_keys=sort_keys,
        )
    )
    return "\n".join(lines).strip()


def save_model_checkpoint(
    model: nn.Module,
    model_spec: ModelSpec,
    path: str | Path,
) -> None:
    """Save a reconstruction-ready model checkpoint bundle.

    Args:
        model: Model instance whose weights should be saved.
        model_spec: Constructor specification required to rebuild ``model``.
        path: Target checkpoint path. Parent directories are created when
            needed.

    Returns:
        ``None``. The function writes one checkpoint bundle to ``path``.
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": _CHECKPOINT_TYPE,
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "model_spec": model_spec_to_dict(model_spec),
        "model_state": _cpu_state_dict(model),
    }
    torch.save(payload, checkpoint_path)


def load_model_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = None,
) -> LoadedModelCheckpoint:
    """Load a reconstruction-ready checkpoint bundle and rebuild the model.

    Args:
        path: Path to the saved checkpoint bundle.
        map_location: Optional device mapping forwarded to ``torch.load`` and
            then used to move the reconstructed model before loading weights.

    Returns:
        ``LoadedModelCheckpoint`` containing the reconstructed model and saved
        model specification.
    """

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must deserialize to a mapping.")

    checkpoint_type = payload.get("checkpoint_type")
    if checkpoint_type != _CHECKPOINT_TYPE:
        raise ValueError(
            f"Unsupported checkpoint_type {checkpoint_type!r}; expected {_CHECKPOINT_TYPE!r}."
        )

    format_version = payload.get("format_version")
    if format_version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version {format_version!r}; "
            f"expected {_CHECKPOINT_FORMAT_VERSION}."
        )

    raw_model_spec = payload.get("model_spec")
    if not isinstance(raw_model_spec, Mapping):
        raise ValueError("Checkpoint is missing a valid model_spec mapping.")
    model_spec = model_spec_from_mapping(raw_model_spec)
    model = instantiate_model(model_spec)
    if map_location is not None:
        model = model.to(torch.device(map_location))

    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint is missing a valid model_state mapping.")
    model.load_state_dict(model_state)

    return LoadedModelCheckpoint(
        model=model,
        model_spec=model_spec,
        checkpoint_path=checkpoint_path,
    )


def load_auto_training_run(
    run_directory: str | Path,
    seed: int,
    *,
    map_location: torch.device | str | None = None,
) -> LoadedAutoTrainingRun:
    """Load one automated run from ``checkpoint_final`` and ``result`` files.

    Args:
        run_directory: Directory containing the saved run files.
        seed: Seed selecting the `checkpoint_final_seed*.pt` and
            `result_seed*.pt` pair.
        map_location: Optional device mapping forwarded to checkpoint loading.

    Returns:
        ``LoadedAutoTrainingRun`` containing the reconstructed model, saved
        history, and saved experiment metadata.
    """

    resolved_directory = Path(run_directory)
    checkpoint_path = resolved_directory / f"checkpoint_final_seed{seed}.pt"
    result_path = resolved_directory / f"result_seed{seed}.pt"
    loaded_checkpoint = load_model_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )

    payload = torch.load(result_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Run result must deserialize to a mapping.")

    training_history = payload.get("training_history")
    if not isinstance(training_history, TrainingHistory):
        raise ValueError("Run result is missing a valid training_history object.")

    resolved_config = payload.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("Run result is missing a valid resolved_config mapping.")

    final_summaries = payload.get("final_summaries")
    if not isinstance(final_summaries, Mapping):
        raise ValueError("Run result is missing a valid final_summaries mapping.")

    collector_states = payload.get("collector_states")
    if not isinstance(collector_states, Mapping):
        raise ValueError("Run result is missing a valid collector_states mapping.")

    saved_seed = payload.get("seed")
    if saved_seed != seed:
        raise ValueError(f"Run result seed mismatch: expected {seed}, got {saved_seed!r}.")

    loss_name = payload.get("loss_name")
    if not isinstance(loss_name, str) or not loss_name:
        raise ValueError("Run result is missing a valid loss_name.")

    return LoadedAutoTrainingRun(
        model=loaded_checkpoint.model,
        model_spec=loaded_checkpoint.model_spec,
        checkpoint_path=loaded_checkpoint.checkpoint_path,
        result_path=result_path,
        seed=seed,
        device=str(payload.get("device")),
        loss_name=loss_name,
        resolved_config=dict(resolved_config),
        final_summaries={
            phase: dict(summary)
            for phase, summary in final_summaries.items()
        },
        training_history=training_history,
        collector_states=dict(collector_states),
    )


def reconstruct_saved_mnist_splits_from_run(
    loaded_run: LoadedAutoTrainingRun,
    *,
    root: str | Path,
    download: bool = True,
) -> PreparedMnistSplits:
    """Rebuild the saved MNIST train/test splits described by a loaded run.

    Args:
        loaded_run: Loaded auto-training run with saved MNIST metadata.
        root: Dataset root forwarded to ``prepare_mnist_splits``.
        download: Forwarded to ``prepare_mnist_splits``.

    Returns:
        ``PreparedMnistSplits`` reconstructed from the saved run metadata.
    """

    mnist_config = loaded_run.saved_mnist_test_config()
    return prepare_mnist_splits(
        root=root,
        samples_per_class=mnist_config["samples_per_class"],
        image_size=mnist_config["image_size"],
        scaled_image_size=mnist_config["scaled_image_size"],
        max_offset=mnist_config["max_offset"],
        seed=mnist_config["seed"],
        download=download,
    )


def reconstruct_run_runner_and_test_loader(
    loaded_run: LoadedAutoTrainingRun,
    splits: PreparedMnistSplits,
    *,
    device: torch.device | str | None = None,
) -> LoadedAutoTrainingRunEvaluationContext:
    """Rebuild a runner and test loader from a loaded run and saved splits.

    Args:
        loaded_run: Loaded auto-training run.
        splits: Reconstructed MNIST splits matching the saved metadata.
        device: Optional execution device. When omitted, the current device of
            ``loaded_run.model`` is reused.

    Returns:
        ``LoadedAutoTrainingRunEvaluationContext`` containing a fresh runner,
        reconstructed test loader, resolved device, and evaluation notes.
    """

    mnist_config = loaded_run.saved_mnist_test_config()
    test_loader_settings = dict(mnist_config["test_loader"])
    resolved_device = (
        torch.device(device)
        if device is not None
        else _infer_module_device(loaded_run.model)
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=int(test_loader_settings.get("batch_size") or 64),
        shuffle=False,
        num_workers=int(test_loader_settings.get("num_workers") or 0),
        pin_memory=bool(test_loader_settings.get("pin_memory", False)),
        drop_last=bool(test_loader_settings.get("drop_last", False)),
    )
    runner = ImageClassifierRunner(model=loaded_run.model, device=resolved_device)
    return LoadedAutoTrainingRunEvaluationContext(
        runner=runner,
        test_loader=test_loader,
        device=resolved_device,
        notes=loaded_run.build_evaluation_notes(),
    )


def evaluate_loaded_auto_training_run_on_saved_mnist_test(
    loaded_run: LoadedAutoTrainingRun,
    *,
    root: str | Path,
    device: torch.device | str | None = None,
    download: bool = True,
) -> LoadedAutoTrainingRunTestEvaluation:
    """Re-evaluate a loaded run on the MNIST test setup saved in its result.

    Args:
        loaded_run: Loaded auto-training run.
        root: Dataset root used when reconstructing MNIST.
        device: Optional execution device for the reconstructed runner.
        download: Forwarded to MNIST reconstruction.

    Returns:
        ``LoadedAutoTrainingRunTestEvaluation`` with the reconstructed context,
        recomputed evaluation summary, and Markdown report.
    """

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
    evaluation_summary = evaluation_context.runner.evaluate_loader(evaluation_context.test_loader)
    evaluation_report = _build_loaded_run_test_report(loaded_run, evaluation_summary)
    return LoadedAutoTrainingRunTestEvaluation(
        context=evaluation_context,
        summary=evaluation_summary,
        report_dict=evaluation_report,
        report_markdown=format_nested_mapping_markdown(
            evaluation_report,
            title="Recomputed Test Report",
        ),
    )


def evaluate_auto_training_snapshot_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budget: int | None,
    root: str | Path,
    device: torch.device | str | None = None,
    download: bool = True,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> AutoTrainingSnapshotTestEvaluation:
    """Re-evaluate one saved snapshot checkpoint on its MNIST test split.

    Args:
        run_directory: Directory containing one completed automated-training run.
        seed: Seed selecting the saved result payload and snapshot checkpoint.
        epoch: Snapshot epoch to evaluate.
        shot_budget: Finite-shot budget used during evaluation. ``None`` means
            exact readout and is rendered as `Inf` in Figure 6.
        root: Dataset root used to reconstruct the saved MNIST split.
        device: Optional execution device. When omitted, CPU is used unless the
            reconstructed model already lives on a different device.
        download: Forwarded to the MNIST reconstruction helper.
        batch_progress_callback: Optional callback receiving
            ``(completed_batches, total_batches)`` once per processed test batch.

    Returns:
        ``AutoTrainingSnapshotTestEvaluation`` with aggregate metrics and the
        full per-sample class predictions for the test split.
    """

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=shot_budget,
        root=root,
        device=device,
        download=download,
    )
    summary, predictions, targets = _evaluate_runner_and_collect_predictions(
        context.runner,
        context.test_loader,
        batch_progress_callback=batch_progress_callback,
    )
    return AutoTrainingSnapshotTestEvaluation(
        checkpoint_path=context.checkpoint_path,
        model_spec=context.model_spec,
        epoch=context.epoch,
        shot_budget=context.shot_budget,
        summary=summary,
        predictions=predictions,
        targets=targets,
    )


def reconstruct_auto_training_snapshot_runner_and_test_loader(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budget: int | None,
    root: str | Path,
    device: torch.device | str | None = None,
    download: bool = True,
) -> AutoTrainingSnapshotEvaluationContext:
    """Rebuild a runner and test loader for one snapshot checkpoint."""

    resolved_directory = Path(run_directory)
    loaded_run = load_auto_training_run(resolved_directory, seed=seed, map_location="cpu")
    checkpoint_path = _resolve_snapshot_checkpoint_path(
        resolved_directory,
        seed=seed,
        epoch=epoch,
    )
    loaded_checkpoint = load_model_checkpoint(checkpoint_path, map_location="cpu")
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    model_spec = _override_model_spec_shot_budget(loaded_checkpoint.model_spec, shot_budget=shot_budget)
    model = instantiate_model(model_spec).to(resolved_device)
    model.load_state_dict(loaded_checkpoint.model.state_dict())

    restored_splits = reconstruct_saved_mnist_splits_from_run(
        loaded_run,
        root=root,
        download=download,
    )
    mnist_config = loaded_run.saved_mnist_test_config()
    test_loader_settings = dict(mnist_config["test_loader"])
    if resolved_device.type == "cuda":
        restored_splits = restored_splits.to(resolved_device)
        test_loader_settings["num_workers"] = 0
        test_loader_settings["pin_memory"] = False
    test_loader = DataLoader(
        restored_splits.test,
        batch_size=int(test_loader_settings.get("batch_size") or 64),
        shuffle=False,
        num_workers=int(test_loader_settings.get("num_workers") or 0),
        pin_memory=bool(test_loader_settings.get("pin_memory", False)),
        drop_last=bool(test_loader_settings.get("drop_last", False)),
    )
    runner = ImageClassifierRunner(model=model, device=resolved_device)
    return AutoTrainingSnapshotEvaluationContext(
        checkpoint_path=checkpoint_path,
        model_spec=model_spec,
        epoch=epoch,
        shot_budget=shot_budget,
        runner=runner,
        test_loader=test_loader,
        device=resolved_device,
    )


def _build_snapshot_test_loader_for_batch_size(
    context: AutoTrainingSnapshotEvaluationContext,
    *,
    batch_size: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    return DataLoader(
        context.test_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def evaluate_auto_training_snapshot_layer_gradient_norms_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    root: str | Path,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    download: bool = True,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> AutoTrainingSnapshotLayerGradientNormsEvaluation:
    """Evaluate Figure S2a layerwise mean-gradient norms on saved MNIST test data."""

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=None,
        root=root,
        device=device,
        download=download,
    )
    model = context.runner.model
    layer_blocks = resolve_snapshot_trainable_layer_blocks(model)
    test_loader = _build_snapshot_test_loader_for_batch_size(context, batch_size=batch_size)
    num_batches = len(test_loader)

    layer_gradient_sums: list[torch.Tensor] = [
        torch.zeros(
            sum(parameter.numel() for _, parameter in group_parameters),
            dtype=group_parameters[0][1].dtype,
            device=context.device,
        )
        for _, _, group_parameters in layer_blocks
    ]
    total_samples = 0

    model.eval()
    for batch_index, (images, labels) in enumerate(test_loader):
        model.zero_grad(set_to_none=True)
        device_images = images.to(context.device, non_blocking=True)
        device_labels = labels.to(device=context.device, dtype=torch.long, non_blocking=True)
        current_batch_size = int(device_labels.shape[0])

        logits = model(device_images)
        loss = context.runner.loss_collector.compute_loss(logits, device_labels)

        grouped_parameters = [group_parameters for _, _, group_parameters in layer_blocks]
        flattened_parameters = [
            parameter
            for group_parameters in grouped_parameters
            for _, parameter in group_parameters
        ]
        gradients = torch.autograd.grad(
            loss,
            flattened_parameters,
            allow_unused=True,
        )

        gradient_offset = 0
        for group_index, group_parameters in enumerate(grouped_parameters):
            current_group_gradients: list[torch.Tensor] = []
            for _, parameter in group_parameters:
                gradient = gradients[gradient_offset]
                gradient_offset += 1
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                current_group_gradients.append(gradient)
            flattened_gradient = _flatten_parameter_like_tensors(current_group_gradients)
            layer_gradient_sums[group_index] += flattened_gradient * current_batch_size

        total_samples += current_batch_size
        if batch_progress_callback is not None:
            batch_progress_callback(batch_index + 1, num_batches)

    if total_samples <= 0:
        raise ValueError("Figure S2a gradient evaluation requires a non-empty test split.")

    mean_gradients = tuple(
        (gradient_sum / float(total_samples)).detach().cpu().clone()
        for gradient_sum in layer_gradient_sums
    )
    gradient_norms = torch.stack(
        [mean_gradient.norm() for mean_gradient in mean_gradients]
    ).to(dtype=torch.float32)
    return AutoTrainingSnapshotLayerGradientNormsEvaluation(
        context=context,
        layer_keys=tuple(layer_key for layer_key, _, _ in layer_blocks),
        layer_labels=tuple(layer_label for _, layer_label, _ in layer_blocks),
        gradient_norms=gradient_norms,
        mean_gradients=mean_gradients,
    )


def evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budget: int,
    root: str | Path,
    device: torch.device | str | None = None,
    batch_size: int = 128,
    download: bool = True,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> AutoTrainingSnapshotReadoutEntropyEvaluation:
    """Evaluate Figure S2b readout entropy on saved MNIST test data."""

    if shot_budget <= 0:
        raise ValueError(f"shot_budget must be positive, got {shot_budget}.")

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=None,
        root=root,
        device=device,
        download=download,
    )
    model = context.runner.model
    exact_quantum_readout = getattr(model, "exact_quantum_readout_probabilities", None)
    if not callable(exact_quantum_readout):
        raise ValueError(
            "Figure S2b readout-entropy evaluation requires a model exposing "
            "exact_quantum_readout_probabilities(...)."
        )

    test_loader = _build_snapshot_test_loader_for_batch_size(context, batch_size=batch_size)
    num_batches = len(test_loader)
    histogram_layer = model.readout_histogram.__class__(shot_budget=shot_budget).eval()
    entropy_batches: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(test_loader):
            device_images = images.to(context.device, non_blocking=True)
            exact_readout = exact_quantum_readout(device_images)
            sampled_histogram = histogram_layer(exact_readout)
            entropy_batches.append(
                compute_histogram_shannon_entropy(sampled_histogram).detach().cpu().reshape(-1)
            )
            if batch_progress_callback is not None:
                batch_progress_callback(batch_index + 1, num_batches)

    if not entropy_batches:
        raise ValueError("Figure S2b readout-entropy evaluation requires a non-empty test split.")

    return AutoTrainingSnapshotReadoutEntropyEvaluation(
        context=context,
        shot_budget=shot_budget,
        entropy=torch.cat(entropy_batches, dim=0),
    )


def evaluate_auto_training_snapshot_repeated_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budget: int | None,
    repetitions: int,
    root: str | Path,
    device: torch.device | str | None = None,
    download: bool = True,
) -> AutoTrainingSnapshotRepeatedEvaluation:
    """Repeatedly reevaluate one snapshot checkpoint on its saved MNIST test split."""

    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}.")

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=shot_budget,
        root=root,
        device=device,
        download=download,
    )
    losses: list[float] = []
    accuracies: list[float] = []
    for _ in range(repetitions):
        summary = context.runner.evaluate_loader(context.test_loader)
        losses.append(summary.loss)
        accuracies.append(summary.metrics["accuracy"])

    return AutoTrainingSnapshotRepeatedEvaluation(
        context=context,
        mean_loss=torch.tensor(losses, dtype=torch.float32),
        mean_accuracy=torch.tensor(accuracies, dtype=torch.float32),
    )


def evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    shot_budgets: Sequence[int | None],
    repetitions: int,
    batch_size: int,
    repetition_block_size: int = 100,
    root: str | Path,
    device: torch.device | str | None = None,
    download: bool = True,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> AutoTrainingSnapshotBatchwiseLossSamplingEvaluation:
    """Repeatedly sample finite-shot or exact batch losses from one snapshot checkpoint."""

    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if repetition_block_size <= 0:
        raise ValueError(f"repetition_block_size must be positive, got {repetition_block_size}.")
    normalized_shot_budgets = tuple(
        None if shot_budget is None else int(shot_budget)
        for shot_budget in shot_budgets
    )
    if not normalized_shot_budgets:
        raise ValueError("shot_budgets must not be empty.")
    if any(shot_budget is not None and shot_budget <= 0 for shot_budget in normalized_shot_budgets):
        raise ValueError(f"shot_budgets must contain only positive integers, got {list(shot_budgets)!r}.")

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=None,
        root=root,
        device=device,
        download=download,
    )
    model = context.runner.model
    exact_quantum_readout = getattr(model, "exact_quantum_readout_probabilities", None)
    classify_readout = getattr(model, "classify_readout_histogram", None)
    if not callable(exact_quantum_readout) or not callable(classify_readout):
        raise ValueError(
            "Figure 7 loss sampling requires a model exposing "
            "exact_quantum_readout_probabilities(...) and classify_readout_histogram(...)."
        )

    test_loader = DataLoader(
        context.test_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    num_batches = len(test_loader)
    batch_sizes = torch.empty(num_batches, dtype=torch.int64)
    histogram_layers = {
        shot_budget: model.readout_histogram.__class__(shot_budget=shot_budget).eval()
        for shot_budget in normalized_shot_budgets
        if shot_budget is not None
    }
    batch_loss_sum = {
        shot_budget: torch.empty(
            ((1 if shot_budget is None else repetitions), num_batches),
            dtype=torch.float32,
        )
        for shot_budget in normalized_shot_budgets
    }
    batch_correct_count = {
        shot_budget: torch.empty(
            ((1 if shot_budget is None else repetitions), num_batches),
            dtype=torch.int64,
        )
        for shot_budget in normalized_shot_budgets
    }

    model.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            device_images = images.to(context.device, non_blocking=True)
            device_labels = labels.to(device=context.device, dtype=torch.long, non_blocking=True)
            current_batch_size = int(device_labels.shape[0])
            batch_sizes[batch_index] = current_batch_size
            exact_readout = exact_quantum_readout(device_images)

            for shot_budget in normalized_shot_budgets:
                if shot_budget is None:
                    logits = classify_readout(exact_readout)
                    per_sample_loss = F.cross_entropy(
                        logits,
                        device_labels,
                        reduction="none",
                    )
                    correct = logits.argmax(dim=-1).eq(device_labels)
                    batch_loss_sum[shot_budget][0, batch_index] = per_sample_loss.sum().detach().cpu()
                    batch_correct_count[shot_budget][0, batch_index] = (
                        correct.sum().to(dtype=torch.int64).detach().cpu()
                    )
                    continue

                histogram_layer = histogram_layers[shot_budget]
                written = 0
                while written < repetitions:
                    current_block_size = min(repetition_block_size, repetitions - written)
                    sampled_histograms = histogram_layer.sample_repeated_histograms(
                        exact_readout,
                        repetitions=current_block_size,
                        block_size=current_block_size,
                    )
                    logits = classify_readout(sampled_histograms)
                    flattened_logits = logits.reshape(current_block_size * current_batch_size, -1)
                    repeated_labels = device_labels.unsqueeze(0).expand(current_block_size, -1)
                    per_sample_loss = F.cross_entropy(
                        flattened_logits,
                        repeated_labels.reshape(-1),
                        reduction="none",
                    ).reshape(current_block_size, current_batch_size)
                    correct = logits.argmax(dim=-1).eq(repeated_labels)
                    batch_loss_sum[shot_budget][written : written + current_block_size, batch_index] = (
                        per_sample_loss.sum(dim=1).detach().cpu()
                    )
                    batch_correct_count[shot_budget][written : written + current_block_size, batch_index] = (
                        correct.sum(dim=1).to(dtype=torch.int64).detach().cpu()
                    )
                    written += current_block_size
            if batch_progress_callback is not None:
                batch_progress_callback(batch_index + 1, num_batches)

    evaluations = [
        AutoTrainingSnapshotBatchwiseShotBudgetEvaluation(
            shot_budget=shot_budget,
            num_draws=(1 if shot_budget is None else repetitions),
            batch_sizes=batch_sizes.clone(),
            batch_loss_sum=batch_loss_sum[shot_budget],
            batch_correct_count=batch_correct_count[shot_budget],
        )
        for shot_budget in normalized_shot_budgets
    ]
    return AutoTrainingSnapshotBatchwiseLossSamplingEvaluation(
        context=context,
        repetitions=repetitions,
        batch_size=batch_size,
        repetition_block_size=repetition_block_size,
        evaluations=evaluations,
    )


_READOUT_LANDSCAPE_LOCAL_PCA_ITERATIONS = 8
_READOUT_LANDSCAPE_LOCAL_PCA_RELATIVE_EIGENVALUE_TOLERANCE = 1e-6
_READOUT_LANDSCAPE_VALIDITY_TOLERANCE = 1e-5


def _project_to_zero_sum_hyperplane(vector: torch.Tensor) -> torch.Tensor:
    return vector - vector.mean(dim=-1, keepdim=True)


def _canonicalize_batched_direction_sign(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"Expected batched direction matrix with shape [B, D], got {tuple(vectors.shape)}.")
    pivot_indices = vectors.abs().argmax(dim=1, keepdim=True)
    pivot_values = vectors.gather(1, pivot_indices)
    sign = torch.where(pivot_values < 0.0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
    return vectors * sign


def _normalize_batched_readout_vectors(
    vectors: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2:
        raise ValueError(f"Expected batched vector matrix with shape [B, D], got {tuple(vectors.shape)}.")
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    valid = torch.isfinite(norms[:, 0]) & (norms[:, 0] > epsilon)
    safe_norms = torch.where(valid[:, None], norms, torch.ones_like(norms))
    normalized = vectors / safe_norms
    normalized = torch.where(valid[:, None], normalized, torch.zeros_like(normalized))
    return normalized, valid


def _orthonormalize_batched_readout_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_zero_sum = _project_to_zero_sum_hyperplane(first)
    q1, valid_first = _normalize_batched_readout_vectors(first_zero_sum, epsilon=epsilon)
    second_zero_sum = _project_to_zero_sum_hyperplane(second)
    second_residual = second_zero_sum - (second_zero_sum * q1).sum(dim=1, keepdim=True) * q1
    q2, valid_second = _normalize_batched_readout_vectors(second_residual, epsilon=epsilon)
    valid = valid_first & valid_second
    q1 = _canonicalize_batched_direction_sign(q1)
    q2 = _canonicalize_batched_direction_sign(q2)
    q1 = torch.where(valid[:, None], q1, torch.zeros_like(q1))
    q2 = torch.where(valid[:, None], q2, torch.zeros_like(q2))
    return q1, q2, valid


def _build_deterministic_local_pca_initial_pair(
    *,
    flattened_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = torch.arange(flattened_dim, dtype=dtype, device=device)
    centered = coordinates - coordinates.mean()
    first = _project_to_zero_sum_hyperplane(centered)
    second = _project_to_zero_sum_hyperplane(centered.square())
    first_norm_sq = torch.dot(first, first)
    if torch.isfinite(first_norm_sq) and first_norm_sq > 0.0:
        second = second - (torch.dot(second, first) / first_norm_sq) * first
    return first, second


def _apply_shot_noise_covariance_to_batched_vectors(
    probabilities: torch.Tensor,
    vectors: torch.Tensor,
    *,
    sigma_shot_budget: int,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probability matrix with shape [B, D], got {tuple(probabilities.shape)}.")
    squeeze_output = False
    if vectors.ndim == 2:
        vectors = vectors.unsqueeze(-1)
        squeeze_output = True
    if vectors.ndim != 3 or vectors.shape[:2] != probabilities.shape:
        raise ValueError(
            "Shot-noise covariance matvec expects vectors shaped [B, D] or [B, D, K] with the same "
            f"leading dimensions as probabilities; got probabilities={tuple(probabilities.shape)} "
            f"and vectors={tuple(vectors.shape)}."
        )
    projected = (probabilities.unsqueeze(-1) * vectors).sum(dim=1, keepdim=True)
    result = probabilities.unsqueeze(-1) * (vectors - projected)
    result = result / float(sigma_shot_budget)
    return result.squeeze(-1) if squeeze_output else result


def _estimate_local_readout_landscape_basis(
    probabilities: torch.Tensor,
    *,
    sigma_shot_budget: int,
    iteration_count: int = _READOUT_LANDSCAPE_LOCAL_PCA_ITERATIONS,
    relative_eigenvalue_tolerance: float = _READOUT_LANDSCAPE_LOCAL_PCA_RELATIVE_EIGENVALUE_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probability matrix with shape [B, D], got {tuple(probabilities.shape)}.")
    if sigma_shot_budget <= 0:
        raise ValueError(f"sigma_shot_budget must be positive, got {sigma_shot_budget}.")
    if iteration_count <= 0:
        raise ValueError(f"iteration_count must be positive, got {iteration_count}.")
    if relative_eigenvalue_tolerance <= 0.0:
        raise ValueError(
            f"relative_eigenvalue_tolerance must be positive, got {relative_eigenvalue_tolerance}."
        )

    batch_size, flattened_dim = probabilities.shape
    if flattened_dim < 3:
        zero_directions = torch.zeros_like(probabilities)
        return zero_directions, zero_directions.clone(), torch.zeros(batch_size, dtype=torch.bool, device=probabilities.device)

    epsilon = torch.finfo(probabilities.dtype).eps * 10.0
    initial_u, initial_v = _build_deterministic_local_pca_initial_pair(
        flattened_dim=flattened_dim,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    q1, q2, _ = _orthonormalize_batched_readout_pair(
        initial_u.expand(batch_size, -1).clone(),
        initial_v.expand(batch_size, -1).clone(),
        epsilon=epsilon,
    )

    for _ in range(iteration_count):
        block = torch.stack((q1, q2), dim=-1)
        updated = _apply_shot_noise_covariance_to_batched_vectors(
            probabilities,
            block,
            sigma_shot_budget=sigma_shot_budget,
        )
        q1, q2, _ = _orthonormalize_batched_readout_pair(
            updated[:, :, 0],
            updated[:, :, 1],
            epsilon=epsilon,
        )

    block = torch.stack((q1, q2), dim=-1)
    covariance_times_block = _apply_shot_noise_covariance_to_batched_vectors(
        probabilities,
        block,
        sigma_shot_budget=sigma_shot_budget,
    )
    small_gram = torch.matmul(block.transpose(1, 2), covariance_times_block)
    small_gram = 0.5 * (small_gram + small_gram.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(small_gram)
    eigenvalues = torch.flip(eigenvalues, dims=(1,))
    eigenvectors = torch.flip(eigenvectors, dims=(2,))
    principal_directions = torch.matmul(block, eigenvectors)
    principal_u = _canonicalize_batched_direction_sign(principal_directions[:, :, 0])
    principal_v = _canonicalize_batched_direction_sign(principal_directions[:, :, 1])

    max_eigenvalue = torch.amax(torch.abs(eigenvalues), dim=1)
    eigenvalue_tolerance = torch.maximum(
        torch.full_like(max_eigenvalue, 1e-12),
        max_eigenvalue * relative_eigenvalue_tolerance,
    )
    valid = (
        torch.isfinite(eigenvalues[:, 0])
        & torch.isfinite(eigenvalues[:, 1])
        & (eigenvalues[:, 0] > eigenvalue_tolerance)
        & (eigenvalues[:, 1] > eigenvalue_tolerance)
    )
    standard_deviations = torch.sqrt(torch.clamp(eigenvalues[:, :2], min=0.0))
    direction_u = principal_u * standard_deviations[:, 0:1]
    direction_v = principal_v * standard_deviations[:, 1:2]
    direction_u = torch.where(valid[:, None], direction_u, torch.zeros_like(direction_u))
    direction_v = torch.where(valid[:, None], direction_v, torch.zeros_like(direction_v))
    return direction_u, direction_v, valid


def _build_symmetric_landscape_axis_values(
    *,
    points: int,
    axis_limit: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if axis_limit <= 0.0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}.")
    values = torch.linspace(-axis_limit, axis_limit, points, dtype=dtype, device=device)
    if points % 2 == 1:
        values[points // 2] = 0.0
    return values


def evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
    root: str | Path,
    device: torch.device | str | None = None,
    alpha_beta_points: int = 81,
    axis_limit: float = 1.0,
    sample_batch_size: int = 128,
    grid_chunk_size: int = 512,
    sigma_shot_budget: int = 128,
    download: bool = True,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> AutoTrainingSnapshotReadoutLandscapeEvaluation:
    """Evaluate the S3 loss landscape on perturbed exact readout distributions."""

    if alpha_beta_points <= 1:
        raise ValueError(f"alpha_beta_points must be at least 2, got {alpha_beta_points}.")
    if axis_limit <= 0.0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}.")
    if sample_batch_size <= 0:
        raise ValueError(f"sample_batch_size must be positive, got {sample_batch_size}.")
    if grid_chunk_size <= 0:
        raise ValueError(f"grid_chunk_size must be positive, got {grid_chunk_size}.")
    if sigma_shot_budget <= 0:
        raise ValueError(f"sigma_shot_budget must be positive, got {sigma_shot_budget}.")

    context = reconstruct_auto_training_snapshot_runner_and_test_loader(
        run_directory,
        seed=seed,
        epoch=epoch,
        shot_budget=None,
        root=root,
        device=device,
        download=download,
    )
    model = context.runner.model
    exact_quantum_readout = getattr(model, "exact_quantum_readout_probabilities", None)
    classify_readout = getattr(model, "classify_readout_histogram", None)
    if not callable(exact_quantum_readout) or not callable(classify_readout):
        raise ValueError(
            "Figure S3 readout-landscape evaluation requires a model exposing "
            "exact_quantum_readout_probabilities(...) and classify_readout_histogram(...)."
        )

    test_loader = DataLoader(
        context.test_loader.dataset,
        batch_size=sample_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    num_batches = len(test_loader)
    total_samples = len(context.test_loader.dataset)
    if total_samples <= 0:
        raise ValueError("Figure S3 readout-landscape evaluation requires a non-empty test split.")

    model.eval()
    alpha_values = _build_symmetric_landscape_axis_values(
        points=alpha_beta_points,
        axis_limit=axis_limit,
        dtype=torch.float32,
        device=context.device,
    )
    beta_values = _build_symmetric_landscape_axis_values(
        points=alpha_beta_points,
        axis_limit=axis_limit,
        dtype=torch.float32,
        device=context.device,
    )
    flat_grid_alpha = alpha_values.repeat(alpha_beta_points)
    flat_grid_beta = beta_values.repeat_interleave(alpha_beta_points)
    total_grid_points = flat_grid_alpha.numel()

    loss_sum = torch.zeros(total_grid_points, dtype=torch.float32, device=context.device)
    valid_count = torch.zeros(total_grid_points, dtype=torch.int64, device=context.device)
    eligible_sample_count = 0

    with torch.no_grad():
        expected_spatial_shape: tuple[int, int, int] | None = None

        for batch_index, (images, labels) in enumerate(test_loader):
            device_images = images.to(context.device, non_blocking=True)
            device_labels = labels.to(device=context.device, dtype=torch.long, non_blocking=True)

            exact_readout = exact_quantum_readout(device_images)
            current_spatial_shape = tuple(exact_readout.shape[-3:])
            if expected_spatial_shape is None:
                expected_spatial_shape = current_spatial_shape
            elif current_spatial_shape != expected_spatial_shape:
                raise ValueError(
                    "All S3 readout batches must share the same spatial readout shape, "
                    f"got {current_spatial_shape} after {expected_spatial_shape}."
                )

            flattened_readout = exact_readout.reshape(exact_readout.shape[0], -1)
            local_direction_u, local_direction_v, eligible_samples = _estimate_local_readout_landscape_basis(
                flattened_readout,
                sigma_shot_budget=sigma_shot_budget,
            )
            if eligible_samples.any():
                eligible_readout = flattened_readout[eligible_samples]
                eligible_labels = device_labels[eligible_samples]
                eligible_direction_u = local_direction_u[eligible_samples]
                eligible_direction_v = local_direction_v[eligible_samples]
                eligible_sample_count += int(eligible_samples.sum().item())
            else:
                if batch_progress_callback is not None:
                    batch_progress_callback(batch_index + 1, num_batches)
                continue

            for grid_start in range(0, total_grid_points, grid_chunk_size):
                grid_stop = min(grid_start + grid_chunk_size, total_grid_points)
                chunk_alpha = flat_grid_alpha[grid_start:grid_stop]
                chunk_beta = flat_grid_beta[grid_start:grid_stop]
                perturbation = (
                    chunk_alpha[None, :, None] * eligible_direction_u[:, None, :]
                    + chunk_beta[None, :, None] * eligible_direction_v[:, None, :]
                )
                perturbed = eligible_readout[:, None, :] + perturbation
                component_valid = (perturbed >= 0.0).all(dim=-1)
                sum_valid = perturbed.sum(dim=-1) <= (1.0 + _READOUT_LANDSCAPE_VALIDITY_TOLERANCE)
                valid_mask = component_valid & sum_valid
                if not valid_mask.any():
                    continue

                valid_grid_indices = (
                    torch.arange(grid_start, grid_stop, device=context.device)[None, :].expand_as(valid_mask)[valid_mask]
                )
                valid_distributions = perturbed[valid_mask]
                valid_labels = eligible_labels[:, None].expand(-1, grid_stop - grid_start)[valid_mask]
                logits = classify_readout(valid_distributions.reshape(-1, *current_spatial_shape))
                per_sample_loss = F.cross_entropy(logits, valid_labels, reduction="none")
                loss_sum.index_add_(0, valid_grid_indices, per_sample_loss)
                valid_count.index_add_(
                    0,
                    valid_grid_indices,
                    torch.ones_like(valid_grid_indices, dtype=torch.int64),
                )

            if batch_progress_callback is not None:
                batch_progress_callback(batch_index + 1, num_batches)

    mean_loss = torch.zeros_like(loss_sum)
    valid_mask = valid_count > 0
    mean_loss[valid_mask] = loss_sum[valid_mask] / valid_count[valid_mask].to(dtype=loss_sum.dtype)
    valid_fraction = valid_count.to(dtype=torch.float32) / float(total_samples)

    return AutoTrainingSnapshotReadoutLandscapeEvaluation(
        context=context,
        alpha=alpha_values.detach().cpu(),
        beta=beta_values.detach().cpu(),
        mean_loss=mean_loss.reshape(alpha_beta_points, alpha_beta_points).detach().cpu(),
        valid_count=valid_count.reshape(alpha_beta_points, alpha_beta_points).detach().cpu(),
        valid_fraction=valid_fraction.reshape(alpha_beta_points, alpha_beta_points).detach().cpu(),
        total_samples=total_samples,
        eligible_sample_count=eligible_sample_count,
    )


def _load_auto_training_manifest(run_directory: str | Path) -> dict[str, Any]:
    manifest_path = Path(run_directory) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run directory is missing manifest.json: {manifest_path.parent}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest.json must deserialize to a mapping.")
    return dict(manifest)


def _resolve_snapshot_checkpoint_path(
    run_directory: str | Path,
    *,
    seed: int,
    epoch: int,
) -> Path:
    resolved_directory = Path(run_directory)
    manifest = _load_auto_training_manifest(resolved_directory)
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("manifest.json must contain a 'runs' list.")

    matching_run = None
    for run_entry in runs:
        if isinstance(run_entry, Mapping) and run_entry.get("seed") == seed:
            matching_run = run_entry
            break
    if not isinstance(matching_run, Mapping):
        raise ValueError(f"manifest.json does not contain a run entry for seed {seed}.")

    snapshots = matching_run.get("snapshots")
    if not isinstance(snapshots, Mapping):
        raise ValueError(f"manifest.json run entry for seed {seed} is missing 'snapshots'.")
    snapshot_name = snapshots.get(str(epoch))
    if not isinstance(snapshot_name, str) or not snapshot_name:
        raise ValueError(
            f"Snapshot checkpoint for epoch {epoch} is missing in run {resolved_directory} seed {seed}."
        )
    checkpoint_path = resolved_directory / snapshot_name
    if not checkpoint_path.is_file():
        raise ValueError(f"Snapshot checkpoint file is missing: {checkpoint_path}")
    return checkpoint_path


def _override_model_spec_shot_budget(spec: ModelSpec, *, shot_budget: int | None) -> ModelSpec:
    constructor_kwargs = dict(spec.constructor_kwargs)
    if "shot_budget" not in constructor_kwargs:
        if spec.module == "qcnn.hybrid" and spec.class_name in {"PCSQCNN", "PCSQCNNNoQFT"}:
            constructor_kwargs["shot_budget"] = shot_budget
        elif shot_budget is None:
            return spec
        else:
            raise ValueError(
                f"Model {spec.module}.{spec.class_name} does not expose a shot_budget constructor argument."
            )
    else:
        constructor_kwargs["shot_budget"] = shot_budget
    return ModelSpec(
        module=spec.module,
        class_name=spec.class_name,
        constructor_kwargs=constructor_kwargs,
    )


def _evaluate_runner_and_collect_predictions(
    runner: ImageClassifierRunner,
    test_loader: DataLoader,
    *,
    batch_progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[EvaluationSummary, torch.Tensor, torch.Tensor]:
    runner.model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    num_batches = len(test_loader)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            device_images = images.to(runner.device, non_blocking=True)
            device_labels = labels.to(device=runner.device, dtype=torch.long, non_blocking=True)
            logits = runner.model(device_images)
            loss = runner.loss_collector.compute_loss(logits, device_labels)
            batch_predictions = logits.argmax(dim=1)
            batch_size = int(device_labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((batch_predictions == device_labels).sum().item())
            total_samples += batch_size
            predictions.append(batch_predictions.detach().cpu())
            targets.append(device_labels.detach().cpu())
            if batch_progress_callback is not None:
                batch_progress_callback(batch_index + 1, num_batches)

    if total_samples == 0:
        raise ValueError("test_loader must yield at least one batch.")

    return (
        EvaluationSummary(
            loss=total_loss / total_samples,
            metrics={"accuracy": total_correct / total_samples},
        ),
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _build_loaded_run_test_report(
    loaded_run: LoadedAutoTrainingRun,
    evaluation_summary: EvaluationSummary,
) -> dict[str, Any]:
    saved_test_summary = loaded_run.final_summaries.get("test", {})
    saved_test_metrics = {}
    saved_test_loss = None
    saved_test_accuracy = None

    if isinstance(saved_test_summary, Mapping):
        saved_test_loss = saved_test_summary.get("loss")
        raw_saved_test_metrics = saved_test_summary.get("metrics")
        if isinstance(raw_saved_test_metrics, Mapping):
            saved_test_metrics = dict(raw_saved_test_metrics)
            saved_test_accuracy = saved_test_metrics.get("accuracy")

    report: dict[str, Any] = {
        "recomputed_test": {
            "loss": evaluation_summary.loss,
            "accuracy": evaluation_summary.metrics["accuracy"],
        },
    }

    if saved_test_loss is not None or saved_test_accuracy is not None:
        saved_section: dict[str, Any] = {}
        if saved_test_loss is not None:
            saved_section["loss"] = float(saved_test_loss)
        if saved_test_accuracy is not None:
            saved_section["accuracy"] = float(saved_test_accuracy)
        report["saved_test"] = saved_section

        delta_section: dict[str, Any] = {}
        if saved_test_loss is not None:
            delta_section["loss"] = evaluation_summary.loss - float(saved_test_loss)
        if saved_test_accuracy is not None:
            delta_section["accuracy"] = (
                evaluation_summary.metrics["accuracy"] - float(saved_test_accuracy)
            )
        if delta_section:
            report["delta"] = delta_section

    return report


def _render_mapping_sections(
    mapping: Mapping[str, Any],
    *,
    level: int,
    float_precision: int,
    sort_keys: bool,
) -> list[str]:
    lines: list[str] = []
    for key, value in _mapping_items(mapping, sort_keys=sort_keys):
        if value is None:
            continue
        lines.extend(
            _render_section(
                label=str(key),
                value=value,
                level=level,
                float_precision=float_precision,
                sort_keys=sort_keys,
            )
        )
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_section(
    *,
    label: str,
    value: Any,
    level: int,
    float_precision: int,
    sort_keys: bool,
) -> list[str]:
    heading = f"{'#' * min(level, 6)} {_prettify_key(label)}"
    lines = [heading, ""]

    if isinstance(value, Mapping):
        scalar_rows: list[str] = []
        nested_lines: list[str] = []

        for child_key, child_value in _mapping_items(value, sort_keys=sort_keys):
            if child_value is None:
                continue
            if _is_mapping_sequence(child_value):
                nested_lines.extend(
                    _render_mapping_sequence(
                        label=str(child_key),
                        items=child_value,
                        level=level + 1,
                        float_precision=float_precision,
                        sort_keys=sort_keys,
                    )
                )
                nested_lines.append("")
                continue
            if isinstance(child_value, Mapping):
                nested_lines.extend(
                    _render_section(
                        label=str(child_key),
                        value=child_value,
                        level=level + 1,
                        float_precision=float_precision,
                        sort_keys=sort_keys,
                    )
                )
                nested_lines.append("")
                continue

            scalar_rows.append(
                f"- {_prettify_key(str(child_key))}: "
                f"{_format_scalar_value(child_value, float_precision=float_precision)}"
            )

        lines.extend(scalar_rows)
        if scalar_rows and nested_lines:
            lines.append("")
        if nested_lines and nested_lines[-1] == "":
            nested_lines.pop()
        lines.extend(nested_lines)
        if len(lines) == 2:
            lines.append("- Empty")
        return lines

    if _is_mapping_sequence(value):
        lines.extend(
            _render_mapping_sequence(
                label=label,
                items=value,
                level=level + 1,
                float_precision=float_precision,
                sort_keys=sort_keys,
            )
        )
        return lines

    lines.append(f"- Value: {_format_scalar_value(value, float_precision=float_precision)}")
    return lines


def _render_mapping_sequence(
    *,
    label: str,
    items: Sequence[Any],
    level: int,
    float_precision: int,
    sort_keys: bool,
) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.extend(
            _render_section(
                label=f"{label} {index}",
                value=item,
                level=level,
                float_precision=float_precision,
                sort_keys=sort_keys,
            )
        )
        lines.append("")

    if not lines:
        lines.append("- Empty")
        return lines

    if lines[-1] == "":
        lines.pop()
    return lines


def _mapping_items(mapping: Mapping[str, Any], *, sort_keys: bool) -> list[tuple[str, Any]]:
    items = [(str(key), value) for key, value in mapping.items()]
    if sort_keys:
        return sorted(items, key=lambda item: item[0])
    return items


def _format_scalar_value(value: Any, *, float_precision: int) -> str:
    plain_value = _to_plain_metadata(value)

    if plain_value is None:
        return "`None`"
    if isinstance(plain_value, bool):
        return f"`{plain_value}`"
    if isinstance(plain_value, int):
        return f"`{plain_value}`"
    if isinstance(plain_value, float):
        return f"`{plain_value:.{float_precision}f}`"
    if isinstance(plain_value, str):
        return f"`{plain_value}`"
    if _is_scalar_sequence(plain_value):
        inline_items = ", ".join(_scalar_item_to_text(item, float_precision=float_precision) for item in plain_value)
        return f"`[{inline_items}]`"
    return f"`{plain_value}`"


def _scalar_item_to_text(value: Any, *, float_precision: int) -> str:
    plain_value = _to_plain_metadata(value)
    if plain_value is None:
        return "None"
    if isinstance(plain_value, float):
        return f"{plain_value:.{float_precision}f}"
    return str(plain_value)


def _is_mapping_sequence(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    if len(value) == 0:
        return False
    return all(isinstance(item, Mapping) for item in value)


def _is_scalar_sequence(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    return all(
        not isinstance(item, Mapping)
        and not _is_mapping_sequence(item)
        for item in value
    )


def _prettify_key(raw_key: str) -> str:
    upper_tokens = {"id", "mnist", "qcnn", "frqi", "utc"}
    words = raw_key.replace("-", "_").split("_")
    return " ".join(
        word.upper() if word.lower() in upper_tokens else word.capitalize()
        for word in words
        if word
    )


def _infer_module_device(module: nn.Module) -> torch.device:
    first_parameter = next(module.parameters(), None)
    if first_parameter is not None:
        return first_parameter.device

    first_buffer = next(module.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device

    return torch.device("cpu")


def _to_plain_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_metadata(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, tuple):
        return [_to_plain_metadata(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_metadata(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_plain_metadata(item) for item in value]
    return repr(value)
