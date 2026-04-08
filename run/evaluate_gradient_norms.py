from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
from typing import Any, Callable, Sequence
import warnings

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
from torchvision.datasets import MNIST

from qcnn.hybrid import PCSQCNN
from qcnn.serialization import resolve_snapshot_trainable_layer_blocks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIRECTORY_NAME = "depth_scaling_gradient_norms"
DEFAULT_OUTPUT_FILENAME = "gradient_norms.pt"
DEFAULT_TASK_RESULT_FILENAME = "result.pt"
DEFAULT_TASK_FORMAT_VERSION = 2
TEST_SHIFT_SEED_OFFSET = 2_000_033

# Canonical scientific configuration for the init-time gradient figure.
DEPTHS: tuple[int, ...] = tuple(range(1, 9))
POST_POOLING_INDEX_QUBITS = 1
FEATURE_QUBITS = 3
NUM_CLASSES = 10
PARAM_SEED_COUNT = 12
DATA_SEED = 0
NUM_TEST_SAMPLES = 256
CLASS_BALANCED_SUBSET = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_BATCH_SIZE_BY_IMAGE_SIZE = {
    2: 128,
    4: 128,
    8: 64,
    16: 64,
    32: 32,
    64: 16,
    128: 8,
    256: 4,
}

# Local safety guard for the current RTX 3060 Laptop GPU machine.
LOCAL_SAFE_MAX_IMAGE_SIZE = 256
LOCAL_SAFE_MAX_PARAM_SEED_COUNT = 12
LOCAL_SAFE_MAX_NUM_TEST_SAMPLES = 256
LOCAL_SAFE_MAX_TOTAL_WORK = 3072


def _build_progress_columns() -> tuple[object, ...]:
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
    )


class _GradientNormProgress:
    def __init__(self, *, total_tasks: int, console: Console | None = None) -> None:
        self.console = Console() if console is None else console
        self.progress = Progress(*_build_progress_columns(), console=self.console)
        self.tasks_task_id = self.progress.add_task("Tasks", total=total_tasks)
        self.samples_task_id = self.progress.add_task("Samples", total=1, visible=False)

    def __enter__(self) -> _GradientNormProgress:
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        try:
            self.progress.__exit__(exc_type, exc, exc_tb)
        finally:
            if not self.console.is_terminal:
                self.console.file.write("\n")
                flush = getattr(self.console.file, "flush", None)
                if callable(flush):
                    flush()

    @staticmethod
    def _build_description(
        *,
        prefix: str,
        mode: str,
        depth: int,
        image_size: int,
        param_seed: int,
    ) -> str:
        return f"{prefix} [{mode}] Q={depth} image={image_size} seed={param_seed}"

    def show_cached_task(self, *, depth: int, image_size: int, param_seed: int) -> None:
        self.progress.update(
            self.tasks_task_id,
            description=self._build_description(
                prefix="Tasks",
                mode="cached",
                depth=depth,
                image_size=image_size,
                param_seed=param_seed,
            ),
        )
        self.progress.update(self.samples_task_id, visible=False)

    def start_running_task(
        self,
        *,
        depth: int,
        image_size: int,
        param_seed: int,
        total_samples: int,
    ) -> None:
        task_description = self._build_description(
            prefix="Tasks",
            mode="running",
            depth=depth,
            image_size=image_size,
            param_seed=param_seed,
        )
        samples_description = self._build_description(
            prefix="Samples",
            mode="running",
            depth=depth,
            image_size=image_size,
            param_seed=param_seed,
        )
        self.progress.update(self.tasks_task_id, description=task_description)
        self.progress.update(
            self.samples_task_id,
            description=samples_description,
            total=total_samples,
            completed=0,
            visible=True,
        )

    def advance_samples(self, batch_size: int) -> None:
        self.progress.advance(self.samples_task_id, batch_size)

    def mark_task_saving(self, *, depth: int, image_size: int, param_seed: int) -> None:
        task_description = self._build_description(
            prefix="Tasks",
            mode="saving",
            depth=depth,
            image_size=image_size,
            param_seed=param_seed,
        )
        samples_description = self._build_description(
            prefix="Samples",
            mode="saving",
            depth=depth,
            image_size=image_size,
            param_seed=param_seed,
        )
        self.progress.update(self.tasks_task_id, description=task_description)
        self.progress.update(self.samples_task_id, description=samples_description, visible=True)

    def complete_task(self) -> None:
        self.progress.advance(self.tasks_task_id)
        self.progress.update(self.samples_task_id, visible=False)

    def emit_oom_reduction(
        self,
        *,
        depth: int,
        param_seed: int,
        old_batch_size: int,
        new_batch_size: int,
    ) -> None:
        self.console.print(
            f"OOM at Q={depth} seed={param_seed}: reducing microbatch "
            f"{old_batch_size} -> {new_batch_size}"
        )


def create_progress_reporter(*, total_tasks: int) -> _GradientNormProgress:
    return _GradientNormProgress(total_tasks=total_tasks)


def resolve_default_device(device: str | None) -> str:
    if device is not None:
        return device
    return DEVICE


def build_output_directory(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root) / DEFAULT_OUTPUT_DIRECTORY_NAME


def build_task_directory(
    artifacts_root: str | Path,
    *,
    depth: int,
    param_seed: int,
) -> Path:
    return build_output_directory(artifacts_root) / f"depth{depth:02d}_pseed{param_seed:02d}"


def build_depth_scaling_image_size(depth: int) -> int:
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}.")
    return 2 ** (POST_POOLING_INDEX_QUBITS + depth - 1)


def build_param_seeds() -> tuple[int, ...]:
    return tuple(range(PARAM_SEED_COUNT))


def build_scaled_image_size(image_size: int) -> int:
    return max(1, image_size // 2)


def build_max_offset(*, image_size: int, scaled_image_size: int) -> int:
    return max(0, (image_size - scaled_image_size) // 2)


def run_preflight_check() -> None:
    max_image_size = max(build_depth_scaling_image_size(depth) for depth in DEPTHS)
    total_work = PARAM_SEED_COUNT * NUM_TEST_SAMPLES
    violations: list[str] = []

    if max_image_size > LOCAL_SAFE_MAX_IMAGE_SIZE:
        violations.append(
            f"max image_size {max_image_size} exceeds LOCAL_SAFE_MAX_IMAGE_SIZE={LOCAL_SAFE_MAX_IMAGE_SIZE}"
        )
    if PARAM_SEED_COUNT > LOCAL_SAFE_MAX_PARAM_SEED_COUNT:
        violations.append(
            "PARAM_SEED_COUNT "
            f"{PARAM_SEED_COUNT} exceeds LOCAL_SAFE_MAX_PARAM_SEED_COUNT={LOCAL_SAFE_MAX_PARAM_SEED_COUNT}"
        )
    if NUM_TEST_SAMPLES > LOCAL_SAFE_MAX_NUM_TEST_SAMPLES:
        violations.append(
            "NUM_TEST_SAMPLES "
            f"{NUM_TEST_SAMPLES} exceeds LOCAL_SAFE_MAX_NUM_TEST_SAMPLES={LOCAL_SAFE_MAX_NUM_TEST_SAMPLES}"
        )
    if total_work > LOCAL_SAFE_MAX_TOTAL_WORK:
        violations.append(
            f"TOTAL_WORK={total_work} exceeds LOCAL_SAFE_MAX_TOTAL_WORK={LOCAL_SAFE_MAX_TOTAL_WORK}"
        )

    missing_batch_sizes = [
        str(build_depth_scaling_image_size(depth))
        for depth in DEPTHS
        if build_depth_scaling_image_size(depth) not in GRAD_BATCH_SIZE_BY_IMAGE_SIZE
    ]
    if missing_batch_sizes:
        violations.append(
            "GRAD_BATCH_SIZE_BY_IMAGE_SIZE is missing entries for image sizes "
            + ", ".join(missing_batch_sizes)
        )

    if violations:
        raise ValueError(
            "Refusing to run gradient diagnostics on this machine because the explicit "
            "configuration exceeds the local safety guard:\n- "
            + "\n- ".join(violations)
        )


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_mnist_test_tensors(root: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MNIST(root=str(root), train=False, download=True)
    return dataset.data.clone(), dataset.targets.clone()


def build_balanced_test_subset_indices(
    labels: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    unique_labels = torch.unique(labels, sorted=True)
    num_classes = int(unique_labels.numel())
    if num_classes < 1:
        raise ValueError("Expected at least one class in the test labels.")

    base_count = num_samples // num_classes
    remainder = num_samples % num_classes
    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices: list[torch.Tensor] = []
    for class_index, class_id in enumerate(unique_labels):
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        target_count = base_count + (1 if class_index < remainder else 0)
        if target_count == 0:
            continue
        if class_indices.numel() < target_count:
            raise ValueError(
                f"Class {int(class_id)} only has {class_indices.numel()} samples, "
                f"cannot select {target_count}."
            )
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        selected_indices.append(class_indices[permutation[:target_count]])

    if not selected_indices:
        raise ValueError("Balanced test-subset selection produced no indices.")
    return torch.sort(torch.cat(selected_indices)).values


def _normalize_and_resize_images(
    images: torch.Tensor,
    *,
    scaled_image_size: int,
) -> torch.Tensor:
    float_images = images.to(dtype=torch.float32) / 255.0
    return F.interpolate(
        float_images[:, None, :, :],
        size=(scaled_image_size, scaled_image_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _build_test_offsets(
    *,
    num_images: int,
    max_offset: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_offset <= 0:
        zeros = torch.zeros(num_images, dtype=torch.int64)
        return zeros, zeros

    generator = torch.Generator()
    generator.manual_seed(seed + TEST_SHIFT_SEED_OFFSET)
    row_offsets = torch.randint(
        -max_offset,
        max_offset + 1,
        (num_images,),
        generator=generator,
    )
    col_offsets = torch.randint(
        -max_offset,
        max_offset + 1,
        (num_images,),
        generator=generator,
    )
    return row_offsets, col_offsets


def _place_images_on_canvas(
    images: torch.Tensor,
    *,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
    seed: int,
) -> torch.Tensor:
    if scaled_image_size == image_size and max_offset == 0:
        return images

    num_images = int(images.shape[0])
    canvas = torch.zeros((num_images, image_size, image_size), dtype=images.dtype)
    base_top = (image_size - scaled_image_size) // 2
    base_left = (image_size - scaled_image_size) // 2
    row_offsets, col_offsets = _build_test_offsets(
        num_images=num_images,
        max_offset=max_offset,
        seed=seed,
    )

    for index in range(num_images):
        top = base_top + int(row_offsets[index].item())
        left = base_left + int(col_offsets[index].item())
        canvas[index, top : top + scaled_image_size, left : left + scaled_image_size] = images[index]

    return canvas


def prepare_test_subset_for_depth(
    *,
    raw_test_images: torch.Tensor,
    raw_test_labels: torch.Tensor,
    subset_indices: torch.Tensor,
    depth: int,
    data_seed: int,
) -> dict[str, Any]:
    image_size = build_depth_scaling_image_size(depth)
    scaled_image_size = build_scaled_image_size(image_size)
    max_offset = build_max_offset(image_size=image_size, scaled_image_size=scaled_image_size)

    selected_images = raw_test_images[subset_indices]
    selected_labels = raw_test_labels[subset_indices].to(dtype=torch.long)
    resized_images = _normalize_and_resize_images(
        selected_images,
        scaled_image_size=scaled_image_size,
    )
    prepared_images = _place_images_on_canvas(
        resized_images,
        image_size=image_size,
        scaled_image_size=scaled_image_size,
        max_offset=max_offset,
        seed=data_seed,
    )
    return {
        "depth": depth,
        "image_size": image_size,
        "scaled_image_size": scaled_image_size,
        "max_offset": max_offset,
        "images": prepared_images.contiguous(),
        "labels": selected_labels.contiguous(),
    }


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def build_task_manifest(
    *,
    param_seed: int,
    depth: int,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
) -> dict[str, Any]:
    return {
        "task_type": "depth_scaling_gradient_norms",
        "format_version": DEFAULT_TASK_FORMAT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "param_seed": param_seed,
        "depth": depth,
        "image_size": image_size,
        "scaled_image_size": scaled_image_size,
        "max_offset": max_offset,
        "post_pooling_index_qubits": POST_POOLING_INDEX_QUBITS,
        "feature_qubits": FEATURE_QUBITS,
        "num_classes": NUM_CLASSES,
        "data_seed": DATA_SEED,
        "num_test_samples": NUM_TEST_SAMPLES,
        "class_balanced_subset": CLASS_BALANCED_SUBSET,
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
        raise ValueError("Gradient diagnostic task result must deserialize to a mapping.")
    return dict(payload)


def is_compatible_task_cache(
    output_directory: str | Path,
    *,
    param_seed: int,
    depth: int,
    image_size: int,
    scaled_image_size: int,
    max_offset: int,
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
    if manifest.get("task_type") != "depth_scaling_gradient_norms":
        return False
    if manifest.get("format_version") != DEFAULT_TASK_FORMAT_VERSION:
        return False

    expected_values = {
        "param_seed": param_seed,
        "depth": depth,
        "image_size": image_size,
        "scaled_image_size": scaled_image_size,
        "max_offset": max_offset,
        "post_pooling_index_qubits": POST_POOLING_INDEX_QUBITS,
        "feature_qubits": FEATURE_QUBITS,
        "num_classes": NUM_CLASSES,
        "data_seed": DATA_SEED,
        "num_test_samples": NUM_TEST_SAMPLES,
        "class_balanced_subset": CLASS_BALANCED_SUBSET,
    }
    return all(manifest.get(key) == value for key, value in expected_values.items()) and (
        resolved_directory / DEFAULT_TASK_RESULT_FILENAME
    ).is_file()


def clear_incompatible_task_caches(*, artifacts_root: Path) -> None:
    for depth in DEPTHS:
        image_size = build_depth_scaling_image_size(depth)
        scaled_image_size = build_scaled_image_size(image_size)
        max_offset = build_max_offset(image_size=image_size, scaled_image_size=scaled_image_size)
        for param_seed in build_param_seeds():
            task_directory = build_task_directory(artifacts_root, depth=depth, param_seed=param_seed)
            if not task_directory.exists():
                continue
            if not is_compatible_task_cache(
                task_directory,
                param_seed=param_seed,
                depth=depth,
                image_size=image_size,
                scaled_image_size=scaled_image_size,
                max_offset=max_offset,
            ):
                shutil.rmtree(task_directory)


def build_parameter_name_groups(
    model: PCSQCNN,
) -> tuple[list[str], list[str], list[str], list[str]]:
    layer_blocks = resolve_snapshot_trainable_layer_blocks(model)
    if not layer_blocks or layer_blocks[-1][0] != "classifier":
        raise ValueError("Expected resolve_snapshot_trainable_layer_blocks() to end with the classifier.")
    quantum_blocks = layer_blocks[:-1]
    if not quantum_blocks:
        raise ValueError("Expected at least one quantum block.")

    layer_keys = [layer_key for layer_key, _, _ in quantum_blocks]
    layer_labels = [layer_label for _, layer_label, _ in quantum_blocks]
    full_quantum_parameter_names = [
        name
        for _, _, group_parameters in quantum_blocks
        for name, _ in group_parameters
    ]
    last_quantum_parameter_names = [name for name, _ in quantum_blocks[-1][2]]
    return layer_keys, layer_labels, full_quantum_parameter_names, last_quantum_parameter_names


def build_per_sample_gradient_function(
    model: PCSQCNN,
) -> tuple[
    OrderedDict[str, torch.Tensor],
    Callable[[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor], OrderedDict[str, torch.Tensor]],
]:
    parameters = OrderedDict(
        (name, parameter.detach())
        for name, parameter in model.named_parameters()
    )
    buffers = OrderedDict(
        (name, buffer.detach())
        for name, buffer in model.named_buffers()
    )

    def loss_for_single_example(
        current_parameters: OrderedDict[str, torch.Tensor],
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.func.functional_call(
            model,
            (current_parameters, buffers),
            (image.unsqueeze(0),),
        )
        return F.cross_entropy(logits, label.unsqueeze(0))

    per_sample_gradient_fn = torch.func.vmap(
        torch.func.grad(loss_for_single_example),
        in_dims=(None, 0, 0),
    )
    return parameters, per_sample_gradient_fn


def compute_per_sample_squared_gradient_norms(
    parameters: OrderedDict[str, torch.Tensor],
    batch_images: torch.Tensor,
    batch_labels: torch.Tensor,
    *,
    per_sample_gradient_fn: Callable[[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor], OrderedDict[str, torch.Tensor]],
    full_quantum_parameter_names: Sequence[str],
    last_quantum_parameter_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    gradients = per_sample_gradient_fn(parameters, batch_images, batch_labels)
    batch_size = int(batch_images.shape[0])
    full_squared_norms = torch.zeros(batch_size, dtype=torch.float64, device=batch_images.device)
    last_squared_norms = torch.zeros(batch_size, dtype=torch.float64, device=batch_images.device)

    for name in full_quantum_parameter_names:
        gradient = gradients[name].reshape(batch_size, -1).to(dtype=torch.float64)
        full_squared_norms += gradient.pow(2).sum(dim=1)

    for name in last_quantum_parameter_names:
        gradient = gradients[name].reshape(batch_size, -1).to(dtype=torch.float64)
        last_squared_norms += gradient.pow(2).sum(dim=1)

    return full_squared_norms, last_squared_norms


def accumulate_microbatched_squared_gradient_norm_sums(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    initial_batch_size: int,
    device: torch.device,
    batch_evaluator: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    on_batch_completed: Callable[[int], None] | None = None,
    on_batch_size_reduced: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_batch_size < 1:
        raise ValueError(f"initial_batch_size must be positive, got {initial_batch_size}.")

    total_full_squared_norm_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_last_squared_norm_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = int(images.shape[0])
    start = 0
    current_batch_size = min(initial_batch_size, sample_count)

    while start < sample_count:
        batch_stop = min(start + current_batch_size, sample_count)
        try:
            batch_full_squared_norms, batch_last_squared_norms = batch_evaluator(
                images[start:batch_stop],
                labels[start:batch_stop],
            )
        except BaseException as exc:
            if not _is_oom_error(exc):
                raise
            if current_batch_size <= 1:
                raise RuntimeError(
                    "Gradient diagnostics exhausted the adaptive microbatch fallback "
                    f"at sample range [{start}:{batch_stop}] on device {device}."
                ) from exc
            old_batch_size = current_batch_size
            current_batch_size = max(1, current_batch_size // 2)
            if on_batch_size_reduced is not None:
                on_batch_size_reduced(old_batch_size, current_batch_size)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        completed_batch_size = batch_stop - start
        total_full_squared_norm_sum += batch_full_squared_norms.sum()
        total_last_squared_norm_sum += batch_last_squared_norms.sum()
        if on_batch_completed is not None:
            on_batch_completed(completed_batch_size)
        start = batch_stop

    return total_full_squared_norm_sum, total_last_squared_norm_sum


def evaluate_depth_seed_gradient_norms(
    *,
    depth: int,
    param_seed: int,
    prepared_images: torch.Tensor,
    prepared_labels: torch.Tensor,
    device: str | torch.device,
    on_batch_completed: Callable[[int], None] | None = None,
    on_batch_size_reduced: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    image_size = build_depth_scaling_image_size(depth)
    scaled_image_size = build_scaled_image_size(image_size)
    max_offset = build_max_offset(image_size=image_size, scaled_image_size=scaled_image_size)
    resolved_device = torch.device(device)
    grad_batch_size = GRAD_BATCH_SIZE_BY_IMAGE_SIZE[image_size]

    _set_global_seeds(param_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = PCSQCNN(
            image_size=image_size,
            num_classes=NUM_CLASSES,
            feature_qubits=FEATURE_QUBITS,
            quantum_layers=depth,
            reduce_readout_to_feature_distribution=False,
        ).to(resolved_device)
    model.eval()

    layer_keys, layer_labels, full_quantum_parameter_names, last_quantum_parameter_names = (
        build_parameter_name_groups(model)
    )
    parameters, per_sample_gradient_fn = build_per_sample_gradient_function(model)

    def batch_evaluator(batch_images: torch.Tensor, batch_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_per_sample_squared_gradient_norms(
            parameters,
            batch_images,
            batch_labels,
            per_sample_gradient_fn=per_sample_gradient_fn,
            full_quantum_parameter_names=full_quantum_parameter_names,
            last_quantum_parameter_names=last_quantum_parameter_names,
        )

    full_squared_norm_sum, last_squared_norm_sum = accumulate_microbatched_squared_gradient_norm_sums(
        prepared_images,
        prepared_labels,
        initial_batch_size=grad_batch_size,
        device=resolved_device,
        batch_evaluator=batch_evaluator,
        on_batch_completed=on_batch_completed,
        on_batch_size_reduced=on_batch_size_reduced,
    )
    sample_count = int(prepared_images.shape[0])
    full_quantum_gradient_rms = torch.sqrt(full_squared_norm_sum / float(sample_count)).to(dtype=torch.float32).cpu()
    last_quantum_layer_gradient_rms = torch.sqrt(last_squared_norm_sum / float(sample_count)).to(
        dtype=torch.float32
    ).cpu()

    return {
        "depth": depth,
        "param_seed": param_seed,
        "image_size": image_size,
        "scaled_image_size": scaled_image_size,
        "max_offset": max_offset,
        "layer_keys": layer_keys,
        "layer_labels": layer_labels,
        "full_quantum_gradient_rms": full_quantum_gradient_rms,
        "last_quantum_layer_gradient_rms": last_quantum_layer_gradient_rms,
    }


def assemble_gradient_norms_payload(*, artifacts_root: str | Path) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []

    for depth in DEPTHS:
        expected_image_size = build_depth_scaling_image_size(depth)
        for param_seed in build_param_seeds():
            task_payload = load_task_result(
                build_task_directory(artifacts_root, depth=depth, param_seed=param_seed)
            )
            if task_payload.get("depth") != depth or task_payload.get("param_seed") != param_seed:
                raise ValueError(
                    f"Cached gradient diagnostic for depth={depth}, param_seed={param_seed} has mismatched identity."
                )
            if task_payload.get("image_size") != expected_image_size:
                raise ValueError(
                    f"Cached gradient diagnostic for depth={depth}, param_seed={param_seed} has mismatched image_size."
                )
            full_quantum_gradient_rms = task_payload.get("full_quantum_gradient_rms")
            last_quantum_layer_gradient_rms = task_payload.get("last_quantum_layer_gradient_rms")
            if not isinstance(full_quantum_gradient_rms, torch.Tensor) or full_quantum_gradient_rms.ndim != 0:
                raise ValueError(
                    "Cached gradient diagnostic is missing a scalar full_quantum_gradient_rms tensor."
                )
            if (
                not isinstance(last_quantum_layer_gradient_rms, torch.Tensor)
                or last_quantum_layer_gradient_rms.ndim != 0
            ):
                raise ValueError(
                    "Cached gradient diagnostic is missing a scalar last_quantum_layer_gradient_rms tensor."
                )
            evaluations.append(
                {
                    "depth": depth,
                    "param_seed": param_seed,
                    "image_size": int(task_payload["image_size"]),
                    "layer_keys": list(task_payload["layer_keys"]),
                    "layer_labels": list(task_payload["layer_labels"]),
                    "full_quantum_gradient_rms": full_quantum_gradient_rms.clone(),
                    "last_quantum_layer_gradient_rms": last_quantum_layer_gradient_rms.clone(),
                }
            )

    return {
        "depths": list(DEPTHS),
        "param_seeds": list(build_param_seeds()),
        "data_seed": DATA_SEED,
        "post_pooling_index_qubits": POST_POOLING_INDEX_QUBITS,
        "feature_qubits": FEATURE_QUBITS,
        "num_test_samples": NUM_TEST_SAMPLES,
        "class_balanced_subset": CLASS_BALANCED_SUBSET,
        "evaluations": evaluations,
    }


def evaluate_gradient_norms(
    *,
    artifacts_root: str | Path = DEFAULT_ARTIFACTS_ROOT,
    root: str | Path = DEFAULT_DATA_ROOT,
    device: str | None = None,
    rebuild: bool = False,
) -> Path:
    run_preflight_check()

    resolved_artifacts_root = Path(artifacts_root).expanduser().resolve()
    resolved_data_root = Path(root).expanduser().resolve()
    resolved_device = resolve_default_device(device)
    output_directory = build_output_directory(resolved_artifacts_root)

    raw_test_images, raw_test_labels = load_raw_mnist_test_tensors(resolved_data_root)
    if CLASS_BALANCED_SUBSET:
        subset_indices = build_balanced_test_subset_indices(
            raw_test_labels,
            num_samples=NUM_TEST_SAMPLES,
            seed=DATA_SEED,
        )
    else:
        subset_indices = torch.arange(min(NUM_TEST_SAMPLES, int(raw_test_labels.shape[0])))

    clear_incompatible_task_caches(artifacts_root=resolved_artifacts_root)
    param_seeds = build_param_seeds()
    total_tasks = len(DEPTHS) * len(param_seeds)

    with create_progress_reporter(total_tasks=total_tasks) as progress:
        for depth in DEPTHS:
            prepared_subset = prepare_test_subset_for_depth(
                raw_test_images=raw_test_images,
                raw_test_labels=raw_test_labels,
                subset_indices=subset_indices,
                depth=depth,
                data_seed=DATA_SEED,
            )
            image_size = int(prepared_subset["image_size"])
            scaled_image_size = int(prepared_subset["scaled_image_size"])
            max_offset = int(prepared_subset["max_offset"])
            prepared_images = prepared_subset["images"].to(resolved_device)
            prepared_labels = prepared_subset["labels"].to(device=resolved_device, dtype=torch.long)

            for param_seed in param_seeds:
                task_directory = build_task_directory(
                    resolved_artifacts_root,
                    depth=depth,
                    param_seed=param_seed,
                )
                if not rebuild and is_compatible_task_cache(
                    task_directory,
                    param_seed=param_seed,
                    depth=depth,
                    image_size=image_size,
                    scaled_image_size=scaled_image_size,
                    max_offset=max_offset,
                ):
                    progress.show_cached_task(
                        depth=depth,
                        image_size=image_size,
                        param_seed=param_seed,
                    )
                    progress.complete_task()
                    continue

                progress.start_running_task(
                    depth=depth,
                    image_size=image_size,
                    param_seed=param_seed,
                    total_samples=int(prepared_images.shape[0]),
                )
                evaluation = evaluate_depth_seed_gradient_norms(
                    depth=depth,
                    param_seed=param_seed,
                    prepared_images=prepared_images,
                    prepared_labels=prepared_labels,
                    device=resolved_device,
                    on_batch_completed=progress.advance_samples,
                    on_batch_size_reduced=lambda old_batch_size, new_batch_size, *, current_depth=depth, current_param_seed=param_seed: progress.emit_oom_reduction(
                        depth=current_depth,
                        param_seed=current_param_seed,
                        old_batch_size=old_batch_size,
                        new_batch_size=new_batch_size,
                    ),
                )
                progress.mark_task_saving(
                    depth=depth,
                    image_size=image_size,
                    param_seed=param_seed,
                )
                save_task_result(
                    task_directory,
                    payload=evaluation,
                    manifest=build_task_manifest(
                        param_seed=param_seed,
                        depth=depth,
                        image_size=image_size,
                        scaled_image_size=scaled_image_size,
                        max_offset=max_offset,
                    ),
                )
                progress.complete_task()

            del prepared_images, prepared_labels
            if torch.cuda.is_available() and torch.device(resolved_device).type == "cuda":
                torch.cuda.empty_cache()

    output_directory.mkdir(parents=True, exist_ok=True)
    payload = assemble_gradient_norms_payload(artifacts_root=resolved_artifacts_root)
    output_path = output_directory / DEFAULT_OUTPUT_FILENAME
    torch.save(payload, output_path)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Figure S2a depth-scaling initialization-time quantum-gradient diagnostics."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional runtime device override. Scientific settings remain fixed in the script constants.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute every cached depth/seed task even when a compatible cache already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = evaluate_gradient_norms(
        device=args.device,
        rebuild=args.rebuild,
    )
    print(output_path)


if __name__ == "__main__":
    main()
