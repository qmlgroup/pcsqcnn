from pathlib import Path

import pytest
import torch

from .script_loading import load_run_script

evaluate_gradient_norms = load_run_script(
    "evaluate_gradient_norms_module",
    "evaluate_gradient_norms.py",
)


class FakeProgressReporter:
    def __init__(self, *, total_tasks: int) -> None:
        self.total_tasks = total_tasks
        self.events: list[tuple[object, ...]] = []

    def __enter__(self) -> "FakeProgressReporter":
        self.events.append(("enter",))
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        del exc_type, exc, exc_tb
        self.events.append(("exit",))

    def show_cached_task(self, *, depth: int, image_size: int, param_seed: int) -> None:
        self.events.append(("cached", depth, image_size, param_seed))

    def start_running_task(
        self,
        *,
        depth: int,
        image_size: int,
        param_seed: int,
        total_samples: int,
    ) -> None:
        self.events.append(("running", depth, image_size, param_seed, total_samples))

    def advance_samples(self, batch_size: int) -> None:
        self.events.append(("advance_samples", batch_size))

    def mark_task_saving(self, *, depth: int, image_size: int, param_seed: int) -> None:
        self.events.append(("saving", depth, image_size, param_seed))

    def complete_task(self) -> None:
        self.events.append(("complete_task",))

    def emit_oom_reduction(
        self,
        *,
        depth: int,
        param_seed: int,
        old_batch_size: int,
        new_batch_size: int,
    ) -> None:
        self.events.append(("oom", depth, param_seed, old_batch_size, new_batch_size))


def make_fake_gradient_evaluation(*, param_seed: int, depth: int) -> dict[str, object]:
    image_size = evaluate_gradient_norms.build_depth_scaling_image_size(depth=depth)
    return {
        "depth": depth,
        "param_seed": param_seed,
        "image_size": image_size,
        "scaled_image_size": image_size // 2,
        "max_offset": max(0, image_size // 4),
        "layer_keys": [f"multiplexers.{layer_index}" for layer_index in range(depth)],
        "layer_labels": [f"Quantum {layer_index + 1}" for layer_index in range(depth)],
        "full_quantum_gradient_rms": torch.tensor(
            0.25 * depth + 0.01 * param_seed,
            dtype=torch.float32,
        ),
        "last_quantum_layer_gradient_rms": torch.tensor(
            0.05 * depth + 0.01 * param_seed,
            dtype=torch.float32,
        ),
    }


def _configure_small_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluate_gradient_norms, "DEPTHS", (1, 2))
    monkeypatch.setattr(evaluate_gradient_norms, "PARAM_SEED_COUNT", 1)
    monkeypatch.setattr(evaluate_gradient_norms, "NUM_TEST_SAMPLES", 8)
    monkeypatch.setattr(
        evaluate_gradient_norms,
        "GRAD_BATCH_SIZE_BY_IMAGE_SIZE",
        {2: 8, 4: 8},
    )


def test_build_depth_scaling_image_size_matches_depth_scaling_formula() -> None:
    assert evaluate_gradient_norms.build_depth_scaling_image_size(depth=1) == 2
    assert evaluate_gradient_norms.build_depth_scaling_image_size(depth=8) == 256


def test_build_balanced_test_subset_indices_is_deterministic_and_balanced() -> None:
    labels = torch.tensor([class_id for class_id in range(10) for _ in range(4)], dtype=torch.long)

    first = evaluate_gradient_norms.build_balanced_test_subset_indices(
        labels,
        num_samples=23,
        seed=17,
    )
    second = evaluate_gradient_norms.build_balanced_test_subset_indices(
        labels,
        num_samples=23,
        seed=17,
    )

    assert torch.equal(first, second)
    selected_labels = labels[first]
    counts = torch.bincount(selected_labels, minlength=10)
    assert counts.sum().item() == 23
    assert counts.min().item() >= 2
    assert counts.max().item() <= 3


def test_parse_args_accepts_device_override_and_rebuild() -> None:
    args = evaluate_gradient_norms.parse_args(
        [
            "--device",
            "cpu",
            "--rebuild",
        ]
    )

    assert args.device == "cpu"
    assert args.rebuild is True


def test_run_preflight_check_refuses_unsafe_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_gradient_norms, "PARAM_SEED_COUNT", 13)
    monkeypatch.setattr(evaluate_gradient_norms, "NUM_TEST_SAMPLES", 400)

    with pytest.raises(ValueError) as excinfo:
        evaluate_gradient_norms.run_preflight_check()

    message = str(excinfo.value)
    assert "PARAM_SEED_COUNT 13 exceeds" in message
    assert "NUM_TEST_SAMPLES 400 exceeds" in message
    assert "TOTAL_WORK=5200 exceeds" in message


def test_accumulate_microbatched_squared_gradient_norm_sums_halves_batch_size_after_oom() -> None:
    images = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    labels = torch.arange(6, dtype=torch.long)
    seen_batch_sizes: list[int] = []
    completed_batches: list[int] = []
    reductions: list[tuple[int, int]] = []

    def fake_batch_evaluator(
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del batch_labels
        batch_size = int(batch_images.shape[0])
        seen_batch_sizes.append(batch_size)
        if batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        full = torch.full((batch_size,), 4.0, dtype=torch.float64)
        last = torch.full((batch_size,), 1.0, dtype=torch.float64)
        return full, last

    full_sum, last_sum = evaluate_gradient_norms.accumulate_microbatched_squared_gradient_norm_sums(
        images,
        labels,
        initial_batch_size=6,
        device=torch.device("cpu"),
        batch_evaluator=fake_batch_evaluator,
        on_batch_completed=completed_batches.append,
        on_batch_size_reduced=lambda old_size, new_size: reductions.append((old_size, new_size)),
    )

    assert seen_batch_sizes == [6, 3, 1, 1, 1, 1, 1, 1]
    assert completed_batches == [1, 1, 1, 1, 1, 1]
    assert reductions == [(6, 3), (3, 1)]
    assert full_sum.item() == pytest.approx(24.0)
    assert last_sum.item() == pytest.approx(6.0)


def test_evaluate_gradient_norms_writes_aggregate_output_and_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_defaults(monkeypatch)
    call_log: list[tuple[int, int]] = []
    fake_progress_holder: dict[str, FakeProgressReporter] = {}

    def fake_load_raw_mnist_test_tensors(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
        del root
        images = torch.zeros((20, 28, 28), dtype=torch.uint8)
        labels = torch.tensor([class_id for class_id in range(10) for _ in range(2)], dtype=torch.long)
        return images, labels

    def fake_evaluate(*, depth, param_seed, **kwargs):
        on_batch_completed = kwargs["on_batch_completed"]
        on_batch_completed(5)
        on_batch_completed(3)
        call_log.append((depth, param_seed))
        return make_fake_gradient_evaluation(param_seed=param_seed, depth=depth)

    def fake_progress_factory(*, total_tasks: int) -> FakeProgressReporter:
        progress = FakeProgressReporter(total_tasks=total_tasks)
        fake_progress_holder["progress"] = progress
        return progress

    monkeypatch.setattr(
        evaluate_gradient_norms,
        "load_raw_mnist_test_tensors",
        fake_load_raw_mnist_test_tensors,
    )
    monkeypatch.setattr(
        evaluate_gradient_norms,
        "evaluate_depth_seed_gradient_norms",
        fake_evaluate,
    )
    monkeypatch.setattr(
        evaluate_gradient_norms,
        "create_progress_reporter",
        fake_progress_factory,
    )

    output_path = evaluate_gradient_norms.evaluate_gradient_norms(
        artifacts_root=tmp_path / "artifacts",
        root=tmp_path / "data",
        device="cpu",
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=False)

    assert output_path == (
        tmp_path
        / "artifacts"
        / evaluate_gradient_norms.DEFAULT_OUTPUT_DIRECTORY_NAME
        / evaluate_gradient_norms.DEFAULT_OUTPUT_FILENAME
    )
    assert output_path.is_file()
    assert payload["depths"] == [1, 2]
    assert payload["param_seeds"] == [0]
    assert payload["data_seed"] == evaluate_gradient_norms.DATA_SEED
    assert payload["post_pooling_index_qubits"] == 1
    assert payload["feature_qubits"] == 3
    assert payload["num_test_samples"] == 8
    assert payload["class_balanced_subset"] is True
    assert len(payload["evaluations"]) == 2
    assert call_log == [(1, 0), (2, 0)]
    assert fake_progress_holder["progress"].total_tasks == 2
    assert fake_progress_holder["progress"].events == [
        ("enter",),
        ("running", 1, 2, 0, 8),
        ("advance_samples", 5),
        ("advance_samples", 3),
        ("saving", 1, 2, 0),
        ("complete_task",),
        ("running", 2, 4, 0, 8),
        ("advance_samples", 5),
        ("advance_samples", 3),
        ("saving", 2, 4, 0),
        ("complete_task",),
        ("exit",),
    ]

    for depth in (1, 2):
        task_directory = evaluate_gradient_norms.build_task_directory(
            tmp_path / "artifacts",
            depth=depth,
            param_seed=0,
        )
        assert (task_directory / "manifest.json").is_file()
        assert (task_directory / evaluate_gradient_norms.DEFAULT_TASK_RESULT_FILENAME).is_file()


def test_evaluate_gradient_norms_reuses_compatible_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_small_defaults(monkeypatch)
    call_log: list[tuple[int, int]] = []
    fake_progress_holder: dict[str, FakeProgressReporter] = {}

    def fake_load_raw_mnist_test_tensors(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
        del root
        images = torch.zeros((20, 28, 28), dtype=torch.uint8)
        labels = torch.tensor([class_id for class_id in range(10) for _ in range(2)], dtype=torch.long)
        return images, labels

    def fake_evaluate(*, depth, param_seed, **kwargs):
        del kwargs
        call_log.append((depth, param_seed))
        return make_fake_gradient_evaluation(param_seed=param_seed, depth=depth)

    def fake_progress_factory(*, total_tasks: int) -> FakeProgressReporter:
        progress = FakeProgressReporter(total_tasks=total_tasks)
        fake_progress_holder["progress"] = progress
        return progress

    monkeypatch.setattr(
        evaluate_gradient_norms,
        "load_raw_mnist_test_tensors",
        fake_load_raw_mnist_test_tensors,
    )
    monkeypatch.setattr(
        evaluate_gradient_norms,
        "evaluate_depth_seed_gradient_norms",
        fake_evaluate,
    )
    monkeypatch.setattr(
        evaluate_gradient_norms,
        "create_progress_reporter",
        fake_progress_factory,
    )

    evaluate_gradient_norms.evaluate_gradient_norms(
        artifacts_root=tmp_path / "artifacts",
        root=tmp_path / "data",
        device="cpu",
    )
    evaluate_gradient_norms.evaluate_gradient_norms(
        artifacts_root=tmp_path / "artifacts",
        root=tmp_path / "data",
        device="cpu",
    )

    assert call_log == [(1, 0), (2, 0)]
    assert fake_progress_holder["progress"].events == [
        ("enter",),
        ("cached", 1, 2, 0),
        ("complete_task",),
        ("cached", 2, 4, 0),
        ("complete_task",),
        ("exit",),
    ]
