import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EVALUATE_LOSS_SAMPLING_PATH = Path(__file__).resolve().parents[1] / "run" / "evaluate_finite_shot_loss_sampling.py"
EVALUATE_LOSS_SAMPLING_SPEC = importlib.util.spec_from_file_location(
    "evaluate_finite_shot_loss_sampling_module",
    EVALUATE_LOSS_SAMPLING_PATH,
)
if EVALUATE_LOSS_SAMPLING_SPEC is None or EVALUATE_LOSS_SAMPLING_SPEC.loader is None:
    raise RuntimeError(f"Could not load evaluate_finite_shot_loss_sampling.py from {EVALUATE_LOSS_SAMPLING_PATH}.")
evaluate_finite_shot_loss_sampling = importlib.util.module_from_spec(EVALUATE_LOSS_SAMPLING_SPEC)
sys.modules[EVALUATE_LOSS_SAMPLING_SPEC.name] = evaluate_finite_shot_loss_sampling
EVALUATE_LOSS_SAMPLING_SPEC.loader.exec_module(evaluate_finite_shot_loss_sampling)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert evaluate_finite_shot_loss_sampling.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"


def make_reference_run_directory(tmp_path: Path, *, seeds: list[int] | None = None) -> Path:
    run_directory = tmp_path / "artifacts" / evaluate_finite_shot_loss_sampling.DEFAULT_REFERENCE_DIRECTORY_NAME
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps({"seeds": [7] if seeds is None else seeds, "runs": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_directory


def make_fake_batchwise_evaluation(
    *,
    shot_budget: int | None,
    repetitions: int,
    loss_base: float = 1.0,
) -> SimpleNamespace:
    num_draws = 1 if shot_budget is None else repetitions
    return SimpleNamespace(
        evaluations=[
            SimpleNamespace(
                shot_budget=shot_budget,
                num_draws=num_draws,
                batch_sizes=torch.tensor([2], dtype=torch.int64),
                batch_loss_sum=torch.full((num_draws, 1), fill_value=loss_base, dtype=torch.float32),
                batch_correct_count=torch.ones((num_draws, 1), dtype=torch.int64),
            )
        ]
    )


def write_cached_task_result(
    run_directory: Path,
    *,
    epoch: int,
    shot_budget: int | None,
    repetitions: int,
    loss_base: float,
) -> None:
    output_directory = evaluate_finite_shot_loss_sampling.build_task_directory(
        run_directory,
        epoch=epoch,
        shot_budget=shot_budget,
    )
    num_draws = 1 if shot_budget is None else repetitions
    evaluate_finite_shot_loss_sampling.save_task_result(
        output_directory,
        payload={
            "seed": 7,
            "epoch": epoch,
            "shot_budget": shot_budget,
            "repetitions": repetitions,
            "batch_size": evaluate_finite_shot_loss_sampling.DEFAULT_BATCH_SIZE,
            "num_draws": num_draws,
            "batch_sizes": torch.tensor([2], dtype=torch.int64),
            "batch_loss_sum": torch.full((num_draws, 1), fill_value=loss_base, dtype=torch.float32),
            "batch_correct_count": torch.ones((num_draws, 1), dtype=torch.int64),
        },
        manifest=evaluate_finite_shot_loss_sampling.build_task_manifest(
            seed=7,
            epoch=epoch,
            shot_budget=shot_budget,
            repetitions=repetitions,
            batch_size=evaluate_finite_shot_loss_sampling.DEFAULT_BATCH_SIZE,
            num_draws=num_draws,
        ),
    )


def install_tiny_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    epochs: tuple[int, ...] = (1,),
    shot_budgets: tuple[int | None, ...] = (3, None),
) -> None:
    original_evaluate = evaluate_finite_shot_loss_sampling.evaluate_finite_shot_loss_sampling

    def tiny_evaluate(*, run_directory, root, repetitions, seed, device, rebuild):
        return original_evaluate(
            run_directory=run_directory,
            root=root,
            epochs=epochs,
            shot_budgets=shot_budgets,
            repetitions=repetitions,
            seed=seed,
            device=device,
            download=False,
            rebuild=rebuild,
        )

    monkeypatch.setattr(evaluate_finite_shot_loss_sampling, "evaluate_finite_shot_loss_sampling", tiny_evaluate)


def test_main_writes_aggregate_output_and_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    call_log: list[tuple[int, int | None]] = []

    def fake_evaluate(run_directory_arg, *, epoch, shot_budgets, repetitions, **kwargs):
        del run_directory_arg, kwargs
        shot_budget = shot_budgets[0]
        call_log.append((epoch, shot_budget))
        return make_fake_batchwise_evaluation(
            shot_budget=shot_budget,
            repetitions=repetitions,
            loss_base=float(epoch if shot_budget is None else epoch + shot_budget),
        )

    monkeypatch.setattr(
        evaluate_finite_shot_loss_sampling,
        "evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test",
        fake_evaluate,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_finite_shot_loss_sampling.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--repetitions",
            "2",
        ]
    )

    output_path = run_directory / evaluate_finite_shot_loss_sampling.DEFAULT_OUTPUT_FILENAME
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    stdout_lines = capsys.readouterr().out.strip().splitlines()

    assert output_path.is_file()
    assert str(output_path) in stdout_lines
    assert call_log == [(1, 3), (1, None)]
    assert payload["seed"] == 7
    assert payload["epochs"] == [1]
    assert payload["shot_budgets"] == [3, None]
    assert payload["repetitions"] == 2
    assert len(payload["evaluations"]) == 2
    assert payload["evaluations"][1]["num_draws"] == 1

    for shot_budget in (3, None):
        task_directory = evaluate_finite_shot_loss_sampling.build_task_directory(
            run_directory,
            epoch=1,
            shot_budget=shot_budget,
        )
        assert (task_directory / "manifest.json").is_file()
        assert (task_directory / evaluate_finite_shot_loss_sampling.DEFAULT_TASK_RESULT_FILENAME).is_file()


def test_main_skips_completed_task_caches_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    write_cached_task_result(run_directory, epoch=1, shot_budget=3, repetitions=2, loss_base=4.0)
    write_cached_task_result(run_directory, epoch=1, shot_budget=None, repetitions=2, loss_base=1.0)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed Figure 7 task cache should have been reused")

    monkeypatch.setattr(
        evaluate_finite_shot_loss_sampling,
        "evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test",
        fail_if_called,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_finite_shot_loss_sampling.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--repetitions",
            "2",
        ]
    )

    payload = torch.load(
        run_directory / evaluate_finite_shot_loss_sampling.DEFAULT_OUTPUT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )

    assert len(payload["evaluations"]) == 2
    assert torch.all(payload["evaluations"][0]["batch_loss_sum"] == torch.full((2, 1), 4.0))
    assert payload["evaluations"][1]["num_draws"] == 1


def test_main_rebuilds_completed_task_caches_with_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    write_cached_task_result(run_directory, epoch=1, shot_budget=3, repetitions=2, loss_base=1.0)
    call_count = 0

    def fake_evaluate(run_directory_arg, *, epoch, shot_budgets, repetitions, **kwargs):
        del run_directory_arg, kwargs
        nonlocal call_count
        call_count += 1
        return make_fake_batchwise_evaluation(
            shot_budget=shot_budgets[0],
            repetitions=repetitions,
            loss_base=13.0,
        )

    monkeypatch.setattr(
        evaluate_finite_shot_loss_sampling,
        "evaluate_auto_training_snapshot_batchwise_loss_sampling_on_saved_mnist_test",
        fake_evaluate,
    )
    install_tiny_entrypoint(monkeypatch, shot_budgets=(3,))

    evaluate_finite_shot_loss_sampling.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--repetitions",
            "2",
            "--rebuild",
        ]
    )

    payload = torch.load(
        run_directory / evaluate_finite_shot_loss_sampling.DEFAULT_OUTPUT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    task_directory = evaluate_finite_shot_loss_sampling.build_task_directory(run_directory, epoch=1, shot_budget=3)

    assert call_count == 1
    assert torch.all(payload["evaluations"][0]["batch_loss_sum"] == torch.full((2, 1), 13.0))
    assert (task_directory / "manifest.json").is_file()
    assert (task_directory / evaluate_finite_shot_loss_sampling.DEFAULT_TASK_RESULT_FILENAME).is_file()


def test_load_single_seed_from_manifest_rejects_multi_seed_runs(tmp_path: Path) -> None:
    run_directory = make_reference_run_directory(tmp_path, seeds=[3, 5, 7])

    assert evaluate_finite_shot_loss_sampling.load_single_seed_from_manifest(run_directory) == 3
    assert evaluate_finite_shot_loss_sampling.load_single_seed_from_manifest(run_directory, seed=5) == 5
    with pytest.raises(ValueError, match="Available seeds: \\[3, 5, 7\\]"):
        evaluate_finite_shot_loss_sampling.load_single_seed_from_manifest(run_directory, seed=11)
