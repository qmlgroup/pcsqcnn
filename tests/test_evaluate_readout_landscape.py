import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EVALUATE_READOUT_LANDSCAPE_PATH = Path(__file__).resolve().parents[1] / "run" / "evaluate_readout_landscape.py"
EVALUATE_READOUT_LANDSCAPE_SPEC = importlib.util.spec_from_file_location(
    "evaluate_readout_landscape_module",
    EVALUATE_READOUT_LANDSCAPE_PATH,
)
if EVALUATE_READOUT_LANDSCAPE_SPEC is None or EVALUATE_READOUT_LANDSCAPE_SPEC.loader is None:
    raise RuntimeError(
        f"Could not load evaluate_readout_landscape.py from {EVALUATE_READOUT_LANDSCAPE_PATH}."
    )
evaluate_readout_landscape = importlib.util.module_from_spec(EVALUATE_READOUT_LANDSCAPE_SPEC)
sys.modules[EVALUATE_READOUT_LANDSCAPE_SPEC.name] = evaluate_readout_landscape
EVALUATE_READOUT_LANDSCAPE_SPEC.loader.exec_module(evaluate_readout_landscape)

def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert evaluate_readout_landscape.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"
    assert evaluate_readout_landscape.DEFAULT_REFERENCE_SEED is None


def make_reference_run_directory(tmp_path: Path, *, seeds: list[int] | None = None) -> Path:
    run_directory = tmp_path / "artifacts" / evaluate_readout_landscape.DEFAULT_REFERENCE_DIRECTORY_NAME
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps({"seeds": [7] if seeds is None else seeds, "runs": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_directory


def make_fake_landscape_evaluation(*, epoch: int, grid_points: int = 5) -> SimpleNamespace:
    axis = torch.linspace(-3.0, 3.0, grid_points)
    valid_count = torch.full((grid_points, grid_points), 2, dtype=torch.int64)
    return SimpleNamespace(
        alpha=axis,
        beta=axis.clone(),
        mean_loss=torch.full((grid_points, grid_points), 0.1 * epoch, dtype=torch.float32),
        valid_count=valid_count,
        valid_fraction=valid_count.to(dtype=torch.float32) / 20.0,
        total_samples=20,
        eligible_sample_count=18,
    )


def write_valid_cached_task_result(
    run_directory: Path,
    *,
    epoch: int,
    alpha_beta_points: int,
    sample_batch_size: int,
    grid_chunk_size: int,
) -> None:
    output_directory = evaluate_readout_landscape.build_task_directory(run_directory, epoch=epoch)
    evaluation = make_fake_landscape_evaluation(epoch=epoch, grid_points=alpha_beta_points)
    evaluate_readout_landscape.save_task_result(
        output_directory,
        payload={
            "epoch": epoch,
            "pc1_sigma": evaluation.alpha,
            "pc2_sigma": evaluation.beta,
            "mean_loss": evaluation.mean_loss,
            "valid_count": evaluation.valid_count,
            "valid_fraction": evaluation.valid_fraction,
            "total_samples": evaluation.total_samples,
            "eligible_sample_count": evaluation.eligible_sample_count,
        },
        manifest=evaluate_readout_landscape.build_task_manifest(
            seed=7,
            epoch=epoch,
            alpha_beta_points=alpha_beta_points,
            axis_limit=evaluate_readout_landscape.DEFAULT_AXIS_LIMIT,
            sample_batch_size=sample_batch_size,
            grid_chunk_size=grid_chunk_size,
            sigma_shot_budget=evaluate_readout_landscape.DEFAULT_SIGMA_SHOT_BUDGET,
            basis_mode=evaluate_readout_landscape.DEFAULT_BASIS_MODE,
            total_samples=evaluation.total_samples,
        ),
    )


def install_tiny_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    epochs: tuple[int, ...] = (1,),
    alpha_beta_points: int = 5,
) -> None:
    original_evaluate = evaluate_readout_landscape.evaluate_readout_landscape
    tiny_epochs = epochs
    tiny_alpha_beta_points = alpha_beta_points

    def tiny_evaluate(
        *,
        run_directory,
        root,
        epochs,
        seed,
        device,
        alpha_beta_points,
        axis_limit,
        sample_batch_size,
        grid_chunk_size,
        sigma_shot_budget,
        rebuild,
    ):
        assert tuple(epochs) == evaluate_readout_landscape.DEFAULT_REFERENCE_EPOCHS
        assert alpha_beta_points == evaluate_readout_landscape.DEFAULT_ALPHA_BETA_POINTS
        assert axis_limit == pytest.approx(evaluate_readout_landscape.DEFAULT_AXIS_LIMIT)
        assert sigma_shot_budget == evaluate_readout_landscape.DEFAULT_SIGMA_SHOT_BUDGET
        return original_evaluate(
            run_directory=run_directory,
            root=root,
            epochs=tiny_epochs,
            seed=seed,
            device=device,
            alpha_beta_points=tiny_alpha_beta_points,
            axis_limit=axis_limit,
            sample_batch_size=sample_batch_size,
            grid_chunk_size=grid_chunk_size,
            sigma_shot_budget=sigma_shot_budget,
            download=False,
            rebuild=rebuild,
        )

    monkeypatch.setattr(evaluate_readout_landscape, "evaluate_readout_landscape", tiny_evaluate)


def test_main_writes_aggregate_output_and_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    call_log: list[int] = []

    def fake_evaluate(*args, epoch: int, axis_limit: float, **kwargs):
        call_log.append(epoch)
        assert axis_limit == pytest.approx(evaluate_readout_landscape.DEFAULT_AXIS_LIMIT)
        return make_fake_landscape_evaluation(epoch=epoch)

    monkeypatch.setattr(
        evaluate_readout_landscape,
        "evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test",
        fake_evaluate,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_readout_landscape.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--sample-batch-size",
            "4",
            "--grid-chunk-size",
            "3",
        ]
    )

    output_path = run_directory / evaluate_readout_landscape.DEFAULT_OUTPUT_FILENAME
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    stdout_lines = capsys.readouterr().out.strip().splitlines()
    task_directory = evaluate_readout_landscape.build_task_directory(run_directory, epoch=1)

    assert output_path.is_file()
    assert str(output_path) in stdout_lines
    assert call_log == [1]
    assert payload["seed"] == 7
    assert payload["epochs"] == [1]
    assert payload["sigma_shot_budget"] == evaluate_readout_landscape.DEFAULT_SIGMA_SHOT_BUDGET
    assert payload["basis_mode"] == evaluate_readout_landscape.DEFAULT_BASIS_MODE
    assert tuple(payload["pc1_sigma"].shape) == (5,)
    assert tuple(payload["pc2_sigma"].shape) == (5,)
    assert len(payload["evaluations"]) == 1
    assert tuple(payload["evaluations"][0]["valid_fraction"].shape) == (5, 5)
    assert payload["evaluations"][0]["eligible_sample_count"] == 18
    assert (task_directory / "manifest.json").is_file()
    assert (task_directory / evaluate_readout_landscape.DEFAULT_TASK_RESULT_FILENAME).is_file()


def test_main_threads_scientific_defaults_and_seed_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = make_reference_run_directory(tmp_path, seeds=[3, 5, 7])
    captured_calls: list[dict[str, object]] = []

    def fake_evaluate_readout_landscape(
        *,
        run_directory,
        root,
        epochs,
        seed,
        device,
        alpha_beta_points,
        axis_limit,
        sample_batch_size,
        grid_chunk_size,
        sigma_shot_budget,
        rebuild,
    ):
        del root, device, sample_batch_size, grid_chunk_size, rebuild
        captured_calls.append(
            {
                "run_directory": Path(run_directory),
                "epochs": tuple(epochs),
                "seed": seed,
                "alpha_beta_points": alpha_beta_points,
                "axis_limit": axis_limit,
                "sigma_shot_budget": sigma_shot_budget,
            }
        )
        return {
            "seed": seed,
            "epochs": list(epochs),
            "pc1_sigma": torch.linspace(-1.0, 1.0, 3),
            "pc2_sigma": torch.linspace(-1.0, 1.0, 3),
            "sigma_shot_budget": sigma_shot_budget,
            "basis_mode": evaluate_readout_landscape.DEFAULT_BASIS_MODE,
            "total_samples": 1,
            "evaluations": [],
        }

    monkeypatch.setattr(
        evaluate_readout_landscape,
        "evaluate_readout_landscape",
        fake_evaluate_readout_landscape,
    )

    evaluate_readout_landscape.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
        ]
    )
    evaluate_readout_landscape.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--seed",
            "5",
        ]
    )

    assert len(captured_calls) == 2
    assert captured_calls[0] == {
        "run_directory": run_directory,
        "epochs": evaluate_readout_landscape.DEFAULT_REFERENCE_EPOCHS,
        "seed": evaluate_readout_landscape.DEFAULT_REFERENCE_SEED,
        "alpha_beta_points": evaluate_readout_landscape.DEFAULT_ALPHA_BETA_POINTS,
        "axis_limit": evaluate_readout_landscape.DEFAULT_AXIS_LIMIT,
        "sigma_shot_budget": evaluate_readout_landscape.DEFAULT_SIGMA_SHOT_BUDGET,
    }
    assert captured_calls[1] == {
        "run_directory": run_directory,
        "epochs": evaluate_readout_landscape.DEFAULT_REFERENCE_EPOCHS,
        "seed": 5,
        "alpha_beta_points": evaluate_readout_landscape.DEFAULT_ALPHA_BETA_POINTS,
        "axis_limit": evaluate_readout_landscape.DEFAULT_AXIS_LIMIT,
        "sigma_shot_budget": evaluate_readout_landscape.DEFAULT_SIGMA_SHOT_BUDGET,
    }


def test_main_skips_completed_task_caches_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    write_valid_cached_task_result(
        run_directory,
        epoch=1,
        alpha_beta_points=5,
        sample_batch_size=4,
        grid_chunk_size=3,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed Figure S3 task cache should have been reused")

    monkeypatch.setattr(
        evaluate_readout_landscape,
        "evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test",
        fail_if_called,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_readout_landscape.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--sample-batch-size",
            "4",
            "--grid-chunk-size",
            "3",
        ]
    )

    payload = torch.load(
        run_directory / evaluate_readout_landscape.DEFAULT_OUTPUT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )

    assert len(payload["evaluations"]) == 1
    assert int(payload["evaluations"][0]["valid_count"].sum().item()) > 0


def test_main_rebuilds_incompatible_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    task_directory = evaluate_readout_landscape.build_task_directory(run_directory, epoch=1)
    task_directory.mkdir(parents=True, exist_ok=True)
    (task_directory / "manifest.json").write_text(
        json.dumps(
            {
                "task_type": "readout_landscape",
                "format_version": 1,
                "seed": 7,
                "epoch": 1,
                "alpha_beta_points": 5,
                "axis_limit": 3.0,
                "sample_batch_size": 4,
                "grid_chunk_size": 3,
                "sigma_shot_budget": 1024,
                "basis_mode": evaluate_readout_landscape.DEFAULT_BASIS_MODE,
                "total_samples": 20,
                "result_filename": evaluate_readout_landscape.DEFAULT_TASK_RESULT_FILENAME,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "epoch": 1,
            "pc1_sigma": torch.linspace(-3.0, 3.0, 5),
            "pc2_sigma": torch.linspace(-3.0, 3.0, 5),
            "direction_u": torch.zeros(8),
            "direction_v": torch.zeros(8),
            "pc1_std": 0.2,
            "pc2_std": 0.1,
            "mean_loss": torch.zeros((5, 5)),
            "valid_count": torch.zeros((5, 5), dtype=torch.int64),
            "total_samples": 20,
        },
        task_directory / evaluate_readout_landscape.DEFAULT_TASK_RESULT_FILENAME,
    )
    call_log: list[int] = []

    def fake_evaluate(*args, epoch: int, **kwargs):
        call_log.append(epoch)
        return make_fake_landscape_evaluation(epoch=epoch)

    monkeypatch.setattr(
        evaluate_readout_landscape,
        "evaluate_auto_training_snapshot_readout_landscape_on_saved_mnist_test",
        fake_evaluate,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_readout_landscape.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--sample-batch-size",
            "4",
            "--grid-chunk-size",
            "3",
        ]
    )

    payload = torch.load(
        run_directory / evaluate_readout_landscape.DEFAULT_OUTPUT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    manifest = json.loads((task_directory / "manifest.json").read_text(encoding="utf-8"))

    assert call_log == [1]
    assert manifest["format_version"] == evaluate_readout_landscape.DEFAULT_TASK_FORMAT_VERSION
    assert int(payload["evaluations"][0]["valid_count"].sum().item()) > 0


def test_load_single_seed_from_manifest_defaults_to_first_seed_and_accepts_override(tmp_path: Path) -> None:
    run_directory = make_reference_run_directory(tmp_path, seeds=[3, 5, 7])

    assert evaluate_readout_landscape.load_single_seed_from_manifest(run_directory) == 3
    assert evaluate_readout_landscape.load_single_seed_from_manifest(run_directory, seed=5) == 5
    with pytest.raises(ValueError, match="Available seeds: \\[3, 5, 7\\]"):
        evaluate_readout_landscape.load_single_seed_from_manifest(run_directory, seed=11)
