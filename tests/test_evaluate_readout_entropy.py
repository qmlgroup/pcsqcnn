import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .script_loading import load_run_script

evaluate_readout_entropy = load_run_script(
    "evaluate_readout_entropy_module",
    "evaluate_readout_entropy.py",
)


def make_reference_run_directory(tmp_path: Path, *, seeds: list[int] | None = None) -> Path:
    run_directory = tmp_path / "artifacts" / evaluate_readout_entropy.DEFAULT_REFERENCE_DIRECTORY_NAME
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps({"seeds": [7] if seeds is None else seeds, "runs": []}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_directory


def make_fake_entropy_evaluation(*, shot_budget: int) -> SimpleNamespace:
    return SimpleNamespace(
        shot_budget=shot_budget,
        entropy=torch.tensor([0.1 * shot_budget, 0.2 * shot_budget], dtype=torch.float32),
    )


def install_tiny_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    epochs: tuple[int, ...] = (100,),
    shot_budgets: tuple[int, ...] = (128, 256),
) -> None:
    original_evaluate = evaluate_readout_entropy.evaluate_readout_entropy

    def tiny_evaluate(*, run_directory, root, batch_size, seed, device, rebuild):
        return original_evaluate(
            run_directory=run_directory,
            root=root,
            epochs=epochs,
            shot_budgets=shot_budgets,
            batch_size=batch_size,
            seed=seed,
            device=device,
            download=False,
            rebuild=rebuild,
        )

    monkeypatch.setattr(evaluate_readout_entropy, "evaluate_readout_entropy", tiny_evaluate)


def test_default_reference_directory_points_to_canonical_16on16_run() -> None:
    assert evaluate_readout_entropy.DEFAULT_REFERENCE_DIRECTORY_NAME == "pcsqcnn_image_size_sweep/16on16"


def test_parse_args_accepts_device_batch_size_and_rebuild() -> None:
    args = evaluate_readout_entropy.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--data-root",
            "/tmp/custom-data",
            "--device",
            "cpu",
            "--batch-size",
            "32",
            "--seed",
            "11",
            "--rebuild",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.data_root == Path("/tmp/custom-data")
    assert args.device == "cpu"
    assert args.batch_size == 32
    assert args.seed == 11
    assert args.rebuild is True


def test_main_writes_aggregate_output_and_task_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    run_directory = make_reference_run_directory(tmp_path)
    call_log: list[tuple[int, int]] = []

    def fake_evaluate(run_directory_arg, *, epoch, shot_budget, **kwargs):
        del run_directory_arg, kwargs
        call_log.append((epoch, shot_budget))
        return make_fake_entropy_evaluation(shot_budget=shot_budget)

    monkeypatch.setattr(
        evaluate_readout_entropy,
        "evaluate_auto_training_snapshot_readout_entropy_on_saved_mnist_test",
        fake_evaluate,
    )
    install_tiny_entrypoint(monkeypatch)

    evaluate_readout_entropy.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
            "--batch-size",
            "16",
        ]
    )

    output_path = run_directory / evaluate_readout_entropy.DEFAULT_OUTPUT_FILENAME
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    stdout_lines = capsys.readouterr().out.strip().splitlines()

    assert output_path.is_file()
    assert str(output_path) in stdout_lines
    assert call_log == [(100, 128), (100, 256)]
    assert payload["seed"] == 7
    assert payload["epochs"] == [100]
    assert payload["shot_budgets"] == [128, 256]
    assert len(payload["evaluations"]) == 2

    for shot_budget in (128, 256):
        task_directory = evaluate_readout_entropy.build_task_directory(
            run_directory,
            epoch=100,
            shot_budget=shot_budget,
        )
        assert (task_directory / "manifest.json").is_file()
        assert (task_directory / evaluate_readout_entropy.DEFAULT_TASK_RESULT_FILENAME).is_file()


def test_load_single_seed_from_manifest_defaults_to_first_seed_and_accepts_override(tmp_path: Path) -> None:
    run_directory = make_reference_run_directory(tmp_path, seeds=[3, 5, 7])

    assert evaluate_readout_entropy.load_single_seed_from_manifest(run_directory) == 3
    assert evaluate_readout_entropy.load_single_seed_from_manifest(run_directory, seed=5) == 5
    with pytest.raises(ValueError, match="Available seeds: \\[3, 5, 7\\]"):
        evaluate_readout_entropy.load_single_seed_from_manifest(run_directory, seed=11)
