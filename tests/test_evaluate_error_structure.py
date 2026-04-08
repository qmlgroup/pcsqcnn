import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qcnn import ErrorAnalysisPayload

from .script_loading import load_run_script

evaluate_error_structure = load_run_script(
    "evaluate_error_structure_module",
    "evaluate_error_structure.py",
)


def make_run_directory(
    tmp_path: Path,
    *,
    scaled_image_size: int = 16,
    image_size: int = 16,
    seeds: list[int] | None = None,
) -> Path:
    run_directory = (
        tmp_path
        / "artifacts"
        / evaluate_error_structure.DEFAULT_SWEEP_DIRECTORY_NAME
        / f"{scaled_image_size}on{image_size}"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps(
            {"seeds": [7] if seeds is None else seeds, "runs": []},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run_directory


def make_error_payload() -> dict[str, object]:
    return {
        "seed": 7,
        "image_size": 16,
        "class_labels": [str(index) for index in range(3)],
        "confusion_matrix": torch.tensor(
            [
                [3, 1, 0],
                [0, 2, 1],
                [0, 1, 2],
            ],
            dtype=torch.long,
        ),
        "misclassified_images": torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4),
        "misclassified_true_labels": torch.tensor([1, 2], dtype=torch.long),
        "misclassified_predicted_labels": torch.tensor([2, 1], dtype=torch.long),
    }


def test_parse_args_defaults_to_project_paths() -> None:
    args = evaluate_error_structure.parse_args([])

    assert args.artifacts_root == evaluate_error_structure.DEFAULT_ARTIFACTS_ROOT
    assert args.data_root == evaluate_error_structure.DEFAULT_DATA_ROOT
    assert args.image_size == evaluate_error_structure.DEFAULT_IMAGE_SIZE
    assert args.scaled_image_size == evaluate_error_structure.DEFAULT_SCALED_IMAGE_SIZE
    assert args.seed is None
    assert args.device is None
    assert args.rebuild is False


def test_parse_args_accepts_overrides() -> None:
    args = evaluate_error_structure.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--data-root",
            "/tmp/custom-data",
            "--image-size",
            "32",
            "--scaled-image-size",
            "28",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--rebuild",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.data_root == Path("/tmp/custom-data")
    assert args.image_size == 32
    assert args.scaled_image_size == 28
    assert args.seed == 11
    assert args.device == "cpu"
    assert args.rebuild is True


def test_build_figure_5b_run_directory_nests_under_image_size_sweep(tmp_path: Path) -> None:
    run_directory = evaluate_error_structure.build_figure_5b_run_directory(
        tmp_path / "artifacts",
        scaled_image_size=28,
        image_size=32,
    )

    assert run_directory == tmp_path / "artifacts" / "pcsqcnn_image_size_sweep" / "28on32"


def test_resolve_error_structure_seed_accepts_singleton_and_explicit_seed(tmp_path: Path) -> None:
    singleton_run = make_run_directory(tmp_path, seeds=[7])
    multi_seed_run = make_run_directory(tmp_path / "other", seeds=[3, 5, 7])

    assert evaluate_error_structure.resolve_error_structure_seed(singleton_run, seed=None) == 7
    assert evaluate_error_structure.resolve_error_structure_seed(multi_seed_run, seed=None) == 3
    assert evaluate_error_structure.resolve_error_structure_seed(multi_seed_run, seed=5) == 5


def test_resolve_error_structure_seed_rejects_missing_seed(tmp_path: Path) -> None:
    multi_seed_run = make_run_directory(tmp_path, seeds=[3, 5, 7])

    with pytest.raises(ValueError, match="Available seeds: \\[3, 5, 7\\]"):
        evaluate_error_structure.resolve_error_structure_seed(multi_seed_run, seed=11)


def test_extract_error_structure_builds_expected_saved_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object()
    test_loader = object()
    fake_loaded_run = SimpleNamespace(
        saved_mnist_test_config=lambda: {"image_size": 16},
    )
    error_payload = ErrorAnalysisPayload(
        class_labels=("0", "1", "2"),
        confusion_matrix=torch.tensor([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=torch.long),
        misclassified_images=torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4),
        misclassified_true_labels=torch.tensor([1, 2], dtype=torch.long),
        misclassified_predicted_labels=torch.tensor([2, 1], dtype=torch.long),
    )

    monkeypatch.setattr(evaluate_error_structure, "load_auto_training_run", lambda *args, **kwargs: fake_loaded_run)
    monkeypatch.setattr(
        evaluate_error_structure,
        "reconstruct_saved_mnist_splits_from_run",
        lambda *args, **kwargs: "splits",
    )
    monkeypatch.setattr(
        evaluate_error_structure,
        "reconstruct_run_runner_and_test_loader",
        lambda *args, **kwargs: SimpleNamespace(runner=runner, test_loader=test_loader),
    )
    monkeypatch.setattr(
        evaluate_error_structure,
        "collect_error_analysis_payload",
        lambda received_runner, received_loader, max_examples: (
            error_payload
            if received_runner is runner and received_loader is test_loader and max_examples is None
            else None
        ),
    )

    payload = evaluate_error_structure.extract_error_structure(
        run_directory="/tmp/fake-run",
        root="/tmp/fake-data",
        seed=7,
        device="cpu",
        download=False,
    )

    assert payload["seed"] == 7
    assert payload["image_size"] == 16
    assert payload["class_labels"] == ["0", "1", "2"]
    assert torch.equal(payload["confusion_matrix"], error_payload.confusion_matrix)
    assert torch.equal(payload["misclassified_images"], error_payload.misclassified_images)
    assert torch.equal(payload["misclassified_true_labels"], error_payload.misclassified_true_labels)
    assert torch.equal(payload["misclassified_predicted_labels"], error_payload.misclassified_predicted_labels)


def test_main_reuses_existing_payload_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    run_directory = make_run_directory(tmp_path, seeds=[7])
    output_path = evaluate_error_structure.build_error_structure_output_path(run_directory, seed=7)
    torch.save(make_error_payload(), output_path)

    monkeypatch.setattr(
        evaluate_error_structure,
        "extract_error_structure",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("existing payload should have been reused")),
    )

    evaluate_error_structure.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert capsys.readouterr().out.strip() == str(output_path)


def test_main_writes_payload_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    run_directory = make_run_directory(tmp_path, seeds=[7])
    fake_payload = make_error_payload()
    call_log: list[dict[str, object]] = []

    def fake_extract(**kwargs):
        call_log.append(kwargs)
        return fake_payload

    monkeypatch.setattr(evaluate_error_structure, "extract_error_structure", fake_extract)

    evaluate_error_structure.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--device",
            "cpu",
        ]
    )

    output_path = evaluate_error_structure.build_error_structure_output_path(run_directory, seed=7)
    saved_payload = torch.load(output_path, map_location="cpu", weights_only=False)

    assert capsys.readouterr().out.strip() == str(output_path)
    assert len(call_log) == 1
    assert call_log[0]["run_directory"] == run_directory
    assert call_log[0]["root"] == (tmp_path / "data")
    assert call_log[0]["seed"] == 7
    assert call_log[0]["device"] == "cpu"
    assert saved_payload["seed"] == 7
    assert saved_payload["image_size"] == 16
