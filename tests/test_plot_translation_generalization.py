from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest
import torch

matplotlib.use("Agg")

from qcnn import TrainingHistory
from tests.script_loading import load_run_script


plot_generalization = load_run_script("plot_translation_generalization_module", "plot_translation_generalization.py")
evaluate_offsets = load_run_script("evaluate_translation_offsets_module", "evaluate_translation_offsets.py")


def write_fake_run(run_directory: Path, *, final_accuracies: list[float], stats_line: str) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed, accuracy in enumerate(final_accuracies):
        test_epoch = list(range(10, 101, 10))
        history = TrainingHistory(
            train_epoch=list(range(1, 101)),
            test_epoch=test_epoch,
            train_loss=[1.0] * 100,
            test_loss=[1.0] * len(test_epoch),
            train_metrics={"accuracy": [0.5] * 100},
            test_metrics={"accuracy": [accuracy * (index + 1) / len(test_epoch) for index in range(len(test_epoch))]},
        )
        torch.save(
            {"seed": seed, "training_history": history, "parameter_stats_line": stats_line},
            run_directory / f"result_seed{seed}.pt",
        )
        runs.append({"seed": seed, "result": f"result_seed{seed}.pt"})
    (run_directory / "manifest.json").write_text(json.dumps({"seeds": [0, 1], "runs": runs}), encoding="utf-8")


def test_radius_and_window_accuracy_average_the_right_offsets() -> None:
    accuracy = torch.zeros(1, 17, 17)
    accuracy[0, 6:11, 6:11] = 1.0  # offsets with max(|dy|, |dx|) <= 2

    rings = plot_generalization.radius_accuracy(accuracy)

    assert rings.shape == (1, 9)
    assert torch.equal(rings[0, :3], torch.ones(3)) and torch.equal(rings[0, 3:], torch.zeros(6))
    assert plot_generalization.window_accuracy(accuracy, inside=True).item() == 1.0
    assert plot_generalization.window_accuracy(accuracy, inside=False).item() == 0.0


def test_offset_accuracy_of_a_translation_invariant_model_is_flat() -> None:
    class BrightnessClassifier(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Total brightness is invariant to every shift that keeps the digit on the canvas.
            total = images.sum(dim=(1, 2))
            return torch.stack((total, torch.full_like(total, 2.5)), dim=1)

    images = torch.zeros(4, 32, 32)
    images[:2, 15:17, 15:17] = 1.0  # brightness 4 > 2.5 -> class 0
    images[2:, 15, 15] = 1.0  # brightness 1 < 2.5 -> class 1
    labels = torch.tensor([0, 0, 1, 1])

    accuracy = evaluate_offsets.offset_accuracy(BrightnessClassifier(), images, labels)

    assert accuracy.shape == (17, 17)
    assert torch.equal(accuracy, torch.ones(17, 17, dtype=torch.float64))


def test_main_renders_figures_and_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifacts_root = tmp_path / "artifacts"
    output_dir = tmp_path / "figs"
    stats = {"pcs": "q16384 q16384 q4096 c2570 Q36864 C2570", "hur": "q120 q16 q120 c2570 Q256 C2570"}
    for model in plot_generalization.MODELS[:3]:
        write_fake_run(
            artifacts_root / model.directory,
            final_accuracies=[0.6, 0.62],
            stats_line=stats["hur" if model.key.startswith("hur") else "pcs"],
        )
    accuracy = torch.full((2, 17, 17), 0.3, dtype=torch.float64)
    accuracy[:, 6:11, 6:11] = 0.9
    offset_path = artifacts_root / plot_generalization.OFFSET_ACCURACY_PATH
    torch.save(
        {"offsets": list(range(-8, 9)), "digits": 10, "runs": {"pcs_train2": {"directory": "x", "seeds": [0, 1], "accuracy": accuracy}}},
        offset_path,
    )

    plot_generalization.main(["--artifacts-root", str(artifacts_root), "--output-dir", str(output_dir), "--epoch-group-size", "20"])

    for filename in (
        plot_generalization.RADIUS_FILENAME,
        plot_generalization.HEATMAP_FILENAME,
        plot_generalization.CURVES_FILENAME,
        plot_generalization.ENCODINGS_FILENAME,
    ):
        assert (output_dir / filename).is_file()
    markdown = (output_dir / f"{plot_generalization.TABLE_STEM}.md").read_text(encoding="utf-8")
    assert "| PCS-QCNN, train ±2 | ±2 | 39,434 | 61.00 ± 1.41 | 90.00 ± 0.00 | 30.00 ± 0.00 |" in markdown
    assert "| Hur-QCNN x8, train ±2 | ±2 | 2,826 | 61.00 ± 1.41 | – | – |" in markdown
    latex = (output_dir / f"{plot_generalization.TABLE_STEM}.tex").read_text(encoding="utf-8")
    assert r"PCS-QCNN, train $\pm$2 & $\pm$2 & 39{,}434 & $61.00 \pm 1.41$ & $90.00 \pm 0.00$ & $30.00 \pm 0.00$ \\" in latex
    assert "Skipping PCS-QCNN, train ±8" in capsys.readouterr().err
