import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

import pytest
import torch

from .script_loading import load_run_script

plot_gradient_norms = load_run_script(
    "plot_gradient_norms_module",
    "plot_gradient_norms.py",
)


def build_fake_payload() -> dict[str, object]:
    return {
        "depths": [1, 2, 3],
        "param_seeds": [0, 1],
        "data_seed": 0,
        "post_pooling_index_qubits": 1,
        "feature_qubits": 3,
        "num_test_samples": 16,
        "class_balanced_subset": True,
        "evaluations": [
            {
                "param_seed": 0,
                "depth": 1,
                "image_size": 2,
                "layer_keys": ["multiplexers.0"],
                "layer_labels": ["Quantum 1"],
                "full_quantum_gradient_rms": torch.tensor(0.10, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.10, dtype=torch.float32),
            },
            {
                "param_seed": 1,
                "depth": 1,
                "image_size": 2,
                "layer_keys": ["multiplexers.0"],
                "layer_labels": ["Quantum 1"],
                "full_quantum_gradient_rms": torch.tensor(0.20, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.20, dtype=torch.float32),
            },
            {
                "param_seed": 0,
                "depth": 2,
                "image_size": 4,
                "layer_keys": ["multiplexers.0", "multiplexers.1"],
                "layer_labels": ["Quantum 1", "Quantum 2"],
                "full_quantum_gradient_rms": torch.tensor(0.30, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.15, dtype=torch.float32),
            },
            {
                "param_seed": 1,
                "depth": 2,
                "image_size": 4,
                "layer_keys": ["multiplexers.0", "multiplexers.1"],
                "layer_labels": ["Quantum 1", "Quantum 2"],
                "full_quantum_gradient_rms": torch.tensor(0.50, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.25, dtype=torch.float32),
            },
            {
                "param_seed": 0,
                "depth": 3,
                "image_size": 8,
                "layer_keys": ["multiplexers.0", "multiplexers.1", "multiplexers.2"],
                "layer_labels": ["Quantum 1", "Quantum 2", "Quantum 3"],
                "full_quantum_gradient_rms": torch.tensor(0.35, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.12, dtype=torch.float32),
            },
            {
                "param_seed": 1,
                "depth": 3,
                "image_size": 8,
                "layer_keys": ["multiplexers.0", "multiplexers.1", "multiplexers.2"],
                "layer_labels": ["Quantum 1", "Quantum 2", "Quantum 3"],
                "full_quantum_gradient_rms": torch.tensor(0.45, dtype=torch.float32),
                "last_quantum_layer_gradient_rms": torch.tensor(0.22, dtype=torch.float32),
            },
        ],
    }


def write_fake_payload(payload_path: Path) -> None:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_fake_payload(), payload_path)


def test_summarize_gradient_norms_payload_builds_full_and_last_layer_series(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_gradient_norms.DEFAULT_INPUT_DIRECTORY_NAME
        / plot_gradient_norms.DEFAULT_PAYLOAD_FILENAME
    )
    write_fake_payload(payload_path)

    series_list = plot_gradient_norms.summarize_gradient_norms_payload(
        payload_path,
        expected_depths=(1, 2, 3),
    )

    assert [series.label for series in series_list] == [
        "All quantum parameters",
        "Last quantum layer",
    ]
    assert series_list[0].summary.epoch == [1, 2, 3]
    assert series_list[0].summary.mean == pytest.approx([0.15, 0.4, 0.4])
    assert series_list[0].summary.lower is not None
    assert series_list[0].summary.upper is not None
    assert series_list[1].summary.mean == pytest.approx([0.15, 0.2, 0.17])


def test_plot_article_figure_s2a_uses_log_y_scale_by_default(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_gradient_norms.DEFAULT_INPUT_DIRECTORY_NAME
        / plot_gradient_norms.DEFAULT_PAYLOAD_FILENAME
    )
    write_fake_payload(payload_path)

    figure = plot_gradient_norms.plot_article_figure_s2a(payload_path=payload_path)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_yscale() == "log"
    assert figure.axes[0].get_xlabel() == "Quantum depth $Q$"
    legend = figure.axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == [
        "All quantum parameters",
        "Last quantum layer",
    ]

    plt.close(figure)


def test_plot_article_figure_s2a_supports_explicit_log_y(tmp_path: Path) -> None:
    payload_path = (
        tmp_path
        / "artifacts"
        / plot_gradient_norms.DEFAULT_INPUT_DIRECTORY_NAME
        / plot_gradient_norms.DEFAULT_PAYLOAD_FILENAME
    )
    write_fake_payload(payload_path)

    figure = plot_gradient_norms.plot_article_figure_s2a(
        payload_path=payload_path,
        log_y=True,
    )

    assert figure.axes[0].get_yscale() == "log"

    plt.close(figure)


def test_parse_args_accepts_artifacts_output_dir_and_log_y() -> None:
    args = plot_gradient_norms.parse_args(
        [
            "--artifacts-root",
            "/tmp/custom-artifacts",
            "--output-dir",
            "/tmp/custom-figs",
            "--log-y",
        ]
    )

    assert args.artifacts_root == Path("/tmp/custom-artifacts")
    assert args.output_dir == Path("/tmp/custom-figs")
    assert args.log_y is True


def test_parse_args_uses_log_y_by_default_and_supports_no_log_y() -> None:
    default_args = plot_gradient_norms.parse_args([])
    linear_args = plot_gradient_norms.parse_args(["--no-log-y"])

    assert default_args.log_y is True
    assert linear_args.log_y is False
