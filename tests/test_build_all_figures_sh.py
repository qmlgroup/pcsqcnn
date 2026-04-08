from __future__ import annotations

from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT_PATH = PROJECT_ROOT / "build_all_figures.sh"
README_PATH = PROJECT_ROOT / "README.md"


def test_build_all_figures_shell_script_has_valid_sh_syntax() -> None:
    result = subprocess.run(
        ["sh", "-n", str(BUILD_SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_build_all_figures_shell_script_lists_all_expected_commands_in_order() -> None:
    script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")
    expected_commands = [
        'run_python_step "Prepare translated-MNIST baselines data" run/prepare_translated_mnist_baselines_data.py',
        'run_python_step "Prepare full-MNIST classical baselines data" run/prepare_full_mnist_classical_baselines_data.py',
        'run_python_step "Prepare brightness sweep data" run/prepare_brightness_sweep_data.py',
        'run_python_step "Prepare PCS-QCNN architecture sweep data" run/prepare_pcsqcnn_architecture_sweep_data.py',
        'run_python_step "Prepare PCS-QCNN image-size sweep data" run/prepare_pcsqcnn_image_size_sweep_data.py',
        'run_python_step "Evaluate gradient norms" run/evaluate_gradient_norms.py',
        'run_python_step "Evaluate readout entropy" run/evaluate_readout_entropy.py',
        'run_python_step "Evaluate finite-shot accuracy snapshots" run/evaluate_finite_shot_snapshots.py',
        'run_python_step "Evaluate finite-shot loss sampling" run/evaluate_finite_shot_loss_sampling.py',
        'run_python_step "Evaluate readout landscape" run/evaluate_readout_landscape.py',
        'run_python_step "Evaluate error structure" run/evaluate_error_structure.py',
        "uv run python run/plot_classical_baseline_architectures.py",
        "uv run python run/plot_translated_mnist_baselines.py",
        "uv run python run/plot_brightness_sweep.py",
        "uv run python run/plot_pcsqcnn_architecture_sweep.py",
        "uv run python run/plot_pcsqcnn_image_size_sweep.py",
        "uv run python run/plot_gradient_norms.py",
        "uv run python run/plot_readout_entropy.py",
        "uv run python run/plot_finite_shot_accuracy.py",
        "uv run python run/plot_finite_shot_loss_histograms.py",
        "uv run python run/plot_readout_landscape.py",
        "uv run python run/plot_error_structure.py",
    ]

    positions = [script.index(command) for command in expected_commands]

    assert positions == sorted(positions)
    assert "REBUILD_FLAG=--rebuild" in script
    assert 'Usage: sh build_all_figures.sh [--rebuild]' in script
    assert '"$1" = "--rebuild"' in script
    assert "uv run python run/prepare_translated_mnist_baselines_data.py --rebuild" not in script


def test_readme_documents_build_all_figures_and_manual_s4_and_5a_steps() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "sh build_all_figures.sh" in readme
    assert "sh build_all_figures.sh --rebuild" in readme
    assert "does not force a rebuild" in readme
    assert "Passing `--rebuild` forces every prep/evaluation step" in readme
    assert "uv run python run/plot_classical_baseline_architectures.py" in readme
    assert "uv run python run/evaluate_gradient_norms.py" in readme
    assert "uv run python run/evaluate_readout_entropy.py" in readme
    assert "uv run python run/prepare_pcsqcnn_architecture_sweep_data.py" in readme
    assert "uv run python run/prepare_full_mnist_classical_baselines_data.py" in readme
    assert "uv run python run/plot_pcsqcnn_architecture_sweep.py" in readme
    assert "uv run python run/plot_gradient_norms.py" in readme
    assert "uv run python run/plot_readout_entropy.py" in readme
    assert "uv run python run/evaluate_error_structure.py" in readme
    assert "uv run python run/plot_error_structure.py" in readme
    assert "full_mnist_classical_baselines.pdf" in readme
