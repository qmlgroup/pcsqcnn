from __future__ import annotations

import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT_PATH = PROJECT_ROOT / "build_all_figures.sh"


def trace_commands(tmp_path, *options):
    executable = tmp_path / "uv"
    executable.write_text("""#!/bin/sh
printf '%s\\n' "$*" >> "$COMMAND_TRACE"
""")
    executable.chmod(0o755)
    trace = tmp_path / "commands.txt"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "COMMAND_TRACE": str(trace)}
    subprocess.run(["sh", str(BUILD_SCRIPT_PATH), *options], env=env,
                   check=True, capture_output=True, text=True)
    return trace.read_text().splitlines()


def test_build_all_figures_shell_script_has_valid_sh_syntax():
    subprocess.run(["sh", "-n", str(BUILD_SCRIPT_PATH)], check=True)


def is_render(command):
    return "python run/plot_" in command or "python run/render_paper_diagrams.py" in command


def test_plots_only_does_not_train_or_evaluate(tmp_path):
    commands = trace_commands(tmp_path, "--plots-only")
    assert commands
    assert all(is_render(command) for command in commands)
    assert any("plot_frqi_comparison.py" in c for c in commands)
    assert any("plot_pcs_translation.py" in c for c in commands)
    assert any("plot_mfa_pcs_comparison.py" in c for c in commands)


def test_training_selects_the_published_translation_protocol(tmp_path):
    commands = trace_commands(tmp_path)
    assert not any("train2" in c or "henderson" in c or
                   "translation_generalization" in c for c in commands)
    encoding, = [c for c in commands if "prepare_encoding_swap_data.py" in c]
    assert "--runs hur_su4_l1_frqi_train8 hur_su4_l1_frqi_train8_e6000_warm" in encoding
    sp, = [c for c in commands if "prepare_sp_qcnn" in c]
    assert "--runs sp_qcnn_train8 --seed-count 2" in sp
    first_plot = next(i for i, c in enumerate(commands) if "python run/plot_" in c)
    assert all(is_render(c) for c in commands[first_plot:])


def test_rebuild_applies_only_to_preparation_and_evaluation(tmp_path):
    commands = trace_commands(tmp_path, "--rebuild")
    for command in commands:
        assert ("--rebuild" in command) == (not is_render(command))


def test_verify_reconstructs_before_rendering_without_training(tmp_path):
    commands = trace_commands(tmp_path, "--verify")
    assert commands[0] == "run python run/verify_pcs_translation.py"
    assert all(is_render(command) for command in commands[1:])
    assert any("render_paper_diagrams.py" in command for command in commands)
    assert not any("--rebuild" in command for command in commands)
