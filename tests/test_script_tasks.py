import io
import warnings
from pathlib import Path

import pytest
from rich.console import Console
from rich.progress import BarColumn

from qcnn.script_tasks import (
    _build_manifest_progress_columns,
    _render_status_line_text,
    _use_compact_live_progress,
    ManifestTaskContext,
    ManifestTaskSpec,
    is_completed_output_directory,
    run_manifest_tasks,
)


class FakeProgress:
    def __init__(self) -> None:
        self._status_line_text = ""
        self._status_line_visible = False
        self.primary_visible = False
        self.secondary_visible = False

    def show_primary(self, *, description: str, total: int, completed: int = 0) -> None:
        del description, total, completed
        self.primary_visible = True

    def show_secondary(self, *, description: str, total: int, completed: int = 0) -> None:
        del description, total, completed
        self.secondary_visible = True

    def update_primary(
        self,
        *,
        description: str | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        del description, total, completed

    def update_secondary(
        self,
        *,
        description: str | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        del description, total, completed

    def advance_secondary(self, steps: int = 1) -> None:
        del steps

    def hide_primary(self) -> None:
        self.primary_visible = False

    def hide_secondary(self) -> None:
        self.secondary_visible = False

    def show_status_line(self, text: str) -> None:
        self._status_line_text = text
        self._status_line_visible = True

    def update_status_line(self, text: str) -> None:
        self._status_line_text = text
        self._status_line_visible = True

    def hide_status_line(self) -> None:
        self._status_line_visible = False

    def clear_details(self) -> None:
        self.hide_primary()
        self.hide_secondary()
        self.hide_status_line()


def test_is_completed_output_directory_requires_completion_file(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"
    incomplete_directory = tmp_path / "incomplete"
    complete_directory = tmp_path / "complete"

    incomplete_directory.mkdir()
    complete_directory.mkdir()
    (complete_directory / "manifest.json").write_text("{}", encoding="utf-8")

    assert is_completed_output_directory(missing_directory) is False
    assert is_completed_output_directory(incomplete_directory) is False
    assert is_completed_output_directory(complete_directory) is True


def test_run_manifest_tasks_skips_completed_outputs_without_rebuild(tmp_path: Path) -> None:
    output_directory = tmp_path / "completed"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "manifest.json").write_text("{}", encoding="utf-8")
    called = False

    def run(task_context: ManifestTaskContext) -> None:
        del task_context
        nonlocal called
        called = True

    run_manifest_tasks((ManifestTaskSpec(name="completed", output_directory=output_directory, run=run),))

    assert called is False


def test_run_manifest_tasks_rebuilds_completed_outputs_with_rebuild(tmp_path: Path) -> None:
    output_directory = tmp_path / "completed"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "manifest.json").write_text("{}", encoding="utf-8")
    (output_directory / "stale.txt").write_text("old", encoding="utf-8")
    stale_seen_during_run: list[bool] = []

    def run(task_context: ManifestTaskContext) -> None:
        del task_context
        stale_seen_during_run.append((output_directory / "stale.txt").exists())
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "manifest.json").write_text("{}", encoding="utf-8")

    run_manifest_tasks(
        (ManifestTaskSpec(name="completed", output_directory=output_directory, run=run),),
        rebuild=True,
    )

    assert stale_seen_during_run == [False]


def test_run_manifest_tasks_removes_incomplete_nested_outputs_before_rerun(tmp_path: Path) -> None:
    output_directory = tmp_path / "nested" / "epoch010_shots128"
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "stale.txt").write_text("incomplete", encoding="utf-8")
    stale_seen_during_run: list[bool] = []

    def run(task_context: ManifestTaskContext) -> None:
        task_context.show_primary_progress(description="Batches", total=2, completed=0)
        task_context.update_primary_progress(completed=1)
        stale_seen_during_run.append((output_directory / "stale.txt").exists())
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "manifest.json").write_text("{}", encoding="utf-8")

    run_manifest_tasks((ManifestTaskSpec(name="nested", output_directory=output_directory, run=run),))

    assert stale_seen_during_run == [False]
    assert (output_directory / "manifest.json").is_file()


def test_manifest_task_context_forwards_status_line_methods() -> None:
    progress = FakeProgress()
    context = ManifestTaskContext(progress)

    context.show_status_line("Loss -- -- Accuracy -- --")
    assert progress._status_line_visible is True
    assert progress._status_line_text == "Loss -- -- Accuracy -- --"

    context.update_status_line("Loss 0.1000 -- Accuracy 75.0% --")
    assert progress._status_line_text == "Loss 0.1000 -- Accuracy 75.0% --"

    context.hide_status_line()
    assert progress._status_line_visible is False


def test_manifest_task_context_clear_progress_hides_status_line() -> None:
    progress = FakeProgress()
    context = ManifestTaskContext(progress)

    context.show_primary_progress(description="Seeds", total=2, completed=0)
    context.show_secondary_progress(description="Epochs", total=3, completed=0)
    context.show_status_line("Loss -- -- Accuracy -- --")

    context.clear_progress()

    assert progress._status_line_visible is False


def test_manifest_task_context_does_not_expose_removed_primary_progress_methods() -> None:
    context = ManifestTaskContext(FakeProgress())

    assert not hasattr(context, "advance_primary_progress")
    assert not hasattr(context, "hide_primary_progress")


def make_interactive_console(*, width: int) -> Console:
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        force_interactive=True,
        width=width,
        _environ={"TERM": "screen-256color"},
    )


def test_use_compact_live_progress_for_narrow_interactive_console() -> None:
    console = make_interactive_console(width=72)

    assert _use_compact_live_progress(console) is True
    assert not any(isinstance(column, BarColumn) for column in _build_manifest_progress_columns(compact=True))


def test_use_full_live_progress_for_wide_interactive_console() -> None:
    console = make_interactive_console(width=140)

    assert _use_compact_live_progress(console) is False
    assert any(isinstance(column, BarColumn) for column in _build_manifest_progress_columns(compact=False))


def test_render_status_line_text_truncates_in_compact_mode() -> None:
    console = make_interactive_console(width=20)
    status_line = "Loss 0.1234 0.2345 Accuracy 98.7% 97.5%"

    assert _render_status_line_text(status_line, console=console) == "Loss 0.1234 0.234..."


def test_render_status_line_text_preserves_full_mode_text() -> None:
    console = make_interactive_console(width=140)
    status_line = "Loss 0.1234 0.2345 Accuracy 98.7% 97.5%"

    assert _render_status_line_text(status_line, console=console) == status_line


def test_run_manifest_tasks_formats_and_deduplicates_user_warnings(
    tmp_path: Path,
    capsys,
) -> None:
    output_directory = tmp_path / "warn"

    def run(task_context: ManifestTaskContext) -> None:
        del task_context
        warnings.warn("first warning", UserWarning)
        warnings.warn("first warning", UserWarning)
        warnings.warn("second warning", UserWarning)
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "manifest.json").write_text("{}", encoding="utf-8")

    run_manifest_tasks((ManifestTaskSpec(name="warn", output_directory=output_directory, run=run),))

    captured = capsys.readouterr()
    assert captured.out.count("Warning first warning") == 1
    assert captured.out.count("Warning second warning") == 1
    assert "UserWarning" not in captured.out
    assert "test_script_tasks.py" not in captured.out


def test_run_manifest_tasks_forwards_non_user_warning_to_default_handler(
    tmp_path: Path,
    capsys,
) -> None:
    class DifferentWarning(Warning):
        pass

    output_directory = tmp_path / "otherwarn"

    def run(task_context: ManifestTaskContext) -> None:
        del task_context
        warnings.warn("other warning", DifferentWarning)
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.warns(DifferentWarning, match="other warning"):
        run_manifest_tasks((ManifestTaskSpec(name="otherwarn", output_directory=output_directory, run=run),))

    captured = capsys.readouterr()
    assert "Warning other warning" not in captured.out


def test_run_manifest_tasks_leaves_stdout_ready_for_followup_print(
    tmp_path: Path,
    capsys,
) -> None:
    output_directory = tmp_path / "followup"

    def run(task_context: ManifestTaskContext) -> None:
        del task_context
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "manifest.json").write_text("{}", encoding="utf-8")

    run_manifest_tasks((ManifestTaskSpec(name="followup", output_directory=output_directory, run=run),))
    print("FOLLOWUP /tmp/result.pt")

    captured = capsys.readouterr()
    assert "FOLLOWUP /tmp/result.pt" in captured.out.splitlines()
