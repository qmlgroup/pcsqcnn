from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import warnings

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
from rich.text import Text


TaskRunCallback = Callable[["ManifestTaskContext"], None]
_COMPACT_LIVE_PROGRESS_WIDTH = 100


@dataclass(frozen=True)
class ManifestTaskSpec:
    """One manifest-tracked script task with its output directory and callback."""

    name: str
    output_directory: Path
    run: TaskRunCallback
    completion_filename: str = "manifest.json"


def is_completed_output_directory(path: Path, *, completion_filename: str = "manifest.json") -> bool:
    return (path / completion_filename).is_file()


def _use_compact_live_progress(console: Console) -> bool:
    return console.is_interactive and console.width < _COMPACT_LIVE_PROGRESS_WIDTH


def _build_manifest_progress_columns(*, compact: bool) -> tuple[object, ...]:
    columns: list[object] = [TextColumn("[progress.description]{task.description}")]
    if not compact:
        columns.append(BarColumn())
    columns.extend((MofNCompleteColumn(), TaskProgressColumn()))
    return tuple(columns)


def _truncate_compact_status_line(text: str, *, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return "." * width
    return f"{text[: width - 3]}..."


def _render_status_line_text(text: str, *, console: Console) -> str:
    if not _use_compact_live_progress(console):
        return text
    return _truncate_compact_status_line(text, width=console.width)


class _ManifestTaskProgress:
    def __init__(self, *, total_tasks: int, console: Console | None = None) -> None:
        self.console = Console() if console is None else console
        self._compact_live_progress = _use_compact_live_progress(self.console)
        self.progress = Progress(
            *_build_manifest_progress_columns(compact=self._compact_live_progress),
            console=self.console,
        )
        self._status_line_text = ""
        self._status_line_visible = False
        self._live_started = False
        self._seen_user_warnings: set[tuple[type[Warning], str]] = set()
        self._warning_context = None
        self._original_showwarning = None
        self._live = Live(
            self._build_renderable(),
            console=self.console,
            refresh_per_second=10,
        )
        self.tasks_task_id = self.progress.add_task("Tasks", total=total_tasks)
        self.primary_task_id = self.progress.add_task("Details", total=1, visible=False)
        self.secondary_task_id = self.progress.add_task("Subtasks", total=1, visible=False)

    def __enter__(self) -> _ManifestTaskProgress:
        self._live_started = True
        self._live.__enter__()
        self._warning_context = warnings.catch_warnings()
        self._warning_context.__enter__()
        self._original_showwarning = warnings.showwarning
        warnings.simplefilter("always", UserWarning)
        warnings.showwarning = self._showwarning
        self._refresh_live()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        try:
            if self._warning_context is not None:
                self._warning_context.__exit__(exc_type, exc, exc_tb)
        finally:
            self._warning_context = None
            self._original_showwarning = None
            self._live.__exit__(exc_type, exc, exc_tb)
            # Rich leaves the final progress render unterminated for captured / non-interactive
            # streams, so make sure any subsequent plain print() starts on its own line.
            if not self.console.is_terminal:
                self.console.file.write("\n")
                flush = getattr(self.console.file, "flush", None)
                if callable(flush):
                    flush()
            self._live_started = False

    def emit_status(self, action: str, name: str, path: Path) -> None:
        action_style = "bold green" if action == "RUN" else "bold yellow"
        self.progress.console.print(
            Text.assemble(
                (datetime.now().strftime("%H:%M"), "cyan"),
                " ",
                (action, action_style),
                " ",
                (name, "bold"),
                " -> ",
                (str(path), "dim"),
            )
        )

    def advance_tasks(self) -> None:
        self.progress.advance(self.tasks_task_id)
        self.clear_details()

    def show_primary(self, *, description: str, total: int, completed: int = 0) -> None:
        self.progress.update(
            self.primary_task_id,
            description=description,
            total=total,
            completed=completed,
            visible=True,
        )

    def show_secondary(self, *, description: str, total: int, completed: int = 0) -> None:
        self.progress.update(
            self.secondary_task_id,
            description=description,
            total=total,
            completed=completed,
            visible=True,
        )

    def update_primary(self, *, description: str | None = None, total: int | None = None, completed: int | None = None) -> None:
        self.progress.update(
            self.primary_task_id,
            description=description,
            total=total,
            completed=completed,
            visible=True,
        )

    def update_secondary(self, *, description: str | None = None, total: int | None = None, completed: int | None = None) -> None:
        self.progress.update(
            self.secondary_task_id,
            description=description,
            total=total,
            completed=completed,
            visible=True,
        )

    def advance_secondary(self, steps: int = 1) -> None:
        self.progress.advance(self.secondary_task_id, steps)

    def hide_primary(self) -> None:
        self.progress.update(self.primary_task_id, visible=False)

    def hide_secondary(self) -> None:
        self.progress.update(self.secondary_task_id, visible=False)

    def show_status_line(self, text: str) -> None:
        self._status_line_text = text
        self._status_line_visible = True
        self._refresh_live()

    def update_status_line(self, text: str) -> None:
        self._status_line_text = text
        self._status_line_visible = True
        self._refresh_live()

    def hide_status_line(self) -> None:
        self._status_line_visible = False
        self._refresh_live()

    def clear_details(self) -> None:
        self.hide_primary()
        self.hide_secondary()
        self.hide_status_line()

    def _build_renderable(self) -> Group:
        renderables: list[object] = [self.progress]
        if self._status_line_visible:
            renderables.append(
                Text(
                    _render_status_line_text(self._status_line_text, console=self.console),
                    no_wrap=self._compact_live_progress,
                )
            )
        return Group(*renderables)

    def _refresh_live(self) -> None:
        if not self._live_started:
            return
        self._live.update(self._build_renderable(), refresh=True)

    def _showwarning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line: str | None = None,
    ) -> None:
        if isinstance(category, type) and issubclass(category, UserWarning):
            warning_key = (category, str(message))
            if warning_key in self._seen_user_warnings:
                return
            self._seen_user_warnings.add(warning_key)
            self.console.print(
                Text.assemble(
                    ("Warning", "bold red"),
                    " ",
                    str(message),
                )
            )
            return

        if self._original_showwarning is None:
            return

        self._original_showwarning(
            message,
            category,
            filename,
            lineno,
            file=file,
            line=line,
        )


class ManifestTaskContext:
    """Per-task progress facade exposed to task callbacks."""

    def __init__(self, progress: _ManifestTaskProgress) -> None:
        self._progress = progress

    def show_primary_progress(self, *, description: str, total: int, completed: int = 0) -> None:
        self._progress.show_primary(description=description, total=total, completed=completed)

    def show_secondary_progress(self, *, description: str, total: int, completed: int = 0) -> None:
        self._progress.show_secondary(description=description, total=total, completed=completed)

    def update_primary_progress(
        self,
        *,
        description: str | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        self._progress.update_primary(description=description, total=total, completed=completed)

    def update_secondary_progress(
        self,
        *,
        description: str | None = None,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        self._progress.update_secondary(description=description, total=total, completed=completed)

    def advance_secondary_progress(self, steps: int = 1) -> None:
        self._progress.advance_secondary(steps)

    def hide_secondary_progress(self) -> None:
        self._progress.hide_secondary()

    def show_status_line(self, text: str) -> None:
        self._progress.show_status_line(text)

    def update_status_line(self, text: str) -> None:
        self._progress.update_status_line(text)

    def hide_status_line(self) -> None:
        self._progress.hide_status_line()

    def clear_progress(self) -> None:
        self._progress.clear_details()


def run_manifest_tasks(
    tasks: Sequence[ManifestTaskSpec],
    *,
    rebuild: bool = False,
) -> None:
    with _ManifestTaskProgress(total_tasks=len(tasks)) as progress:
        for task in tasks:
            output_directory = task.output_directory
            completion_path = output_directory / task.completion_filename
            if output_directory.exists():
                if is_completed_output_directory(output_directory, completion_filename=task.completion_filename):
                    if not rebuild:
                        progress.emit_status("SKIP", task.name, output_directory)
                        progress.advance_tasks()
                        continue
                    shutil.rmtree(output_directory)
                else:
                    shutil.rmtree(output_directory)

            progress.emit_status("RUN", task.name, output_directory)
            context = ManifestTaskContext(progress)
            try:
                task.run(context)
            finally:
                context.clear_progress()
            if not completion_path.is_file():
                raise RuntimeError(
                    f"Task {task.name!r} finished without writing {task.completion_filename}: {output_directory}"
                )
            progress.advance_tasks()
