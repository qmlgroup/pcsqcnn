"""Export MNIST/FRQI figure inputs and comparator-table evidence as compact JSON artifacts.

The training and evaluation scripts write complete runs (results, checkpoints,
snapshots) with ``torch.save``. The plot scripts read only part of that: the
``manifest.json`` and per-seed results of the runs they show, and the
evaluation payloads. This script finds exactly those files and writes them to
a separate folder:

- each selected paper plot script that takes ``--artifacts-root`` is run once against the
  full artifacts folder (``--source``) with its file accesses recorded; the
  figures it renders go to a temporary folder;
- the recorded files are written below ``--destination`` with the same
  relative paths, portable data/artifact path metadata, and every
  ``torch.save`` file as its JSON export (``qcnn.json_artifacts``) with the
  suffix ``.json``; no checkpoint is needed by any plot, so none is exported;
- table-only comparator runs (replaceable with ``--extra-runs``) are exported the same way: their ``manifest.json`` and
  per-seed results, which hold the complete training histories, and the
  files in ``EXTRA_FILES`` (the QCS validation-search summary) are exported
  with the same path normalization;
- ``INDEX.json`` lists the exported files of every plot script and of every
  extra run.

Lines in which a plot script reports skipped inputs (``Warning:`` or
``Skipping``) are printed for every render, because a figure that is skipped
in both folders would otherwise pass the comparison unnoticed.

The plot scripts read the JSON exports in place of the original files, so
``--artifacts-root DESTINATION`` renders the same figures. ``--check``
verifies this by rendering every plot from both folders and comparing the
output files byte by byte (with ``SOURCE_DATE_EPOCH`` fixed, so PDF dates
agree).

Usage::

    python run/export_plot_artifacts.py --source FULL_ARTIFACTS --destination artifacts --check
"""

from __future__ import annotations

import argparse
import builtins
from collections.abc import Sequence
import filecmp
import io
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "run"
INDEX_FILENAME = "INDEX.json"
RENDER_ENVIRONMENT = {"SOURCE_DATE_EPOCH": "0", "MPLBACKEND": "Agg"}
# Fixed selection of main MNIST and common FRQI plots.
PUBLISHED_PLOT_SCRIPTS = ('plot_main_panels.py', 'plot_translated_mnist_baselines.py', 'plot_brightness_sweep.py', 'plot_pcsqcnn_architecture_sweep.py', 'plot_pcsqcnn_image_size_sweep.py', 'plot_gradient_norms.py', 'plot_readout_entropy.py', 'plot_finite_shot_accuracy.py', 'plot_finite_shot_loss_histograms.py', 'plot_readout_landscape.py', 'plot_error_structure.py', 'plot_frqi_comparison.py')
# Compact evidence for Supplementary Table S5, beyond the plotted runs.
EXTRA_RUN_DIRECTORIES: tuple[str, ...] = (
    "qcsqcnn_translated_mnist/fq2_ql3_sl1",
    "qcsqcnn_translated_mnist/fq2_ql3_sl8",
)
EXTRA_FILES: tuple[str, ...] = (
    "qcsqcnn_tuning/fq2_ql3_sl8/search_summary.json",
)


def data_plot_scripts() -> list[Path]:
    """Return the paper's artifact-based plotters in name order."""
    return [RUN_DIRECTORY / name for name in sorted(PUBLISHED_PLOT_SCRIPTS)]


def _trace_one(script: Path, artifacts_root: Path, render_dir: Path, trace_path: Path) -> None:
    """Run ``script`` in this process and record the files it opens below ``artifacts_root``."""

    root = str(artifacts_root.resolve())
    opened: set[str] = set()
    original_open = builtins.open

    def tracing_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = str(Path(file).resolve())
            if path.startswith(root + os.sep):
                opened.add(path[len(root) + 1 :])
        return original_open(file, *args, **kwargs)

    builtins.open = tracing_open
    io.open = tracing_open
    sys.argv = [str(script), "--artifacts-root", str(artifacts_root), "--output-dir", str(render_dir)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        builtins.open = original_open
        io.open = original_open
    trace_path.write_text(json.dumps(sorted(opened), indent=1), encoding="utf-8")


def _skip_reports(stderr: str) -> list[str]:
    return [line.strip() for line in stderr.splitlines() if line.startswith("Warning:") or "Skipping" in line]


def _render(script: Path, artifacts_root: Path, render_dir: Path, trace_path: Path) -> tuple[list[str], list[str]]:
    """Render ``script`` from ``artifacts_root`` in a fresh interpreter; return the files read and skip reports."""

    render_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--trace-one", str(script),
        "--source", str(artifacts_root), "--render-dir", str(render_dir), "--trace-path", str(trace_path),
    ]
    completed = subprocess.run(command, env={**os.environ, **RENDER_ENVIRONMENT}, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{script.name} failed:\n{completed.stderr[-2000:]}")
    return json.loads(trace_path.read_text(encoding="utf-8")), _skip_reports(completed.stderr)


def portable_metadata(value):
    """Relativize machine-specific data paths without changing numerical values.

    Refuse unknown absolute paths rather than silently publishing a broken link
    or discarding metadata. This operates on the JSON representation, preserving
    tensor, tuple and dataclass tags.
    """
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        parts = value.split("/")
        for anchor in ("artifacts", "data"):
            if anchor in parts:
                return "/".join(parts[parts.index(anchor):])
        raise ValueError(f"Unrecognized absolute path in exported metadata: {value}")
    return value


def export_file(source: Path, destination_root: Path, relative: str) -> str:
    """Write one portable plot input and return its relative path."""
    from qcnn.json_artifacts import load_payload, to_json_data

    source_path = source / relative
    target = Path(relative).with_suffix(".json") if source_path.suffix == ".pt" else Path(relative)
    destination = destination_root / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix == ".pt":
        data = to_json_data(load_payload(source_path))
        destination.write_text(json.dumps(portable_metadata(data), separators=(",", ":")), encoding="utf-8")
    elif source_path.suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        portable = portable_metadata(data)
        if portable == data:
            shutil.copyfile(source_path, destination)
        else:
            destination.write_text(json.dumps(portable, separators=(",", ":")), encoding="utf-8")
    else:
        raise ValueError(f"Unexpected artifact type: {source_path.suffix}")
    return target.as_posix()


def run_files(source: Path, relative_directory: str) -> list[str]:
    """Return the manifest and the per-seed result files of one saved run, relative to ``source``."""

    manifest = json.loads((source / relative_directory / "manifest.json").read_text(encoding="utf-8"))
    return [f"{relative_directory}/manifest.json", *(f"{relative_directory}/{run['result']}" for run in manifest["runs"])]


def compare_renders(reference: Path, candidate: Path) -> list[str]:
    """Return the relative paths of output files that differ or exist in only one folder."""

    names = {path.relative_to(reference).as_posix() for path in reference.rglob("*") if path.is_file()}
    names |= {path.relative_to(candidate).as_posix() for path in candidate.rglob("*") if path.is_file()}
    return sorted(
        name
        for name in names
        if not ((reference / name).is_file() and (candidate / name).is_file() and filecmp.cmp(reference / name, candidate / name, shallow=False))
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MNIST/FRQI figure inputs and comparator-table evidence as JSON.")
    parser.add_argument("--source", type=Path, required=True, help="Full artifacts folder written by the training and evaluation scripts.")
    parser.add_argument("--destination", type=Path, help="Folder that receives the compact figure and table inputs.")
    parser.add_argument("--check", action="store_true", help="Render every plot from both folders and compare the outputs.")
    parser.add_argument(
        "--extra-runs",
        nargs="*",
        default=list(EXTRA_RUN_DIRECTORIES),
        help="Run directories, relative to --source, to export in addition to the plot inputs (default: EXTRA_RUN_DIRECTORIES).",
    )
    parser.add_argument("--trace-one", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-path", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.trace_one is not None:
        _trace_one(args.trace_one, args.source, args.render_dir, args.trace_path)
        return
    if args.destination is None:
        raise SystemExit("--destination is required.")
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if source == destination or destination in source.parents or source in destination.parents:
        raise SystemExit("--destination must be a separate folder outside --source.")

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        index: dict[str, list[str]] = {}
        exported: dict[str, str] = {}
        for script in data_plot_scripts():
            opened, skipped = _render(script, source, work / "reference" / script.stem, work / f"{script.stem}.trace.json")
            for relative in opened:
                if relative not in exported:
                    exported[relative] = export_file(source, destination, relative)
            index[script.name] = sorted(exported[relative] for relative in opened)
            print(f"{script.name}: {len(opened)} files")
            for report in skipped:
                print(f"  source render: {report}")
        runs_index: dict[str, list[str]] = {}
        for relative_directory in args.extra_runs:
            files = run_files(source, relative_directory.rstrip("/"))
            for relative in files:
                if relative not in exported:
                    exported[relative] = export_file(source, destination, relative)
            runs_index[relative_directory] = sorted(exported[relative] for relative in files)
            print(f"{relative_directory}: {len(files)} files")
        for relative in EXTRA_FILES:
            if relative not in exported:
                exported[relative] = export_file(source, destination, relative)
            runs_index[relative] = [exported[relative]]
            print(f"{relative}: 1 file")
        (destination / INDEX_FILENAME).write_text(
            json.dumps({"plots": index, "runs": runs_index, "files": len(exported),
                        "sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                                   for name in sorted(set(exported.values()))}}, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(f"Exported {len(exported)} files to {destination}")

        if args.check:
            mismatches: dict[str, list[str]] = {}
            for script in data_plot_scripts():
                candidate = work / "candidate" / script.stem
                _, skipped = _render(script, destination, candidate, work / f"{script.stem}.check.json")
                for report in skipped:
                    print(f"  export render: {report}")
                differing = compare_renders(work / "reference" / script.stem, candidate)
                if differing:
                    mismatches[script.name] = differing
                print(f"check {script.name}: {'identical' if not differing else f'{len(differing)} outputs differ'}")
            if mismatches:
                raise SystemExit(f"Renders from {destination} differ from the full artifacts: {mismatches}")


if __name__ == "__main__":
    main()
