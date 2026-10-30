"""Run the main translated-MNIST MFA benchmark, one seed at a time.

Every seed uses the fixed 2000-epoch protocol. Completed runs are checked and
skipped; runs with checkpoints but no final manifest resume automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _completed_run_is_valid(path: Path, *, regime: str, seed: int) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["resolved_config"]
    dataset = config["dataset"]
    expected_scaled_size = 16 if regime == "translated" else 32
    expected_offset = 8 if regime == "translated" else 0
    expected_samples = 1000 if regime == "translated" else None
    expected_batch = 256 if regime == "translated" else 512
    if (
        manifest["seeds"] != [seed]
        or config["model"]["class_name"] != "MFAClassifier"
        or config["optimizer"]["kind"] != "mfa_adam"
        or config["optimizer"]["learning_rate"] != 0.01
        or config["training"]["num_epochs"] != 2000
        or config["training"]["test_evaluation_interval_epochs"] != 10
        or dataset["samples_per_class"] != expected_samples
        or dataset["image_size"] != 32
        or dataset["scaled_image_size"] != expected_scaled_size
        or dataset["max_offset"] != expected_offset
        or dataset["train_batch_size"] != expected_batch
        or dataset["test_batch_size"] != 1600
    ):
        raise ValueError(f"Existing run has a different protocol: {path}")
    if not (path / f"checkpoint_final_seed{seed}.pt").is_file():
        raise ValueError(f"Existing manifest has no final model checkpoint: {path}")
    if not (path / f"result_seed{seed}.pt").is_file():
        raise ValueError(f"Existing manifest has no result payload: {path}")
    return True


def _wait_for_first_run(pid: int, path: Path) -> None:
    stat_tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1]
    start_ticks = int(stat_tail.split()[19])
    boot_line = next(
        line for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        if line.startswith("btime ")
    )
    start_unix_time = int(boot_line.split()[1]) + start_ticks / os.sysconf("SC_CLK_TCK")
    print(f"Waiting for running seed at PID {pid}: {path}", flush=True)
    while not (path / "manifest.json").is_file():
        try:
            os.kill(pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError(f"First MFA run stopped before writing its manifest: {path}") from exc
        time.sleep(20)
    end_unix_time = (path / "manifest.json").stat().st_mtime
    _write_runtime(path, start_unix_time, end_unix_time)


def _write_runtime(path: Path, start_unix_time: float, end_unix_time: float) -> None:
    (path / "runtime.json").write_text(
        json.dumps(
            {
                "start_unix_time": start_unix_time,
                "end_unix_time": end_unix_time,
                "wall_seconds": end_unix_time - start_unix_time,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/mfa"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()

    script = Path(__file__).with_name("prepare_mfa_baseline_data.py")
    for regime in ("translated",):
        for seed in range(3):
            directory_name = f"full_{regime}_seed{seed}"
            run_path = args.artifacts_root / directory_name
            if regime == "translated" and seed == 0 and args.wait_pid is not None:
                _wait_for_first_run(args.wait_pid, run_path)
            if _completed_run_is_valid(run_path, regime=regime, seed=seed):
                print(f"Complete: {run_path}", flush=True)
                continue
            command = [
                sys.executable,
                "-u",
                str(script),
                "--regime", regime,
                "--data-root", str(args.data_root),
                "--artifacts-root", str(args.artifacts_root),
                "--directory-name", directory_name,
                "--epochs", "2000",
                "--seed-count", "1",
                "--base-seed", str(seed),
                "--device", args.device,
                "--test-interval", "10",
            ]
            if run_path.exists():
                command.append("--resume")
            args.artifacts_root.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_root / f"{directory_name}.log"
            print(f"Running {directory_name}; log: {log_path}", flush=True)
            start_unix_time = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            end_unix_time = time.time()
            if not _completed_run_is_valid(run_path, regime=regime, seed=seed):
                raise RuntimeError(f"Run finished without complete artifacts: {run_path}")
            _write_runtime(run_path, start_unix_time, end_unix_time)
            print(f"Complete: {run_path}", flush=True)

    subprocess.run(
        [sys.executable, str(script.with_name("summarize_mfa_benchmarks.py")),
         "--root", str(args.artifacts_root)],
        check=True,
    )


if __name__ == "__main__":
    main()
