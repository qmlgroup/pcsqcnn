"""Finish the translated-MNIST MFA seeds while preserving the running GPU seed.

The original sequential orchestrator must be waiting for the live translated
seed-0 child. This script stops only the orchestrator, leaves that child
running, computes the other seeds, then resumes the orchestrator so it writes
its runtime and summary. It finally re-evaluates all three final checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TextIO

from run_mfa_full_matrix import _completed_run_is_valid, _write_runtime


@dataclass(frozen=True)
class JobSpec:
    regime: str
    seed: int
    device: str

    @property
    def directory_name(self) -> str:
        return f"full_{self.regime}_seed{self.seed}"


@dataclass
class ActiveJob:
    spec: JobSpec
    process: subprocess.Popen
    log: TextIO
    started: float
    run_path: Path


TRANSLATED_CPU = (JobSpec("translated", 1, "cpu"), JobSpec("translated", 2, "cpu"))


def _proc_details(pid: int) -> tuple[str, str] | None:
    proc = Path(f"/proc/{pid}")
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        state = (proc / "stat").read_text(encoding="utf-8").rsplit(") ", 1)[1][0]
    except FileNotFoundError:
        return None
    return command, state


def _validate_existing_jobs(orchestrator_pid: int, child_pid: int) -> None:
    parent = _proc_details(orchestrator_pid)
    child = _proc_details(child_pid)
    if parent is None or "run_mfa_full_matrix.py" not in parent[0] or "--device cuda" not in parent[0]:
        raise RuntimeError(f"PID {orchestrator_pid} is not the CUDA MFA orchestrator")
    if child is None or "prepare_mfa_baseline_data.py" not in child[0] or "full_translated_seed0" not in child[0]:
        raise RuntimeError(f"PID {child_pid} is not the live translated seed-0 trainer")
    ppid = int(Path(f"/proc/{child_pid}/status").read_text().split("PPid:\t", 1)[1].splitlines()[0])
    if ppid != orchestrator_pid:
        raise RuntimeError(f"Seed-0 trainer belongs to PID {ppid}, not {orchestrator_pid}")
    if parent[1] == "T":
        raise RuntimeError("The original orchestrator is already stopped")
    if child[1] in ("Z", "X"):
        raise RuntimeError("The original seed-0 trainer has already stopped")


def _abort_jobs(jobs: list[ActiveJob]) -> None:
    for job in jobs:
        if job.process.poll() is None:
            job.process.terminate()
    for job in jobs:
        try:
            job.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            job.process.kill()
            job.process.wait()
        job.log.close()


def _start_jobs(specs: tuple[JobSpec, ...], *, script: Path, data_root: Path,
                artifacts_root: Path, cpu_threads: int) -> list[ActiveJob]:
    jobs = []
    try:
        for spec in specs:
            run_path = artifacts_root / spec.directory_name
            if _completed_run_is_valid(run_path, regime=spec.regime, seed=spec.seed):
                print(f"Already complete: {run_path}", flush=True)
                continue
            command = [sys.executable, "-u", str(script),
                       "--regime", spec.regime, "--data-root", str(data_root),
                       "--artifacts-root", str(artifacts_root),
                       "--directory-name", spec.directory_name,
                       "--epochs", "2000", "--seed-count", "1",
                       "--base-seed", str(spec.seed), "--device", spec.device,
                       "--test-interval", "10"]
            if run_path.exists():
                command.append("--resume")
            env = os.environ.copy()
            if spec.device == "cpu":
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    env[key] = str(cpu_threads)
            log_path = artifacts_root / f"{spec.directory_name}.log"
            started = time.time()
            log = log_path.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            except BaseException:
                log.close()
                raise
            jobs.append(ActiveJob(spec, process, log, started, run_path))
            print(f"Started {spec.directory_name} on {spec.device}: PID {process.pid}", flush=True)
    except BaseException:
        _abort_jobs(jobs)
        raise
    return jobs


def _finish_jobs(jobs: list[ActiveJob]) -> None:
    errors = []
    for job in jobs:
        code = job.process.wait()
        ended = time.time()
        job.log.close()
        if code != 0:
            errors.append(f"{job.spec.directory_name} exited {code}")
            continue
        try:
            valid = _completed_run_is_valid(job.run_path, regime=job.spec.regime,
                                             seed=job.spec.seed)
            if not valid:
                raise RuntimeError("final manifest/checkpoints are missing")
            _write_runtime(job.run_path, job.started, ended)
        except Exception as exc:
            errors.append(f"{job.spec.directory_name}: {exc}")
        else:
            print(f"Complete {job.spec.directory_name}: {ended - job.started:.1f}s", flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))


def _wait_for_seed_zero(child_pid: int, run_path: Path) -> None:
    while True:
        valid = _completed_run_is_valid(run_path, regime="translated", seed=0)
        child = _proc_details(child_pid)
        if valid and (child is None or child[1] in ("Z", "X")):
            print(f"Original translated seed 0 complete: {run_path}", flush=True)
            return
        if child is None or child[1] in ("Z", "X"):
            raise RuntimeError("Original translated seed 0 stopped without complete artifacts")
        time.sleep(20)


def _wait_for_summary(path: Path, orchestrator_pid: int, *, since: float) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_mtime >= since:
            print(f"Orchestrator wrote {path}", flush=True)
            return
        if _proc_details(orchestrator_pid) is None:
            raise RuntimeError("Original orchestrator stopped without a benchmark summary")
        time.sleep(5)
    raise TimeoutError(f"Original orchestrator did not write {path} within ten minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrator-pid", type=int, required=True)
    parser.add_argument("--existing-seed0-pid", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/mfa"))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    _validate_existing_jobs(args.orchestrator_pid, args.existing_seed0_pid)
    for phase in (TRANSLATED_CPU,):
        print("Next: " + ", ".join(f"{job.directory_name}={job.device}" for job in phase),
              flush=True)
    if args.dry_run:
        return

    def stop_requested(signum: int, frame) -> None:
        del frame
        raise KeyboardInterrupt(f"Fanout received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop_requested)

    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.artifacts_root / "fanout.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = Path(__file__).with_name("prepare_mfa_baseline_data.py")
        _validate_existing_jobs(args.orchestrator_pid, args.existing_seed0_pid)
        started = time.time()
        active_jobs: list[ActiveJob] = []
        try:
            os.kill(args.orchestrator_pid, signal.SIGSTOP)
            print(f"Paused original orchestrator PID {args.orchestrator_pid}; seed 0 continues", flush=True)
            active_jobs = _start_jobs(TRANSLATED_CPU, script=script,
                                      data_root=args.data_root,
                                      artifacts_root=args.artifacts_root,
                                      cpu_threads=args.cpu_threads)
            _finish_jobs(active_jobs)
            active_jobs = []
            _wait_for_seed_zero(args.existing_seed0_pid,
                                args.artifacts_root / "full_translated_seed0")
        finally:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            _abort_jobs(active_jobs)
            if _proc_details(args.orchestrator_pid) is not None:
                os.kill(args.orchestrator_pid, signal.SIGCONT)
                print(f"Resumed original orchestrator PID {args.orchestrator_pid}", flush=True)

        _wait_for_summary(args.artifacts_root / "benchmark_summary.json",
                          args.orchestrator_pid, since=started)
        verify = Path(__file__).with_name("verify_mfa_benchmarks.py")
        subprocess.run([sys.executable, str(verify), "--root", str(args.artifacts_root),
                        "--data-root", str(args.data_root), "--device", "cuda"], check=True)
        print("All three translated MFA runs independently verified", flush=True)


if __name__ == "__main__":
    main()
