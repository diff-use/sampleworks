"""Build job argv and orchestrate parallel subprocess execution."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Job, Preset


GRID_SEARCH_SCRIPT = "/app/run_grid_search.py"


@dataclass
class JobInvocation:
    job: Job
    argv: list[str]
    env: dict[str, str]
    log_path: Path


def build_invocations(preset: Preset, *, results_dir: Path) -> list[JobInvocation]:
    """Build the subprocess argv + env + log path for every job in the preset."""
    invocations: list[JobInvocation] = []
    for job in preset.jobs:
        args = preset.effective_args(job)
        args.setdefault("output-dir", str(results_dir / job.output_subdir))
        argv = _build_argv(job.env, args)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": job.gpus}
        log_path = results_dir / f"{job.name}_run.log"
        invocations.append(JobInvocation(job=job, argv=argv, env=env, log_path=log_path))
    return invocations


def _build_argv(pixi_env: str, args: dict[str, Any]) -> list[str]:
    argv = ["pixi", "run", "-e", pixi_env, "python", GRID_SEARCH_SCRIPT]
    for key, value in args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is None:
            continue
        else:
            argv.extend([flag, str(value)])
    return argv


def run(preset: Preset, *, results_dir: Path, dry_run: bool = False) -> int:
    """Launch every job in parallel; tee output to per-job logs; return 0 iff all succeed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    invocations = build_invocations(preset, results_dir=results_dir)

    if dry_run:
        for inv in invocations:
            _print_dry_run(inv)
        return 0

    _print_launch_summary(preset, invocations)
    processes = [_spawn(inv) for inv in invocations]
    return _wait_all(processes)


def _print_dry_run(inv: JobInvocation) -> None:
    print(f"# job: {inv.job.name}  (env={inv.job.env}, gpus={inv.job.gpus})", file=sys.stderr)
    print(f"# log: {inv.log_path}", file=sys.stderr)
    print(f"CUDA_VISIBLE_DEVICES={inv.job.gpus} {_shell_join(inv.argv)}")
    print(file=sys.stderr)


def _print_launch_summary(preset: Preset, invocations: list[JobInvocation]) -> None:
    bar = "=" * 60
    print(bar, file=sys.stderr)
    print(f"preset: {preset.name}", file=sys.stderr)
    if preset.description:
        print(f"  {preset.description}", file=sys.stderr)
    for inv in invocations:
        print(
            f"  - {inv.job.name}: env={inv.job.env}, gpus={inv.job.gpus}, log={inv.log_path}",
            file=sys.stderr,
        )
    print(bar, file=sys.stderr)


@dataclass
class _RunningJob:
    inv: JobInvocation
    proc: subprocess.Popen[bytes]
    tee_thread: threading.Thread


def _spawn(inv: JobInvocation) -> _RunningJob:
    inv.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(inv.log_path, "wb")
    proc = subprocess.Popen(
        inv.argv,
        env=inv.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    thread = threading.Thread(
        target=_tee,
        args=(inv.job.name, proc.stdout, log_file),
        daemon=True,
    )
    thread.start()
    print(f"[{_ts()}] launched {inv.job.name} (pid {proc.pid})", file=sys.stderr)
    return _RunningJob(inv=inv, proc=proc, tee_thread=thread)


def _wait_all(jobs: list[_RunningJob]) -> int:
    failures = 0
    for j in jobs:
        exit_code = j.proc.wait()
        j.tee_thread.join()
        if exit_code == 0:
            print(f"[{_ts()}] {j.inv.job.name} succeeded", file=sys.stderr)
        else:
            print(f"[{_ts()}] {j.inv.job.name} FAILED (exit {exit_code})", file=sys.stderr)
            failures += 1
    return 0 if failures == 0 else 1


def _tee(prefix: str, src: Any, dest: Any) -> None:
    for line in iter(src.readline, b""):
        dest.write(line)
        dest.flush()
        sys.stderr.write(f"[{prefix}] {line.decode('utf-8', errors='replace')}")
        sys.stderr.flush()
    dest.close()


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _shell_join(argv: list[str]) -> str:
    return shlex.join(argv)
