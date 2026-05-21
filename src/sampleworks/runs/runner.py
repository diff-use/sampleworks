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


DEFAULT_GRID_SEARCH_SCRIPT = "/app/run_grid_search.py"


@dataclass(frozen=True)
class JobInvocation:
    """The fully resolved command to launch for one job.

    Parameters
    ----------
    job : Job
        Originating :class:`Job` (kept for introspection in logs).
    argv : list of str
        Subprocess command line (starts with ``pixi run -e <env> python ...``).
    env : dict of str to str
        Process environment, including ``CUDA_VISIBLE_DEVICES``.
    log_path : Path
        File to tee stdout+stderr into.
    output_dir : Path
        Resolved ``--output-dir`` value (mkdir'd by the runner before launch
        because ``run_grid_search.py`` assumes its existence).
    """

    job: Job
    argv: list[str]
    env: dict[str, str]
    log_path: Path
    output_dir: Path


def build_invocations(preset: Preset, *, results_dir: Path) -> list[JobInvocation]:
    """Build the subprocess invocation for every job in the preset.

    Per-job ``args`` are merged on top of :attr:`Preset.shared_args`, with
    ``--output-dir`` auto-injected from ``results_dir / job.output_subdir`` if
    not already present.

    Parameters
    ----------
    preset : Preset
        Resolved preset to launch.
    results_dir : Path
        Root directory for outputs and per-job log files.

    Returns
    -------
    list of JobInvocation
        One :class:`JobInvocation` per job, in declaration order.
    """
    invocations: list[JobInvocation] = []
    for job in preset.jobs:
        args = preset.effective_args(job)
        args.setdefault("output-dir", str(results_dir / job.output_subdir))
        argv = _build_argv(job.env, args)
        env = _job_env(job.env, {**os.environ, "CUDA_VISIBLE_DEVICES": job.gpus})
        log_path = results_dir / f"{job.name}_run.log"
        output_dir = Path(args["output-dir"])
        invocations.append(
            JobInvocation(job=job, argv=argv, env=env, log_path=log_path, output_dir=output_dir)
        )
    return invocations


def _build_argv(pixi_env: str, args: dict[str, Any]) -> list[str]:
    """Assemble the ``pixi run`` argv list for one job's args dict.

    ``True`` bools become bare flags, ``False``/``None`` are dropped, all other
    values are stringified.

    Parameters
    ----------
    pixi_env : str
        Pixi environment name passed to ``-e``.
    args : dict of str to Any
        Flag-name to value map (kebab-case keys, no leading ``--``).

    Returns
    -------
    list of str
        Subprocess argv.
    """
    env_python = _pixi_env_python(pixi_env)
    if env_python:
        argv = [env_python, _grid_search_script()]
    else:
        argv = ["pixi", "run", "-e", pixi_env, "python", _grid_search_script()]
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


def _pixi_env_python(pixi_env: str) -> str | None:
    """Return the direct Python binary for a baked pixi environment when available.

    The sampleworks ACTL image already contains fully-installed environments at
    ``/app/.pixi/envs/<env>``. Calling those Python binaries directly avoids
    ``pixi run`` trying to refresh Git/PyPI caches on shared pod storage.

    Parameters
    ----------
    pixi_env : str
        Pixi environment name from the preset job.

    Returns
    -------
    str or None
        Executable Python path, or ``None`` to fall back to ``pixi run``.
    """
    if os.environ.get("SAMPLEWORKS_FORCE_PIXI", "").lower() in {"1", "true", "yes"}:
        return None

    env_key = pixi_env.upper().replace("-", "_")
    override = os.environ.get(f"SAMPLEWORKS_{env_key}_PYTHON")
    if override:
        return override

    candidate = _pixi_project_dir() / ".pixi" / "envs" / pixi_env / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _job_env(pixi_env: str, env: dict[str, str]) -> dict[str, str]:
    """Return an environment equivalent to activating a direct pixi env.

    Parameters
    ----------
    pixi_env : str
        Pixi environment name used by the job.
    env : dict of str to str
        Base process environment.

    Returns
    -------
    dict of str to str
        Environment with the pixi env's ``bin`` directory and compiler/CUDA
        paths exposed when the job runs a direct Python binary.
    """
    env_python = _pixi_env_python(pixi_env)
    if env_python is None:
        return env

    env_dir = Path(env_python).resolve().parent.parent
    bin_dir = env_dir / "bin"
    activated = dict(env)
    activated["PATH"] = f"{bin_dir}{os.pathsep}{activated.get('PATH', '')}"
    activated["CONDA_PREFIX"] = str(env_dir)
    activated.setdefault("CUDA_HOME", str(env_dir))
    activated["PYTHONNOUSERSITE"] = "1"
    return activated


def _pixi_project_dir() -> Path:
    """Return the pixi project directory for env lookup and fallback pixi runs.

    Returns
    -------
    Path
        Project directory, defaulting to ``/app`` for the sampleworks image or
        the current working directory outside that image.
    """
    override = os.environ.get("SAMPLEWORKS_PIXI_PROJECT_DIR")
    if override:
        return Path(override)
    app = Path("/app")
    if (app / "pyproject.toml").exists():
        return app
    return Path.cwd()


def _grid_search_script() -> str:
    """Return the ``run_grid_search.py`` path used by worker jobs.

    Resolution is intentionally simple for the ACTL sampleworks image: the
    baked image keeps a stable copy at :data:`DEFAULT_GRID_SEARCH_SCRIPT`, while
    synced PR worktrees can point the runner at their checkout with
    ``SAMPLEWORKS_GRID_SEARCH_SCRIPT=/home/dev/workspace/run_grid_search.py``.

    Returns
    -------
    str
        Path to execute with ``python`` inside each pixi environment.
    """
    return os.environ.get("SAMPLEWORKS_GRID_SEARCH_SCRIPT", DEFAULT_GRID_SEARCH_SCRIPT)


def run(preset: Preset, *, results_dir: Path, dry_run: bool = False) -> int:
    """Launch every job in parallel and wait for completion.

    Stdout+stderr from each job is teed to a per-job log file under
    ``results_dir`` and also echoed to the driver's stderr with a ``[job_name]``
    prefix.

    Parameters
    ----------
    preset : Preset
        Preset to launch.
    results_dir : Path
        Root directory for outputs and logs. Created if missing.
    dry_run : bool, optional
        If True, print the resolved commands instead of launching anything.

    Returns
    -------
    int
        ``0`` if all jobs exited 0 (or ``dry_run`` was set), ``1`` otherwise.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    invocations = build_invocations(preset, results_dir=results_dir)

    if dry_run:
        for inv in invocations:
            _print_dry_run(inv)
        return 0

    pixi_envs = sorted({inv.job.env for inv in invocations})
    for pixi_env in pixi_envs:
        _prepare_pixi_env(pixi_env)
    invocations = build_invocations(preset, results_dir=results_dir)

    _print_launch_summary(preset, invocations)
    processes: list[_RunningJob] = []
    try:
        for inv in invocations:
            processes.append(_spawn(inv))
    except BaseException:
        _terminate_all(processes)
        raise
    return _wait_all(processes)


def _terminate_all(jobs: list[_RunningJob]) -> None:
    """Terminate any already-launched jobs (used when a later spawn fails).

    Parameters
    ----------
    jobs : list of _RunningJob
        Jobs whose subprocesses should be SIGTERM'd, waited on, and whose tee
        threads should be joined.
    """
    for j in jobs:
        if j.proc.poll() is None:
            j.proc.terminate()
    for j in jobs:
        j.proc.wait()
        j.tee_thread.join()


def _prepare_pixi_env(pixi_env: str) -> None:
    """Prepare a pixi environment before parallel job launch.

    ``pixi run`` is deliberately called once per env even when the interpreter
    directory already exists, because pixi may still need to materialize PyPI
    packages into that environment after image startup.

    Parameters
    ----------
    pixi_env : str
        Pixi environment to prepare.

    Raises
    ------
    subprocess.CalledProcessError
        If pixi cannot prepare the environment.
    """
    if os.environ.get("SAMPLEWORKS_SKIP_ENV_PREPARE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return

    env = {
        **os.environ,
        "PIXI_CACHE_DIR": os.environ.get("PIXI_CACHE_DIR", "/tmp/pixi-cache"),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache"),
    }
    cmd = ["pixi", "run", "-e", pixi_env, "python", "-c", "print('ready')"]
    print(
        f"[{_ts()}] preparing pixi env {pixi_env!r} with {shlex.join(cmd)}",
        file=sys.stderr,
    )
    subprocess.run(cmd, cwd=str(_pixi_project_dir()), env=env, check=True)


def _print_dry_run(inv: JobInvocation) -> None:
    """Print the exact command for one job without launching it.

    Parameters
    ----------
    inv : JobInvocation
        Invocation to print.
    """
    print(f"# job: {inv.job.name}  (env={inv.job.env}, gpus={inv.job.gpus})", file=sys.stderr)
    print(f"# log: {inv.log_path}", file=sys.stderr)
    print(f"CUDA_VISIBLE_DEVICES={inv.job.gpus} {_shell_join(inv.argv)}")
    print(file=sys.stderr)


def _print_launch_summary(preset: Preset, invocations: list[JobInvocation]) -> None:
    """Print a banner describing what is about to be launched.

    Parameters
    ----------
    preset : Preset
        Preset being launched.
    invocations : list of JobInvocation
        Jobs about to be spawned.
    """
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


@dataclass(frozen=True)
class _RunningJob:
    """Internal handle: a spawned subprocess and its log-tee thread.

    Parameters
    ----------
    inv : JobInvocation
        Originating invocation.
    proc : subprocess.Popen
        The subprocess (PIPE'd stdout merged with stderr).
    tee_thread : threading.Thread
        Daemon thread copying ``proc.stdout`` to the log file and to
        ``sys.stderr`` with a per-job prefix.
    """

    inv: JobInvocation
    proc: subprocess.Popen[bytes]
    tee_thread: threading.Thread


def _spawn(inv: JobInvocation) -> _RunningJob:
    """Start one subprocess and a thread to tee its output.

    Parameters
    ----------
    inv : JobInvocation
        Invocation to spawn.

    Returns
    -------
    _RunningJob
        Handle covering the subprocess and the tee thread.

    Raises
    ------
    OSError
        Propagated if the subprocess fails to start (e.g. binary missing).
    """
    inv.log_path.parent.mkdir(parents=True, exist_ok=True)
    inv.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(inv.log_path, "wb")
    try:
        proc = subprocess.Popen(
            inv.argv,
            env=inv.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except BaseException:
        log_file.close()
        raise
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
    """Wait for every job to exit and aggregate their exit codes.

    Parameters
    ----------
    jobs : list of _RunningJob
        Jobs to wait on.

    Returns
    -------
    int
        ``0`` if all jobs exited 0, ``1`` if any failed.
    """
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
    """Copy bytes from ``src`` to ``dest`` and to stderr with a label.

    Parameters
    ----------
    prefix : str
        Per-line label prepended to the stderr echo (e.g. job name).
    src : file-like
        Readable byte stream (typically ``Popen.stdout`` with stderr merged).
    dest : file-like
        Writable byte stream for the on-disk log file. Closed when ``src`` is
        exhausted.
    """
    for line in iter(src.readline, b""):
        dest.write(line)
        dest.flush()
        sys.stderr.write(f"[{prefix}] {line.decode('utf-8', errors='replace')}")
        sys.stderr.flush()
    dest.close()


def _ts() -> str:
    """Return the current local time as a ``YYYY-MM-DD HH:MM:SS`` string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _shell_join(argv: list[str]) -> str:
    """Quote ``argv`` so the result can be pasted into a POSIX shell.

    Parameters
    ----------
    argv : list of str
        Argument vector.

    Returns
    -------
    str
        Single shell-quoted command line.
    """
    return shlex.join(argv)
