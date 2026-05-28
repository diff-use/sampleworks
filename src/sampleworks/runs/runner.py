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
WORKSPACE_GRID_SEARCH_SCRIPT = "/home/dev/workspace/run_grid_search.py"
DISABLE_GPU_ASSIGNMENTS = frozenset({"none", "void", "nodevfiles"})
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 10
TEE_THREAD_JOIN_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class JobInvocation:
    """The fully resolved command to launch for one job.

    Parameters
    ----------
    job : Job
        Originating :class:`Job` (kept for introspection in logs).
    argv : list of str
        Subprocess command line, preferably the baked pixi env Python followed
        by the job script.
    env : dict of str to str
        Process environment, including ``CUDA_VISIBLE_DEVICES``.
    gpus : str
        Resolved CUDA-visible GPU assignment. For jobs that declare
        ``gpu_count``, this is the concrete auto-assigned GPU list.
    log_path : Path
        File to tee stdout+stderr into.
    output_dir : Path
        Directory mkdir'd by the runner before launch. Experiment jobs use this
        as their injected ``--output-dir``; analysis jobs can use it only as a
        side-effect directory for logs or scratch space.
    """

    job: Job
    argv: list[str]
    env: dict[str, str]
    gpus: str
    log_path: Path
    output_dir: Path


def build_invocations(preset: Preset, *, results_dir: Path) -> list[JobInvocation]:
    """Build the subprocess invocation for every job in the preset.

    Per-job ``args`` are merged on top of :attr:`Preset.shared_args`, with a
    job-specific output argument auto-injected from
    ``results_dir / job.output_subdir`` when configured and absent.

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
    gpu_assignments = _resolve_gpu_assignments(preset.jobs)
    invocations: list[JobInvocation] = []
    for job in preset.jobs:
        args = preset.effective_args(job)
        output_dir = results_dir / job.output_subdir
        if job.output_arg:
            args.setdefault(job.output_arg, str(output_dir))
            output_dir = Path(args[job.output_arg])
        argv = _build_argv(job.env, args, script=job.script or None)
        gpus = gpu_assignments[job.name]
        env = _job_env(job.env, {**os.environ, "CUDA_VISIBLE_DEVICES": gpus})
        log_path = results_dir / f"{job.name}_run.log"
        invocations.append(
            JobInvocation(
                job=job,
                argv=argv,
                env=env,
                gpus=gpus,
                log_path=log_path,
                output_dir=output_dir,
            )
        )
    return invocations


def _resolve_gpu_assignments(jobs: list[Job]) -> dict[str, str]:
    """Resolve explicit ``gpus`` and automatic ``gpu_count`` declarations.

    Explicit assignments reserve those GPU tokens. Jobs with ``gpu_count`` then
    consume remaining visible GPU IDs in preset declaration order. When GPU
    discovery is unavailable (for local dry-runs/tests), synthetic ordinals are
    generated so command construction stays deterministic.
    """
    explicit: dict[str, str] = {job.name: job.gpus for job in jobs if job.gpus}
    reserved = {gpu for value in explicit.values() for gpu in _split_gpu_list(value)}
    total_auto = sum(job.gpu_count or 0 for job in jobs)
    available = _detect_available_gpus()
    if available:
        pool = [gpu for gpu in available if gpu not in reserved]
        if len(pool) < total_auto:
            raise RuntimeError(
                "Not enough visible GPUs for preset auto-assignment. "
                f"Visible GPUs: {available}. Reserved GPUs: {sorted(reserved)}. "
                f"Auto-requested GPUs: {total_auto}."
            )
    elif _cuda_visible_devices_disables_gpus() and total_auto:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES disables GPU access, so gpu_count auto-assignment "
            "cannot allocate any GPUs."
        )
    else:
        pool = _synthetic_gpu_pool(reserved, total_auto)

    assignments: dict[str, str] = {}
    cursor = 0
    for job in jobs:
        if job.gpus:
            assignments[job.name] = job.gpus
            continue
        count = job.gpu_count or 0
        assigned = pool[cursor : cursor + count]
        cursor += count
        assignments[job.name] = ",".join(assigned)
    return assignments


def _synthetic_gpu_pool(reserved: set[str], count: int) -> list[str]:
    """Return deterministic CUDA ordinals when real GPU discovery is unavailable."""
    pool: list[str] = []
    candidate = 0
    while len(pool) < count:
        token = str(candidate)
        if token not in reserved:
            pool.append(token)
        candidate += 1
    return pool


def _split_gpu_list(value: str) -> list[str]:
    """Split a comma-separated GPU assignment into normalized tokens.

    Parameters
    ----------
    value : str
        GPU assignment string such as ``"0,1"``.

    Returns
    -------
    list of str
        Non-empty stripped GPU tokens.
    """
    return [part.strip() for part in value.split(",") if part.strip()]


def _gpu_assignment_disables_gpus(value: str) -> bool:
    """Return True when an explicit assignment intentionally hides all GPUs.

    Parameters
    ----------
    value : str
        CUDA_VISIBLE_DEVICES assignment from a job, such as ``"none"``.

    Returns
    -------
    bool
        True when ``value`` is one of the CUDA-recognized no-GPU tokens.
    """
    return value.strip().lower() in DISABLE_GPU_ASSIGNMENTS


def _all_integer_tokens(values: list[str]) -> bool:
    """Return True when every GPU token is a CUDA ordinal.

    Parameters
    ----------
    values : list of str
        GPU tokens to inspect.

    Returns
    -------
    bool
        True if all tokens are non-negative integer strings.
    """
    return all(value.isdigit() for value in values)


def _detect_available_gpus() -> list[str]:
    """Return GPU ordinals visible to the runner process.

    Returns
    -------
    list of str
        Visible GPU identifiers, or an empty list when GPU discovery is not
        available. Empty means validation should be skipped.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cuda_visible_key = cuda_visible.lower()
    if _cuda_visible_devices_disables_gpus():
        return []
    if cuda_visible and cuda_visible_key != "all":
        return _split_gpu_list(cuda_visible)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _cuda_visible_devices_disables_gpus() -> bool:
    """Return True when CUDA_VISIBLE_DEVICES explicitly hides all GPUs."""
    return _gpu_assignment_disables_gpus(os.environ.get("CUDA_VISIBLE_DEVICES", ""))


def _validate_gpu_assignments(invocations: list[JobInvocation]) -> None:
    """Fail fast when a preset asks for GPUs not present in this pod.

    Parameters
    ----------
    invocations : list of JobInvocation
        Jobs whose ``CUDA_VISIBLE_DEVICES`` assignments should be checked.

    Raises
    ------
    RuntimeError
        If numeric preset assignments reference unavailable visible GPU IDs,
        or if multiple jobs claim the same GPU without opting into
        oversubscription.
    """
    available = _detect_available_gpus()
    if not available:
        return

    requested: dict[str, list[str]] = {}
    for inv in invocations:
        if _gpu_assignment_disables_gpus(inv.gpus):
            continue
        for gpu in _split_gpu_list(inv.gpus):
            requested.setdefault(gpu, []).append(inv.job.name)

    requested_tokens = list(requested)
    if not _all_integer_tokens(available) or not _all_integer_tokens(requested_tokens):
        return

    available_set = set(available)
    unavailable = {gpu: names for gpu, names in requested.items() if gpu not in available_set}
    if unavailable:
        details = ", ".join(
            f"GPU {gpu} requested by {', '.join(names)}"
            for gpu, names in sorted(unavailable.items())
        )
        raise RuntimeError(
            "Preset requests GPUs that are not visible in this pod. "
            f"Visible GPUs: {', '.join(available)}. {details}. "
            "Edit the preset's jobs.*.gpus/gpu_count values or run a smaller --jobs subset."
        )

    allow_oversubscription = os.environ.get(
        "SAMPLEWORKS_ALLOW_GPU_OVERSUBSCRIPTION", ""
    ).lower() in {"1", "true", "yes"}
    duplicates = {gpu: names for gpu, names in requested.items() if len(names) > 1}
    if duplicates and not allow_oversubscription:
        details = ", ".join(
            f"GPU {gpu} requested by {', '.join(names)}"
            for gpu, names in sorted(duplicates.items())
        )
        raise RuntimeError(
            "Preset assigns the same GPU to multiple jobs. "
            f"{details}. Set SAMPLEWORKS_ALLOW_GPU_OVERSUBSCRIPTION=1 to allow this."
        )


def _build_argv(pixi_env: str, args: dict[str, Any], *, script: str | None = None) -> list[str]:
    """Assemble the ``pixi run`` argv list for one job's args dict.

    ``True`` bools become bare flags, ``False``/``None`` are dropped, all other
    values are stringified.

    Parameters
    ----------
    pixi_env : str
        Pixi environment name passed to ``-e``.
    args : dict of str to Any
        Flag-name to value map (kebab-case keys, no leading ``--``).
    script : str or None, optional
        Script path to execute. If ``None``, the default grid-search script is
        used for backward-compatible experiment presets.

    Returns
    -------
    list of str
        Subprocess argv.
    """
    script_path = _resolve_script_path(script)
    env_python = _pixi_env_python(pixi_env)
    if env_python:
        argv = [env_python, script_path]
    elif _require_prebuilt_envs():
        raise RuntimeError(_missing_prebuilt_env_message(pixi_env))
    else:
        argv = ["pixi", "run", "-e", pixi_env, "python", script_path]
    for key, value in args.items():
        _append_cli_arg(argv, key, value)
    return argv


def _append_cli_arg(argv: list[str], key: str, value: Any) -> None:
    """Append one TOML-configured CLI argument to ``argv``.

    Parameters
    ----------
    argv : list of str
        Mutable command vector to extend.
    key : str
        Flag name without leading ``--``.
    value : Any
        TOML value. ``True`` emits a bare flag, ``False``/``None`` are omitted,
        lists emit one flag followed by one value per element, and all other
        values are stringified.
    """
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            argv.append(flag)
    elif value is None:
        return
    elif isinstance(value, (list, tuple)):
        if value:
            argv.append(flag)
            argv.extend(str(item) for item in value)
    else:
        argv.extend([flag, str(value)])


def _pixi_env_python(pixi_env: str) -> str | None:
    """Return the direct Python binary for a baked pixi environment when available.

    The ACTL pixi/checkpoint image already contains fully-installed environments at
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


def _truthy_env(name: str) -> bool:
    """Return True when an environment variable is set to a truthy value.

    Parameters
    ----------
    name : str
        Environment variable name to inspect.

    Returns
    -------
    bool
        True for ``1``, ``true``, or ``yes`` values, case-insensitive.
    """
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _require_prebuilt_envs() -> bool:
    """Return True when runtime pixi fallback must be disabled.

    Returns
    -------
    bool
        True when the ACTL wrapper/image requires baked pixi environments and
        the caller has not explicitly opted into runtime pixi installation.
    """
    allow_runtime_pixi = _truthy_env("RUNTIME_PIXI") or _truthy_env(
        "SAMPLEWORKS_ALLOW_RUNTIME_PIXI"
    )
    return _truthy_env("SAMPLEWORKS_REQUIRE_PREBUILT_PIXI") and not allow_runtime_pixi


def _missing_prebuilt_env_message(pixi_env: str) -> str:
    """Build the error message for a missing baked pixi environment.

    Parameters
    ----------
    pixi_env : str
        Required pixi environment name.

    Returns
    -------
    str
        Human-readable error message explaining how to fix the pod/image.
    """
    expected = _pixi_project_dir() / ".pixi" / "envs" / pixi_env / "bin" / "python"
    return (
        f"Prebuilt pixi environment is missing for job env {pixi_env!r}: {expected}. "
        "The pixi-with-checkpoints image must contain ready-to-use boltz, "
        "protenix, and rf3 environments. Refusing to fall back to 'pixi run' "
        "because that would install or refresh packages inside the pod. "
        "Recreate the pod with the current image, or set RUNTIME_PIXI=1 only "
        "when intentionally debugging pixi."
    )


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
        Project directory, defaulting to ``/app`` for the ACTL pixi/checkpoint
        image or the current working directory outside that image.
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

    Prefer the synced ACTL checkout when it exists; otherwise fall back to the
    historical baked ``/app`` path or an explicit
    ``SAMPLEWORKS_GRID_SEARCH_SCRIPT``.

    Returns
    -------
    str
        Path to execute with ``python`` inside each pixi environment.
    """
    override = os.environ.get("SAMPLEWORKS_GRID_SEARCH_SCRIPT")
    if override:
        return override
    workspace_script = Path(WORKSPACE_GRID_SEARCH_SCRIPT)
    if workspace_script.exists():
        return str(workspace_script)
    return DEFAULT_GRID_SEARCH_SCRIPT


def _resolve_script_path(script: str | None) -> str:
    """Resolve a job script path for subprocess execution.

    Parameters
    ----------
    script : str or None
        Script configured by a TOML job. Empty/``None`` selects the historical
        ``run_grid_search.py`` default.

    Returns
    -------
    str
        Absolute path when the script can be found in a known Sampleworks
        checkout, otherwise the original expanded path so subprocess startup
        errors remain clear.
    """
    if not script:
        return _grid_search_script()

    expanded = Path(os.path.expandvars(script)).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    for root in _source_root_candidates():
        candidate = root / expanded
        if candidate.exists():
            return str(candidate)
    return str(expanded)


def _source_root_candidates() -> list[Path]:
    """Return likely Sampleworks checkout roots for relative script paths.

    Returns
    -------
    list of pathlib.Path
        Candidate directories in precedence order, deduplicated after
        expansion. The synced ACTL checkout and current working directory are
        preferred over stale files under the baked image.
    """
    candidates: list[Path] = []
    for env_var in ("SAMPLEWORKS_SOURCE_DIR", "SAMPLEWORKS_SCRIPT_ROOT"):
        override = os.environ.get(env_var)
        if override:
            candidates.append(Path(override))

    candidates.append(Path("/home/dev/workspace"))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    module_path = Path(__file__).resolve()
    candidates.extend([module_path.parent, *module_path.parents])

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


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
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    invocations = build_invocations(preset, results_dir=results_dir)
    _validate_gpu_assignments(invocations)

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
        Jobs whose subprocesses should be SIGTERM'd, escalated to SIGKILL if
        needed, and whose tee threads should be joined with bounded waits.
    """
    for j in jobs:
        if j.proc.poll() is None:
            j.proc.terminate()
    for j in jobs:
        try:
            j.proc.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            j.proc.kill()
            try:
                j.proc.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(
                    f"[{_ts()}] {j.inv.job.name} did not exit after SIGKILL",
                    file=sys.stderr,
                )
        j.tee_thread.join(timeout=TEE_THREAD_JOIN_TIMEOUT_SECONDS)


def _prepare_pixi_env(pixi_env: str) -> None:
    """Prepare a pixi environment before parallel job launch.

    Preparation is skipped when a baked interpreter is already available, when
    prebuilt environments are required, or when ``SAMPLEWORKS_SKIP_ENV_PREPARE``
    is truthy. Otherwise, ``pixi run`` is called once for the environment.

    Parameters
    ----------
    pixi_env : str
        Pixi environment to prepare.

    Raises
    ------
    subprocess.CalledProcessError
        If pixi cannot prepare the environment.
    """
    if _pixi_env_python(pixi_env) is not None:
        return

    if _require_prebuilt_envs():
        raise RuntimeError(_missing_prebuilt_env_message(pixi_env))

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
    print(f"# job: {inv.job.name}  (env={inv.job.env}, gpus={inv.gpus})", file=sys.stderr)
    print(f"# log: {inv.log_path}", file=sys.stderr)
    print(f"CUDA_VISIBLE_DEVICES={inv.gpus} {_shell_join(inv.argv)}")
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
            f"  - {inv.job.name}: env={inv.job.env}, gpus={inv.gpus}, log={inv.log_path}",
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
    proc: subprocess.Popen[bytes] | None = None
    thread: threading.Thread | None = None
    try:
        proc = subprocess.Popen(
            inv.argv,
            env=inv.env,
            cwd=str(_pixi_project_dir()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        if proc.stdout is None:
            raise RuntimeError(f"Job {inv.job.name!r} started without a stdout pipe")
        thread = threading.Thread(
            target=_tee,
            args=(inv.job.name, proc.stdout, log_file),
            daemon=True,
        )
        thread.start()
    except BaseException:
        log_file.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(
                    f"[{_ts()}] {inv.job.name} did not exit after failed spawn cleanup",
                    file=sys.stderr,
                )
        raise
    if proc is None or thread is None:
        raise RuntimeError(f"Job {inv.job.name!r} failed to initialize")
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
