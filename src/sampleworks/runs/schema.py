"""Dataclasses for the run preset schema.

A preset describes one or more parallel script jobs. Experiment presets default
to ``run_grid_search.py`` while analysis presets set ``script`` explicitly to
one of the evaluation scripts. Each job runs in its configured pixi environment,
either through ``pixi run`` or a baked environment Python, with
``CUDA_VISIBLE_DEVICES`` set from an explicit GPU assignment or an automatically
allocated ``gpu_count``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_PIXI_ENVS = (
    "analysis",
    "analysis-dev",
    "boltz",
    "boltz-analysis",
    "boltz-dev",
    "protenix",
    "protenix-dev",
    "rf3",
    "rf3-dev",
)


@dataclass(frozen=True)
class Job:
    """One parallel script invocation within a preset.

    Parameters
    ----------
    name : str
        Identifier used for per-job log files and ``--jobs`` selection. Must be
        unique within the parent :class:`Preset`.
    env : str
        Pixi environment to run the job in. Must be one of
        :data:`VALID_PIXI_ENVS`.
    gpus : str
        Explicit value to set as ``CUDA_VISIBLE_DEVICES`` for the subprocess
        (e.g. ``"4"``, ``"0,1"``, or ``"none"`` for CPU-only jobs). Mutually
        exclusive with ``gpu_count``.
    gpu_count : int or None, optional
        Number of visible GPUs to auto-assign for this job. The runner assigns
        concrete GPU IDs in declaration order.
    output_subdir : str
        Path appended to the run's ``results_dir`` to form the job's
        output argument, when one is not given explicitly in ``args``. Also used
        as the directory the runner creates before launch.
    script : str, optional
        Script path to execute for this job. Relative paths are resolved against
        the active Sampleworks checkout. If empty, the runner uses its default
        experiment script.
    output_arg : str, optional
        CLI flag name that should receive ``results_dir / output_subdir`` when
        absent from ``args``. The default is ``"output-dir"`` for experiment
        runs. Set to ``""`` for scripts, such as analysis/eval scripts, that do
        not accept an output directory flag.
    args : dict of str to Any, optional
        Per-job overrides merged on top of the preset's
        :attr:`Preset.shared_args`. Keys are CLI flag names (without the
        leading ``--``); bools become bare flags (``True``) or omitted
        (``False``).

    Raises
    ------
    ValueError
        If ``env`` is not in :data:`VALID_PIXI_ENVS`, if neither/both ``gpus``
        and ``gpu_count`` are set, or if ``output_subdir`` is empty.
    """

    name: str
    env: str
    output_subdir: str
    gpus: str = ""
    gpu_count: int | None = None
    script: str = ""
    output_arg: str = "output-dir"
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ``env`` and required string fields."""
        if self.env not in VALID_PIXI_ENVS:
            raise ValueError(
                f"Job {self.name!r}: env must be one of {VALID_PIXI_ENVS}, got {self.env!r}"
            )
        if bool(self.gpus) == (self.gpu_count is not None):
            raise ValueError(f"Job {self.name!r}: set exactly one of gpus or gpu_count")
        if self.gpu_count is not None and self.gpu_count <= 0:
            raise ValueError(f"Job {self.name!r}: gpu_count must be positive")
        if not self.output_subdir:
            raise ValueError(f"Job {self.name!r}: output_subdir must be non-empty")
        if self.output_arg.startswith("--"):
            raise ValueError(
                f"Job {self.name!r}: output_arg must omit leading dashes, got {self.output_arg!r}"
            )


@dataclass(frozen=True)
class Preset:
    """A named bundle of parallel jobs orchestrated as a unit.

    Parameters
    ----------
    name : str
        Identifier (matches the experiment TOML filename without the ``.toml``
        suffix, or the stem of a user-supplied path).
    description : str
        Human-readable summary shown by ``--list`` and the launch banner.
    defaults : dict of str to str, optional
        Default values for ``${VAR}`` interpolation. The process environment
        takes precedence; this block only fills in unset keys.
    shared_args : dict of str to Any, optional
        Args merged into every job's ``args`` before argv is built. Per-job
        ``args`` win on collision.
    jobs : list of Job
        Jobs to launch in parallel. Must be non-empty and have unique names.

    Raises
    ------
    ValueError
        If ``jobs`` is empty or contains duplicate names.
    """

    name: str
    description: str
    defaults: dict[str, str] = field(default_factory=dict)
    shared_args: dict[str, Any] = field(default_factory=dict)
    jobs: list[Job] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the job list is non-empty and names are unique."""
        if not self.jobs:
            raise ValueError(f"Preset {self.name!r}: must declare at least one job")
        seen: set[str] = set()
        for job in self.jobs:
            if job.name in seen:
                raise ValueError(f"Preset {self.name!r}: duplicate job name {job.name!r}")
            seen.add(job.name)

    def job(self, name: str) -> Job:
        """Return the :class:`Job` with the given name.

        Parameters
        ----------
        name : str
            Job name to look up.

        Returns
        -------
        Job
            The matching job.

        Raises
        ------
        KeyError
            If no job has the given name.
        """
        for j in self.jobs:
            if j.name == name:
                return j
        raise KeyError(f"Preset {self.name!r} has no job {name!r}")

    def effective_args(self, job: Job) -> dict[str, Any]:
        """Merge :attr:`shared_args` with a job's per-job overrides.

        Parameters
        ----------
        job : Job
            Job whose ``args`` override the shared defaults.

        Returns
        -------
        dict of str to Any
            New dict; mutating it does not affect the preset.
        """
        return {**self.shared_args, **job.args}
