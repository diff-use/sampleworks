"""Dataclasses for the preset schema.

A preset describes one or more parallel ``run_grid_search.py`` jobs. Each job
is launched as ``pixi run -e <env> python /app/run_grid_search.py <args>`` with
``CUDA_VISIBLE_DEVICES`` set to the job's GPU assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_PIXI_ENVS = ("boltz", "protenix", "rf3")


@dataclass
class Job:
    name: str
    env: str
    gpus: str
    output_subdir: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.env not in VALID_PIXI_ENVS:
            raise ValueError(
                f"Job {self.name!r}: env must be one of {VALID_PIXI_ENVS}, got {self.env!r}"
            )
        if not self.gpus:
            raise ValueError(f"Job {self.name!r}: gpus must be non-empty")
        if not self.output_subdir:
            raise ValueError(f"Job {self.name!r}: output_subdir must be non-empty")


@dataclass
class Preset:
    name: str
    description: str
    defaults: dict[str, str] = field(default_factory=dict)
    shared_args: dict[str, Any] = field(default_factory=dict)
    jobs: list[Job] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError(f"Preset {self.name!r}: must declare at least one job")
        seen: set[str] = set()
        for job in self.jobs:
            if job.name in seen:
                raise ValueError(f"Preset {self.name!r}: duplicate job name {job.name!r}")
            seen.add(job.name)

    def job(self, name: str) -> Job:
        for j in self.jobs:
            if j.name == name:
                return j
        raise KeyError(f"Preset {self.name!r} has no job {name!r}")

    def effective_args(self, job: Job) -> dict[str, Any]:
        """Return ``shared_args`` merged with per-job overrides."""
        return {**self.shared_args, **job.args}
