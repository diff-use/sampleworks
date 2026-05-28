"""Command-line entry point for ``sampleworks-runs``."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import loader, runner
from .schema import Job, Preset


DEFAULT_PRESET = "full_8gpu"
DEFAULT_PRESET_ALIASES = frozenset({"all", "full", "full_8gpu"})


@dataclass(frozen=True)
class CliConfig:
    """Configuration for one preset-driven runner CLI.

    Parameters
    ----------
    prog : str
        Program name shown by argparse.
    description : str
        Parser description.
    target_help : str
        Help text for the optional positional target.
    preset_help : str
        Help text for ``--preset``.
    list_help : str
        Help text for ``--list``.
    preset_dir_name : str
        Top-level directory containing TOML presets.
    preset_dir_env_var : str
        Environment variable that can override the preset directory.
    default_preset : str
        Preset to use when the user gives no target.
    default_aliases : frozenset of str
        Positional targets that resolve to ``default_preset``.
    results_default_keys : tuple of str
        Keys from ``preset.defaults`` to consult for the runner's results/log
        root.
    results_env_vars : tuple of str
        Environment variables to consult after ``results_default_keys``.
    results_fallback : str
        Final fallback results/log root.
    """

    prog: str
    description: str
    target_help: str
    preset_help: str
    list_help: str
    preset_dir_name: str
    preset_dir_env_var: str
    default_preset: str
    default_aliases: frozenset[str]
    results_default_keys: tuple[str, ...]
    results_env_vars: tuple[str, ...]
    results_fallback: str


EXPERIMENT_CLI_CONFIG = CliConfig(
    prog="sampleworks-runs",
    description=(
        "Run Sampleworks experiment presets. With no target, runs the full_8gpu preset. "
        "A target like 'rf3', 'boltz', or 'protenix' runs that preset; "
        "comma-separated targets like 'rf3,protenix' select jobs from full_8gpu."
    ),
    target_help=(
        "Preset name from experiments/ (rf3, boltz, protenix, etc.), comma-separated "
        "job shortcut from full_8gpu, or 'full'/'full_8gpu'."
    ),
    preset_help="Preset name from experiments/ or path to a .toml file. Default: full_8gpu.",
    list_help="List experiments/*.toml presets and exit",
    preset_dir_name="experiments",
    preset_dir_env_var="SAMPLEWORKS_EXPERIMENTS_DIR",
    default_preset=DEFAULT_PRESET,
    default_aliases=DEFAULT_PRESET_ALIASES,
    results_default_keys=("RESULTS_DIR",),
    results_env_vars=("RESULTS_DIR",),
    results_fallback="./grid_search_results",
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sampleworks-runs`` console script.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments excluding the program name. When ``None``
        (the default), :mod:`argparse` reads from :data:`sys.argv`.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``: ``0`` on success, non-zero on
        job failure or fatal CLI error.
    """
    return run_cli(argv, config=EXPERIMENT_CLI_CONFIG)


def run_cli(argv: list[str] | None = None, *, config: CliConfig) -> int:
    """Run a configured preset CLI.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments excluding the program name.
    config : CliConfig
        CLI behavior and preset-directory configuration.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """
    parser = _build_parser(config)
    args = parser.parse_args(argv)

    if args.list:
        for name in loader.list_presets(
            preset_dir_name=config.preset_dir_name,
            preset_dir_env_var=config.preset_dir_env_var,
        ):
            print(name)
        return 0

    preset_name, job_filter = _resolve_target(
        args.target,
        args.preset,
        args.jobs,
        parser,
        config=config,
    )
    preset = loader.load_preset(
        preset_name,
        overrides=args.set,
        preset_dir_name=config.preset_dir_name,
        preset_dir_env_var=config.preset_dir_env_var,
    )
    if job_filter:
        preset = _filter_jobs(preset, job_filter)

    if args.show:
        _print_show(preset)
        return 0

    results_dir = Path(args.results_dir or _default_results_dir(preset, config=config))
    try:
        return runner.run(preset, results_dir=results_dir, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser(config: CliConfig) -> argparse.ArgumentParser:
    """Construct the :mod:`argparse` parser for a configured runner CLI.

    Parameters
    ----------
    config : CliConfig
        CLI behavior and help-text configuration.

    Returns
    -------
    argparse.ArgumentParser
        Parser covering preset selection, overrides, and execution flags.
    """
    parser = argparse.ArgumentParser(prog=config.prog, description=config.description)
    parser.add_argument("target", nargs="?", help=config.target_help)
    parser.add_argument("--preset", default="", help=config.preset_help)
    parser.add_argument("--list", action="store_true", help=config.list_help)
    parser.add_argument("--show", action="store_true", help="Print the resolved preset and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved job commands instead of executing them",
    )
    parser.add_argument(
        "--jobs",
        default="",
        help="Comma-separated job names to run from the selected preset. Default: all jobs.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED_KEY=VALUE",
        help=(
            "Override a value in the loaded preset. Examples: "
            "--set defaults.DATA_DIR=/data/foo, "
            "--set jobs.rf3.args.gradient-weights='0.0 0.01', "
            "--set jobs.0.gpu_count=4"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override the runner results/log directory for this run.",
    )
    return parser


def _resolve_target(
    target: str | None,
    preset: str,
    jobs: str,
    parser: argparse.ArgumentParser,
    *,
    config: CliConfig = EXPERIMENT_CLI_CONFIG,
) -> tuple[str, str]:
    """Resolve the user-facing target grammar into preset plus job filter.

    Parameters
    ----------
    target : str or None
        Optional positional target. Without ``--preset`` this is either a
        default preset alias or a job selector from the configured default
        preset. With ``--preset`` it is a shorthand job selector for that
        explicit preset.
    preset : str
        Explicit preset name/path from ``--preset``.
    jobs : str
        Explicit comma-separated job selector from ``--jobs``.
    parser : argparse.ArgumentParser
        Parser used to report grammar errors.
    config : CliConfig, optional
        CLI behavior and preset-directory configuration.

    Returns
    -------
    tuple of str, str
        ``(preset_name_or_path, comma_separated_job_filter)``.
    """
    if preset:
        if target and jobs:
            parser.error("pass jobs either as the positional target or with --jobs, not both")
        return preset, jobs or target or ""

    if target is None or target in config.default_aliases:
        return config.default_preset, jobs

    if jobs:
        parser.error("pass jobs either as the positional target or with --jobs, not both")

    if target.endswith(".toml") or "/" in target:
        parser.error("pass custom preset paths with --preset path/to/preset.toml")

    if "," not in target and target in loader.list_presets(
        preset_dir_name=config.preset_dir_name,
        preset_dir_env_var=config.preset_dir_env_var,
    ):
        return target, ""

    return config.default_preset, target


def _filter_jobs(preset: Preset, jobs: str) -> Preset:
    """Return a new :class:`Preset` containing only the named jobs.

    Parameters
    ----------
    preset : Preset
        Source preset.
    jobs : str
        Comma-separated list of job names to keep.

    Returns
    -------
    Preset
        New preset with the same ``description``, ``defaults``, and
        ``shared_args`` and only the filtered jobs.

    Raises
    ------
    SystemExit
        If any name in ``jobs`` does not match a job in ``preset``.
    """
    names = [n.strip() for n in jobs.split(",") if n.strip()]
    keep = [j for j in preset.jobs if j.name in names]
    missing = set(names) - {j.name for j in keep}
    if missing:
        raise SystemExit(f"job selector references unknown jobs: {sorted(missing)}")
    description = f"Subset of {preset.name}: {', '.join(names)}"
    name = f"{preset.name}:{','.join(names)}"
    return Preset(
        name=name,
        description=description,
        defaults=preset.defaults,
        shared_args=preset.shared_args,
        pre_jobs=preset.pre_jobs,
        jobs=keep,
    )


def _print_show(preset: Preset) -> None:
    """Print a human-readable rendering of a resolved preset to stdout.

    Parameters
    ----------
    preset : Preset
        Resolved preset to display (used by ``--show``).
    """
    print(f"name: {preset.name}")
    if preset.description:
        print(f"description: {preset.description}")
    if preset.defaults:
        print("defaults:")
        for k, v in preset.defaults.items():
            print(f"  {k} = {v}")
    if preset.pre_jobs:
        print("pre_jobs:")
        for j in preset.pre_jobs:
            _print_job(j)
    print("jobs:")
    for j in preset.jobs:
        _print_job(j)


def _print_job(j: Job) -> None:
    """Print one resolved preset job for ``--show``.

    Parameters
    ----------
    j : Job
        Job to print.
    """
    print(f"  - name: {j.name}")
    print(f"    env: {j.env}")
    if j.gpus:
        print(f"    gpus: {j.gpus}")
    else:
        print(f"    gpu_count: {j.gpu_count}")
    if j.script:
        print(f"    script: {j.script}")
    print(f"    output_subdir: {j.output_subdir}")
    if j.output_arg != "output-dir":
        print(f"    output_arg: {j.output_arg!r}")
    print("    args:")
    for k, v in j.args.items():
        print(f"      {k} = {v!r}")


def _default_results_dir(preset: Preset, *, config: CliConfig = EXPERIMENT_CLI_CONFIG) -> str:
    """Pick a sensible default ``--results-dir`` when none is given.

    Parameters
    ----------
    preset : Preset
        Resolved preset (its ``defaults`` have already been merged with env).
    config : CliConfig, optional
        CLI behavior and result-directory fallback configuration.

    Returns
    -------
    str
        Path to use as the run's root output/log directory.
    """
    for key in config.results_default_keys:
        if preset.defaults.get(key):
            return preset.defaults[key]
    for env_var in config.results_env_vars:
        value = os.environ.get(env_var)
        if value:
            return value
    return config.results_fallback


if __name__ == "__main__":
    sys.exit(main())
