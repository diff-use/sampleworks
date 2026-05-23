"""Command-line entry point for ``sampleworks-runs``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import loader, runner
from .schema import Preset


DEFAULT_PRESET = "full_8gpu"
DEFAULT_PRESET_ALIASES = frozenset({"all", "full", "full_8gpu"})


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
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for name in loader.list_presets():
            print(name)
        return 0

    preset_name, job_filter = _resolve_target(args.target, args.preset, args.jobs, parser)
    preset = loader.load_preset(preset_name, overrides=args.set)
    if job_filter:
        preset = _filter_jobs(preset, job_filter)

    if args.show:
        _print_show(preset)
        return 0

    results_dir = Path(args.results_dir or _default_results_dir(preset))
    try:
        return runner.run(preset, results_dir=results_dir, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    """Construct the :mod:`argparse` parser for ``sampleworks-runs``.

    Returns
    -------
    argparse.ArgumentParser
        Parser covering preset selection, overrides, and execution flags.
    """
    parser = argparse.ArgumentParser(
        prog="sampleworks-runs",
        description=(
            "Run Sampleworks experiment presets. With no target, runs the "
            "full_8gpu preset. A target like 'rf3', 'boltz', or 'protenix' "
            "runs that preset; comma-separated targets like 'rf3,protenix' "
            "select jobs from full_8gpu."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help=(
            "Preset name from experiments/ (rf3, boltz, protenix, etc.), "
            "comma-separated job shortcut from full_8gpu, or 'full'/'full_8gpu'."
        ),
    )
    parser.add_argument(
        "--preset",
        default="",
        help="Preset name from experiments/ or path to a .toml file. Default: full_8gpu.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments/*.toml presets and exit",
    )
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
            "--set jobs.0.gpus=5"
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override RESULTS_DIR for this run (also controls per-job log location).",
    )
    return parser


def _resolve_target(
    target: str | None,
    preset: str,
    jobs: str,
    parser: argparse.ArgumentParser,
) -> tuple[str, str]:
    """Resolve the user-facing target grammar into preset plus job filter.

    Parameters
    ----------
    target : str or None
        Optional positional target. Without ``--preset`` this is either a
        default preset alias (``full``/``full_8gpu``/``all``) or a job selector
        from :data:`DEFAULT_PRESET`. With ``--preset`` it is a shorthand job
        selector for that explicit preset.
    preset : str
        Explicit preset name/path from ``--preset``.
    jobs : str
        Explicit comma-separated job selector from ``--jobs``.
    parser : argparse.ArgumentParser
        Parser used to report grammar errors.

    Returns
    -------
    tuple of str, str
        ``(preset_name_or_path, comma_separated_job_filter)``.
    """
    if preset:
        if target and jobs:
            parser.error("pass jobs either as the positional target or with --jobs, not both")
        return preset, jobs or target or ""

    if target is None or target in DEFAULT_PRESET_ALIASES:
        return DEFAULT_PRESET, jobs

    if jobs:
        parser.error("pass jobs either as the positional target or with --jobs, not both")

    if target.endswith(".toml") or "/" in target:
        parser.error("pass custom preset paths with --preset path/to/preset.toml")

    if "," not in target and target in loader.list_presets():
        return target, ""

    return DEFAULT_PRESET, target


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
    print("jobs:")
    for j in preset.jobs:
        print(f"  - name: {j.name}")
        print(f"    env: {j.env}")
        print(f"    gpus: {j.gpus}")
        print(f"    output_subdir: {j.output_subdir}")
        print("    args:")
        for k, v in j.args.items():
            print(f"      {k} = {v!r}")


def _default_results_dir(preset: Preset) -> str:
    """Pick a sensible default ``--results-dir`` when none is given.

    Order of preference:
      1. The preset's ``[defaults]`` ``RESULTS_DIR``.
      2. The ``RESULTS_DIR`` environment variable.
      3. ``./grid_search_results``.

    Parameters
    ----------
    preset : Preset
        Resolved preset (its ``defaults`` have already been merged with env).

    Returns
    -------
    str
        Path to use as the run's root output directory.
    """
    return (
        preset.defaults.get("RESULTS_DIR")
        or os.environ.get("RESULTS_DIR")
        or "./grid_search_results"
    )


if __name__ == "__main__":
    sys.exit(main())
