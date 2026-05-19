"""Command-line entry point for ``sampleworks-runs``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import loader, runner
from .schema import Preset


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for name in loader.list_bundled_presets():
            print(name)
        return 0

    if args.preset is None:
        parser.error("PRESET is required (or pass --list)")

    preset = loader.load_preset(args.preset, overrides=args.set)
    if args.only:
        preset = _filter_only(preset, args.only)

    if args.show:
        _print_show(preset)
        return 0

    results_dir = Path(args.results_dir or _default_results_dir(preset))
    return runner.run(preset, results_dir=results_dir, dry_run=args.dry_run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sampleworks-runs",
        description=(
            "Run a preset of parallel run_grid_search.py jobs. "
            "Presets are TOML files bundled under sampleworks.runs.presets, "
            "or pass a path to a .toml file directly."
        ),
    )
    parser.add_argument("preset", nargs="?", help="Bundled preset name or path to a .toml file")
    parser.add_argument("--list", action="store_true", help="List bundled presets and exit")
    parser.add_argument("--show", action="store_true", help="Print the resolved preset and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pixi run commands instead of executing them",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated job names to run (subset). Default: all jobs.",
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


def _filter_only(preset: Preset, only: str) -> Preset:
    names = [n.strip() for n in only.split(",") if n.strip()]
    keep = [j for j in preset.jobs if j.name in names]
    missing = set(names) - {j.name for j in keep}
    if missing:
        raise SystemExit(f"--only references unknown jobs: {sorted(missing)}")
    return Preset(
        name=preset.name,
        description=preset.description,
        defaults=preset.defaults,
        shared_args=preset.shared_args,
        jobs=keep,
    )


def _print_show(preset: Preset) -> None:
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
    return (
        preset.defaults.get("RESULTS_DIR")
        or os.environ.get("RESULTS_DIR")
        or "./grid_search_results"
    )


if __name__ == "__main__":
    sys.exit(main())
