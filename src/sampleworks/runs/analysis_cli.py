"""Command-line entry point for ``sampleworks-analysis``."""

from __future__ import annotations

import sys

from sampleworks.runs.cli import CliConfig, run_cli


ANALYSIS_CLI_CONFIG = CliConfig(
    prog="sampleworks-analysis",
    description=(
        "Run Sampleworks analysis/evaluation presets. With no target, runs the "
        "analyze_grid_search preset. A target like 'all' runs that preset; "
        "comma-separated targets like 'rscc,lddt' select jobs from analyze_grid_search."
    ),
    target_help=(
        "Preset name from analyses/ (analyze_grid_search, all, external_tools, etc.) or "
        "comma-separated job shortcut from analyze_grid_search."
    ),
    preset_help="Preset name from analyses/ or path to a .toml file. Default: analyze_grid_search.",
    list_help="List analyses/*.toml presets and exit",
    preset_dir_name="analyses",
    preset_dir_env_var="SAMPLEWORKS_ANALYSES_DIR",
    default_preset="analyze_grid_search",
    default_aliases=frozenset({"analyze_grid_search", "grid_search"}),
    results_default_keys=("GRID_SEARCH_RESULTS_DIR", "ALTLOC_ANALYSIS_DIR", "RESULTS_DIR"),
    results_env_vars=("GRID_SEARCH_RESULTS_DIR", "ALTLOC_ANALYSIS_DIR", "RESULTS_DIR"),
    results_fallback="./grid_search_results",
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``sampleworks-analysis`` console script.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments excluding the program name. When ``None``
        (the default), :mod:`argparse` reads from :data:`sys.argv`.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """
    return run_cli(argv, config=ANALYSIS_CLI_CONFIG)


if __name__ == "__main__":
    sys.exit(main())
