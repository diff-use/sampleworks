"""
Utilities for evaluating grid search results.
All eval scripts should use these methods to avoid any deviations.
"""

import argparse
import re
from importlib.resources import files
from pathlib import Path

from loguru import logger
from sampleworks.eval.eval_dataclasses import Experiment, ExperimentList
from sampleworks.eval.occupancy_utils import extract_protein_and_occupancy
from sampleworks.utils.guidance_constants import StructurePredictor


# TODO: this either (both) needs tests or (and) there needs to be a clearer "API"
#  for how the folder names are generated.
def parse_experiment_dir(exp_dir: Path) -> dict[str, int | float | None]:
    """Parse experiment directory name to extract parameters.

    Handles both:
    - fk_steering format: ens{N}_gw{W}_gd{D}
    - pure_guidance format: ens{N}_gw{W}
    """
    dir_name = exp_dir.name
    logger.debug(f"Parsing experiment directory: {dir_name}")

    # Extract ensemble size
    ens_match = re.search(r"ens(\d+)", dir_name)
    ensemble_size = int(ens_match.group(1)) if ens_match else None

    # Extract guidance weight
    gw_match = re.search(r"gw([\d.]+)", dir_name)
    guidance_weight = float(gw_match.group(1)) if gw_match else None

    # Extract gradient descent steps (for fk_steering)
    gd_match = re.search(r"gd(\d+)", dir_name)
    gd_steps = int(gd_match.group(1)) if gd_match else None

    return {
        "ensemble_size": ensemble_size,
        "guidance_weight": guidance_weight,
        "gd_steps": gd_steps,
    }


# TODO: this method is now more flexible about how it scans the grid search results directory,
#  but that means we should be more strict about the output "API" directory structure.
def scan_grid_search_results(
    current_directory: Path,
    current_depth: int = 0,
    target_depth: int = 4,
    target_filename: str = "refined.cif",
) -> ExperimentList:
    """
    Recursively scan the grid_search_results directory for all experiments with refined.cif files.

    Parameters:
    - current_directory: (Path) Path to the current directory being scanned.
    - current_depth: (int) Current depth of the recursion, default 0
    - target_depth: (int) Depth where we expect to find experiment output files.
    - target_filename: (str) Name of the target file to look for, default "refined.cif"

    Returns:
    - List of dictionaries containing experiment metadata.
    """
    experiments = ExperimentList()

    if not current_directory.exists():
        if current_depth == 0:
            logger.error(
                f"Grid search directory not found: {current_directory} at depth {current_depth}"
            )
        return experiments

    # Check if we found a refined.cif file in the current directory
    refined_cif = current_directory / target_filename
    if current_depth == target_depth and refined_cif.exists():
        # Reconstruct metadata from path structure
        # Expected structure: .../protein_dir/model_dir/scaler_dir/exp_dir/refined.cif
        exp_dir = current_directory
        scaler_dir = exp_dir.parent
        model_dir = scaler_dir.parent
        protein_dir = model_dir.parent

        protein, occ_a = extract_protein_and_occupancy(protein_dir.name)
        method, model = get_method_and_model_name(model_dir.name)

        params = parse_experiment_dir(exp_dir)
        guidance_weight = float(params["guidance_weight"]) if params["guidance_weight"] else None
        gd_steps = int(params["gd_steps"]) if params["gd_steps"] else None

        # Validate parameters to satisfy pyright
        if (
            protein is None
            or occ_a is None
            or (model == StructurePredictor.BOLTZ_2 and method is None)
            or params["ensemble_size"] is None
            or (guidance_weight is None and gd_steps is None)
        ):
            logger.warning(f"Skipping experiment in {exp_dir} due to missing metadata")
            return experiments

        experiments.append(
            Experiment(
                protein=protein,
                occ_a=occ_a,
                model=model,
                method=method,
                scaler=scaler_dir.name,
                ensemble_size=int(params["ensemble_size"]),
                guidance_weight=guidance_weight,
                gd_steps=gd_steps,
                exp_dir=exp_dir,
                refined_cif_path=refined_cif,
                protein_dir_name=protein_dir.name,
            )
        )

        return experiments

    # Stop recursion if max depth reached, this should not happen, but it will prevent any
    # accidental infinite recursion if the directory structure changes in the future.
    if current_depth >= target_depth:
        return experiments

    # Recurse into subdirectories
    for item in current_directory.iterdir():
        if item.is_dir() and not item.name.endswith(".json"):
            experiments.extend(scan_grid_search_results(item, current_depth + 1, target_depth))

    return experiments


def get_method_and_model_name(model_name: str) -> tuple[str | None, str]:
    if "MD" in model_name:
        method = "MD"
        model = model_name.replace("_MD", "")
    elif "X-RAY" in model_name:
        method = "X-RAY"
        model = model_name.replace("_X-RAY_DIFFRACTION", "")
    else:
        method = None
        model = model_name
    return method, model


def parse_args(description: str | None = None):
    """
    Return a common set of arguments for grid search evaluation scripts,
    with a custom description, which is passed to argparse.ArgumentParser.

    All eval scripts should use this same framework
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="Path containing the grid search results directory, e.g. if results are "
        "at $HOME/grid_search_results, $HOME should be what you pass",
    )
    parser.add_argument(
        "--grid-search-inputs-path",
        type=Path,
        help="Path to the directory containing the grid search inputs, if it is different "
        "than the workspace root.",
        default=None,
    )
    parser.add_argument(
        "--protein-configs-csv",
        type=Path,
        help="Path to the CSV file containing protein configurations, like ${HOME}/configs.csv "
        "Defaults to sampleworks/data/protein_configs.csv",
        default=files("sampleworks.data") / "protein_configs.csv",
    )
    return parser.parse_args()
