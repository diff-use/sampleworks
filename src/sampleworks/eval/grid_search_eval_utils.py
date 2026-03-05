"""
Utilities for evaluating grid search results.
All eval scripts should use these methods to avoid any deviations.
"""

import argparse
import re
import sys
from importlib.resources import files
from pathlib import Path

from loguru import logger
from sampleworks.eval.constants import OCCUPANCY_LEVELS
from sampleworks.eval.eval_dataclasses import Trial, TrialList, ProteinConfig
from sampleworks.eval.occupancy_utils import extract_protein_and_occupancy
from sampleworks.utils.guidance_constants import StructurePredictor


# TODO: this either (both) needs tests or (and) there needs to be a clearer "API"
#  for how the folder names are generated.
#  https://github.com/diff-use/sampleworks/issues/121
def parse_trial_dir(trial_dir: Path) -> dict[str, int | float | None]:
    """Parse trial directory name to extract parameters.

    Handles both:
    - fk_steering format: ens{N}_gw{W}_gd{D}
    - pure_guidance format: ens{N}_gw{W}
    """
    dir_name = trial_dir.name
    logger.debug(f"Parsing trial directory: {trial_dir}")

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
) -> TrialList:
    """Recursively scan the grid_search_results directory for all trial with refined.cif
    files.

    Parameters
    ----------
    current_directory : Path
        Path to the current directory being scanned.
    current_depth : int
        Current depth of the recursion, default 0.
    target_depth : int
        Depth where we expect to find trial output files.
    target_filename : str
        Name of the target file to look for, default "refined.cif"

    Returns
    -------
    TrialList
        List of trial metadata objects.
    """
    trials = TrialList()

    if not current_directory.exists():
        if current_depth == 0:
            logger.error(
                f"Grid search directory not found: {current_directory} at depth {current_depth}"
            )
        return trials

    # FIXME https://github.com/diff-use/sampleworks/issues/121
    # Check if we found a refined.cif file in the current directory
    refined_cif = current_directory / target_filename
    if current_depth == target_depth and refined_cif.exists():
        # Reconstruct metadata from path structure
        # Expected structure: .../protein_dir/model_dir/scaler_dir/trial_dir/refined.cif
        trial_dir = current_directory
        scaler_dir = trial_dir.parent
        model_dir = scaler_dir.parent
        protein_dir = model_dir.parent

        protein, altloc_occupancies = extract_protein_and_occupancy(protein_dir.name)
        method, model = get_method_and_model_name(model_dir.name)

        params = parse_trial_dir(trial_dir)
        guidance_weight = None
        if params["guidance_weight"] is not None:
            guidance_weight = float(params["guidance_weight"])
        gd_steps = int(params["gd_steps"]) if params["gd_steps"] is not None else None

        # Validate parameters to satisfy ty
        if (
            protein is None
            or not altloc_occupancies
            or (model == StructurePredictor.BOLTZ_2 and method is None)
            or params["ensemble_size"] is None
            or (guidance_weight is None and gd_steps is None)
        ):
            logger.warning(f"Skipping trial in {trial_dir} due to missing metadata")
            return trials

        trials.append(
            Trial(
                protein=protein,
                altloc_occupancies=altloc_occupancies,
                model=model,
                method=method,
                scaler=scaler_dir.name,
                ensemble_size=int(params["ensemble_size"]),
                guidance_weight=guidance_weight,
                gd_steps=gd_steps,
                trial_dir=trial_dir,
                refined_cif_path=refined_cif,
                protein_dir_name=protein_dir.name,
            )
        )

        return trials

    # Stop recursion if max depth reached, this should not happen, but it will prevent any
    # accidental infinite recursion if the directory structure changes in the future.
    if current_depth >= target_depth:
        return trials

    # Recurse into subdirectories
    for item in current_directory.iterdir():
        if item.is_dir() and not item.name.endswith(".json"):
            grid_search_trials = scan_grid_search_results(
                item, current_depth + 1, target_depth, target_filename=target_filename
            )
            trials.extend(grid_search_trials)

    return trials


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


def parse_eval_args(description: str | None = None):
    """
    Return a common set of arguments for grid search evaluation scripts,
    with a custom description, which is passed to argparse.ArgumentParser.

    All eval scripts should use this same framework
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--grid-search-results-path",
        type=Path,
        required=True,
        help="Path to the top-level grid search results directory, usu. called "
             "``grid_search_results``",
    )
    # not technically used everywhere yet, but requiring it future-proofs.
    parser.add_argument(
        "--grid-search-inputs-path",
        type=Path,
        required=True,
        help="Path to the directory containing the grid search inputs, in particular "
             "the protein configuration CSV file, maps, and reference structures.",
        default=None,
    )
    parser.add_argument(
        "--protein-configs-csv",
        type=Path,
        help="Path to the CSV file containing protein configurations, like "
             "``${HOME}/configs.csv``. Defaults to sampleworks/data/protein_configs.csv",
        default=files("sampleworks.data") / "protein_configs.csv",
    )
    parser.add_argument(
        "--occupancies",
        nargs="+",
        type=float,
        help=f"Occupancies to evaluate, defaults to {OCCUPANCY_LEVELS}",
        default=OCCUPANCY_LEVELS,
    )
    parser.add_argument(
        "--target-filename",
        default="refined.cif",
        help="Target filename for the CIF files to process, defaults to 'refined.cif'",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel jobs to run. -1 uses all CPUs.",
        default=16,
    )
    return parser.parse_args()


def setup_evaluation_parameters(
        args: argparse.Namespace
) -> tuple[ExperimentList, dict[str, ProteinConfig]]:
    grid_search_dir = Path(args.grid_search_results_path)

    # Protein configurations: base map paths, structure selections, and resolutions
    protein_inputs_dir = args.grid_search_inputs_path
    protein_configs = ProteinConfig.from_csv(protein_inputs_dir, args.protein_configs_csv)

    logger.info(f"Grid search directory: {grid_search_dir}")
    logger.info(f"Proteins configured: {list(protein_configs.keys())}")

    # Scan for experiments (look for refined.cif files)
    all_experiments = scan_grid_search_results(
        grid_search_dir, target_filename=args.target_filename
    )
    logger.info(f"Found {len(all_experiments)} experiments with refined.cif files")

    if all_experiments:
        all_experiments.summarize()  # Prints some summary stats, e.g. number of unique proteins
    else:
        logger.error("No experiments found in grid search directory. Exiting with status 1.")
        sys.exit(1)

    return all_experiments, protein_configs
