"""
# RSCC Analysis for Grid Search Results
# ported to a Python script by Marcus Collins marcus.collins@astera.org, from a notebook file
# provided by karson.chrispens@ucsf.edu

This script calculates the Real Space Correlation Coefficient (RSCC) between computed maps
from refined structures and reference (ground truth) maps for all experiments in the grid search results.

## Workflow:
1. Scan the `grid_search_results` directory for completed experiments
2. For each experiment with a `refined.cif`, compute the electron density map
3. Compare against the corresponding base map and calculate RSCC
4. Aggregate and visualize results by ensemble size, guidance weight, and scaler type
"""
import argparse
import copy
import re
import traceback

from biotite.structure import AtomArrayStack, AtomArray
from loguru import logger
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

# Import local modules for density calculation
from atomworks.io.parser import parse
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import setup_scattering_params
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import ATOMIC_NUM_TO_ELEMENT

from sampleworks.eval.constants import OCCUPANCY_LEVELS, DEFAULT_SELECTION_PADDING
from sampleworks.eval.eval_dataclasses import Experiment, ExperimentList, ProteinConfig
from sampleworks.eval.metrics import rscc
from sampleworks.eval.occupancy_utils import extract_protein_and_occupancy
from sampleworks.eval.structure_utils import get_asym_unit_from_structure, get_reference_structure


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
            logger.error(f"Grid search directory not found: {current_directory} at depth {current_depth}")
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

        experiments.append(
            Experiment(
                protein=protein,
                occ_a=occ_a,
                model=model,
                method=method,
                scaler=scaler_dir.name,
                ensemble_size=params["ensemble_size"],
                guidance_weight=params["guidance_weight"],
                gd_steps=params["gd_steps"],
                exp_dir=exp_dir,
                refined_cif_path=refined_cif,
                protein_dir_name=protein_dir.name,
            )
        )

        return experiments

    # Stop recursion if max depth reached, this should not happen, but it will prevent any accidental
    # infinite recursion if the directory structure changes in the future.
    if current_depth >= target_depth:
        return experiments

    # Recurse into subdirectories
    for item in current_directory.iterdir():
        if item.is_dir() and not item.name.endswith(".json"):
            experiments.extend(
                scan_grid_search_results(item, current_depth + 1, target_depth)
            )

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


def resize_to_ensemble(tensor: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    """Resize a tensor to the specified ensemble size by repeating the first dimension."""
    if tensor.ndim < 2:
        tensor = tensor.unsqueeze(0)
    # expand the first dimension to the ensemble size, all others remain the same
    return tensor.repeat(ensemble_size, *[1]*(tensor.ndim - 1))

    # Check if we have any valid coordinates left
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No valid (finite) coordinates after filtering NaN/Inf from "
            f"selection: '{selection}'"
        )

def compute_density_from_structure(structure: dict, xmap: XMap, device=None) -> np.ndarray:
    """
    Compute electron density from a structure dictionary.

    Parameters
    ----------
    structure : dict
        Atomworks parsed structure dictionary
    xmap : XMap
        Reference XMap for grid parameters
    device : torch.device, optional
        Device to use for computation

    Returns
    -------
    np.ndarray
        Computed electron density array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get atom array from the input structure and filter out positions with zero occupancy
    atom_array = get_asym_unit_from_structure(structure)
    # atom_array = atom_array[atom_array.occupancy > 0]

    # Set up scattering parameters
    scattering_params = setup_scattering_params(structure, em=False)

    # Create differentiable transformer
    xmap_torch = XMap_torch(xmap, device=device)
    transformer = DifferentiableTransformer(
        xmap=xmap_torch,
        scattering_params=scattering_params.to(device),
        em=False,
        device=device,
        use_cuda_kernels=torch.cuda.is_available(),
    )

    # get the ensemble size to weight each structure.
    ensemble_size = atom_array.shape[0] if len(atom_array.shape) < 3 else 1

    # Prepare input tensors
    # borrowed from another of @k.chrispens' scripts, not sure this actually works for AtomArray
    elements = torch.tensor(
        [ATOMIC_NUM_TO_ELEMENT.index(elem.title()) for elem in atom_array.element], device=device
    )
    coordinates = torch.from_numpy(atom_array.coord).float().to(device)
    b_factors = torch.from_numpy(atom_array.b_factor).float().to(device)

    # resize the tensors to the ensemble size
    elements = resize_to_ensemble(elements, ensemble_size)
    b_factors = resize_to_ensemble(b_factors, ensemble_size)
    # TODO: I need to think or not to use the original occupancies--they should be one anyway
    #  for a computed structure.
    occupancies = torch.ones_like(b_factors).float().to(device) / ensemble_size

    # Compute density--use the batch functionality to handle the ensemble.
    with torch.no_grad():
        density = transformer(
            coordinates=coordinates,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    if density.ndim == 4:
        density = density.sum(dim=0)

    return density.cpu().numpy().squeeze()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RSCC on grid search results.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="Path to the grid search results directory, like /home/kchrispens/sampleworks",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    workspace_root = Path(args.workspace_root)
    grid_search_dir = workspace_root / "grid_search_results"  # TODO make more general

    # TODO: need to put these in a config file.
    # Protein configurations: base map paths, structure selections, and resolutions
    protein_configs = {
        "1vme": ProteinConfig(
            protein="1vme",
            base_map_dir=workspace_root / "1vme_final_carved_edited",
            selection="chain A and resi 326-339",
            resolution=1.8,
            map_pattern="1vme_final_carved_edited_{occ_str}_1.80A.ccp4",
            structure_pattern="1vme_final_carved_edited_{occ_str}.cif",
        ),
        "4ole": ProteinConfig(
            protein="4ole",
            base_map_dir=workspace_root / "4ole_final_carved",
            selection="chain B and resi 60-67",
            resolution=2.52,
            map_pattern="4ole_final_carved_{occ_str}_2.52A.ccp4",
            structure_pattern="4ole_final_carved_{occ_str}.cif",
        ),
        "5sop": ProteinConfig(
            protein="5sop",
            base_map_dir=workspace_root / "5sop",
            selection="chain A and resi 129-135",
            resolution=1.05,
            map_pattern="5sop_{occ_str}_1.05A.ccp4",
            structure_pattern="5sop_{occ_str}.cif",
        ),
        "6b8x": ProteinConfig(
            protein="6b8x",
            base_map_dir=workspace_root / "6b8x",
            selection="chain A and resi 180-184",
            resolution=1.74,
            map_pattern="6b8x_{occ_str}_1.74A.ccp4",
            structure_pattern="6b8x_synthetic_{occ_str}.cif",
        )
    }

    logger.info(f"Grid search directory: {grid_search_dir}")
    logger.info(f"Proteins configured: {list(protein_configs.keys())}")

    # Test base map path resolution
    logger.debug("Testing base map path resolution:")
    for _, config in protein_configs.items():
        for _occ in OCCUPANCY_LEVELS:  # TODO make configurable
            _path = config.get_base_map_path_for_occupancy(_occ)  # will warn if not found
            if _path:
                logger.debug(f"  {config.protein} occ={_occ}: {_path}")


    # Scan for experiments (look for refined.cif files)
    all_experiments = scan_grid_search_results(grid_search_dir)
    logger.info(f"Found {len(all_experiments)} experiments with refined.cif files")

    if all_experiments:
        all_experiments.summarize()  # Prints some summary stats, e.g. number of unique proteins

    logger.info("Pre-loading reference structures for each protein for coordinate extraction")
    ref_coords = {}
    for protein_key, protein_config in protein_configs.items():
        protein_ref_coords = get_reference_structure(protein_config, protein_key)
        if protein_ref_coords is not None:
            ref_coords[protein_key] = protein_ref_coords

    # Calculate RSCC for all experiments
    # (BIG) TODO: implement a sliding-window version (global can be achieved with different selections.
    logger.info("Calculating RSCC values for all experiments...")
    logger.warning("Note: RSCC is computed on the region around altloc residues (defined by selection)")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    results = []
    base_map_cache: dict[tuple[str, float], tuple[XMap, XMap]] = {}
    for _i, _exp in enumerate(all_experiments):  # TODO parallelize this loop? It uses GPU, so be careful.
        if _exp.protein not in protein_configs:
            logger.warning(f"Skipping protein with no configuration: {_exp.protein}")
            continue

        protein_config = protein_configs[_exp.protein]

        # Check if we have reference coordinates for region extraction
        if _exp.protein not in ref_coords:
            logger.warning(
                f"Skipping {_exp.protein_dir_name}: no reference structure available for {_exp.protein}, "
                f"this may be due to a selection with zero atoms or NaN/Inf coordinates. Check logs above."
            )
            continue

        _selection_coords = ref_coords[_exp.protein]
        _base_map_path = protein_config.get_base_map_path_for_occupancy(_exp.occ_a)
        if _base_map_path is None:
            logger.warning(f"Skipping {_exp.protein_dir_name}: base map for occupancy {_exp.occ_a} not found")
            continue

        try:
            # TODO: we will reload these maps A LOT. Fix that by caching them somewhere?
            # Load base map for canonical unit cell, don't extract selection as we'll use the full map later too.
            if (_exp.protein, _exp.occ_a) not in base_map_cache:
                _base_xmap = protein_config.load_map(_base_map_path)

                # Extract the region around altloc residues from the base map
                _extracted_base = _base_xmap.extract(_selection_coords, padding=DEFAULT_SELECTION_PADDING)
                logger.info(f"Caching base and subselected maps for {_exp.protein} occ_a={_exp.occ_a}")
                base_map_cache[(_exp.protein, _exp.occ_a)] = (_base_xmap, _extracted_base)
            else:
                _base_map, _extracted_base = base_map_cache[(_exp.protein, _exp.occ_a)]


            # Validate extraction
            if _extracted_base.array.size == 0:
                raise ValueError(f"Extracted base map from {_base_map_path} is empty")

            # Load refined structure
            _structure = parse(_exp.refined_cif_path, ccd_mirror_path=None)

            # Compute density from refined structure
            _computed_density = compute_density_from_structure(_structure, _base_xmap, _device)

            # Create an XMap from the computed density by copying the base xmap
            # and replacing its array with the computed density
            _computed_xmap = copy.deepcopy(_base_xmap)
            _computed_xmap.array = _computed_density
            _extracted_computed = _computed_xmap.extract(_selection_coords, padding=DEFAULT_SELECTION_PADDING)

            # Validate extraction
            if _extracted_computed.array.size == 0:
                raise ValueError("Extracted computed map is empty")

            # Calculate RSCC on extracted regions
            _exp.rscc = rscc(_extracted_base.array, _extracted_computed.array)
            _exp.base_map_path = _base_map_path

        except Exception as _e:
            logger.error(f"ERROR processing {_exp.exp_dir}: {_e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            _exp.error = str(_e)
            _exp.rscc = np.nan  # this is the default, but better to be explicit.
            _exp.base_map_path = _base_map_path

        results.append(_exp)
        if (_i + 1) % 10 == 0 or _i == 0:
            logger.debug(
                f"  [{_i + 1}/{len(all_experiments)}] {_exp.protein_dir_name} / "
                f"{_exp.model} / {_exp.scaler} / ens{_exp.ensemble_size}_"
                f"gw{_exp.guidance_weight}: RSCC = {_exp.rscc:.4f}"
            )

    logger.info(f"\nCompleted RSCC calculation for {len(results)} experiments")

    # Create DataFrame from results
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(grid_search_dir / "rscc_results.csv", index=False)

    if not df.empty:
        # Remove error column for display if present
        drop_cols = [
            "exp_dir", "refined_cif_path", "base_map_path", "error", "protein_dir_name",
        ]

        logger.info("Results Summary:")
        logger.info(df.drop(drop_cols, axis=1).head(20).to_string())  # noqa

        logger.info("\n\nSummary Statistics by Protein and Scaler:")
        summary = (
            df.groupby(["protein", "scaler"])["rscc"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        logger.info(summary)

    # Calculate correlation between base maps and pure conformer maps
    logger.info("Calculating correlations between base maps and pure conformer maps...")
    logger.info("This shows how well single conformers explain occupancy-mixed data")

    base_pure_correlations = []

    for protein_key, protein_config in protein_configs.items():
        if protein_key not in ref_coords:
            print(f"Skipping {protein_key}: no reference coordinates available")
            continue

        # We re-use the selection coordinates from the reference structure computed at 0.5 occupancy above.
        _selection_coords = ref_coords[protein_key]

        logger.info(f"\nProcessing {protein_key} single conformer explanatory power:")
        map_path_1occA = protein_config.get_base_map_path_for_occupancy(1.0)
        map_path_1occB = protein_config.get_base_map_path_for_occupancy(0.0)
        if map_path_1occA is None or map_path_1occB is None:
            logger.warning(f"Skipping {protein_key}: pure conformer maps not found")
            continue
        try:
            # Load pure conformer maps--returns canonical unit cell by default, extract selection with padding 0.0
            _extracted_pure_A = protein_config.load_map(map_path_1occA, selection_coords=_selection_coords)
            _extracted_pure_B = protein_config.load_map(map_path_1occB, selection_coords=_selection_coords)

            logger.info(f"  Pure A reference: {map_path_1occA}\n  Pure B reference: {map_path_1occB}")

            # Calculate correlations for each occupancy
            for _occ_a in OCCUPANCY_LEVELS:  # TODO make configurable
                try:
                    _base_map_path = protein_config.get_base_map_path_for_occupancy(_occ_a)
                    if _base_map_path is None:  # map file not found, will warn.
                        continue

                    logger.info(f"  Processing occ_A={_occ_a}: {_base_map_path.name}")

                    # Load the base map for this occupancy, and do the selection--default padding is zero.
                    _extracted_base = protein_config.load_map(_base_map_path, selection_coords=_selection_coords)

                    # Calculate correlations
                    _corr_base_vs_pureA = rscc(_extracted_base.array, _extracted_pure_A.array)
                    _corr_base_vs_pureB = rscc(_extracted_base.array, _extracted_pure_B.array)

                    base_pure_correlations.append(
                        {
                            "protein": protein_key,
                            "occ_a": _occ_a,
                            "base_vs_1occA": _corr_base_vs_pureA,
                            "base_vs_1occB": _corr_base_vs_pureB,
                        }
                    )

                    logger.info(f"    Base map vs pure A: {_corr_base_vs_pureA:.4f}")
                    logger.info(f"    Base map vs pure B: {_corr_base_vs_pureB:.4f}")

                except Exception as _e:
                    logger.error(f"  Error processing occ_A={_occ_a} for {protein_key}: {_e}")
                    logger.error(f"  Traceback: {traceback.format_exc()}")

        except Exception as _e:
            logger.error(f"Error calculating correlations for {protein_key}: {_e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")

    df_base_vs_pure = pd.DataFrame(base_pure_correlations)
    df.to_csv(grid_search_dir / "rscc_results_for_pure_conformer_maps.csv", index=False)
    logger.info(
        f"\nCalculated single conformer explanatory power for "
        f"{len(df_base_vs_pure)} occupancy points"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
