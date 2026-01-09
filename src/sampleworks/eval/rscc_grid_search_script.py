"""
# RSCC Analysis for Grid Search Results
# ported to a Python script by Marcus Collins marcus.collins@astera.org, from a notebook file
# provided by karson.chrispens@ucsf.edu

This script calculates the Real Space Correlation Coefficient (RSCC) between computed maps
from refined structures and reference (ground truth) maps for all experiments in the grid
search results.

## Workflow:
1. Scan the `grid_search_results` directory for completed experiments
2. For each experiment with a `refined.cif`, compute the electron density map
3. Compare against the corresponding base map and calculate RSCC
4. Aggregate and visualize results by ensemble size, guidance weight, and scaler type
"""

import argparse
import copy
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import local modules for density calculation
from atomworks.io.parser import parse
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING, OCCUPANCY_LEVELS
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.eval.grid_search_eval_utils import parse_args, scan_grid_search_results
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import (
    get_asym_unit_from_structure,
    get_reference_structure_coords,
)
from sampleworks.utils.density_utils import compute_density_from_atomarray


def main(args: argparse.Namespace):
    workspace_root = Path(args.workspace_root)
    grid_search_dir = workspace_root / "grid_search_results"  # TODO make more general

    # Protein configurations: base map paths, structure selections, and resolutions
    protein_inputs_dir = args.grid_search_inputs_path or workspace_root
    protein_configs = ProteinConfig.from_csv(protein_inputs_dir, args.protein_configs_csv)

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
        # NOTE THAT THIS will be by default _two_ structures, one computed for occupancy 0 and
        # one for occupancy 1 of altloc A. Historically this is because these coordinates are
        # used here only as a mask for map comparisons.
        # TODO: change that method to return the coordinates for occupancy 0 and 1 separately,
        #  and then we can merge them here.
        protein_ref_coords = get_reference_structure_coords(protein_config, protein_key)
        if protein_ref_coords is not None:
            ref_coords[protein_key] = protein_ref_coords

    # Calculate RSCC for all experiments
    # (BIG) TODO: implement a sliding-window version (global can be achieved with diff't selections.
    logger.info("Calculating RSCC values for all experiments...")
    logger.warning(
        "Note: RSCC is computed on the region around altloc residues (defined by selection)"
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    results = []
    base_map_cache: dict[tuple[str, float], tuple[XMap, XMap]] = {}
    # TODO parallelize this loop? It uses GPU, so be careful.
    for _i, _exp in enumerate(all_experiments):
        if _exp.protein not in protein_configs:
            logger.warning(f"Skipping protein with no configuration: {_exp.protein}")
            continue

        protein_config = protein_configs[_exp.protein]

        # Check if we have reference coordinates for region extraction
        if _exp.protein not in ref_coords:
            logger.warning(
                f"Skipping {_exp.protein_dir_name}: no reference structure available "
                f"for {_exp.protein}, this may be due to a selection with zero atoms "
                f"or NaN/Inf coordinates. Check logs above."
            )
            continue

        _selection_coords = ref_coords[_exp.protein]
        _base_map_path = protein_config.get_base_map_path_for_occupancy(_exp.occ_a)
        if _base_map_path is None:
            logger.warning(
                f"Skipping {_exp.protein_dir_name}: base map for occupancy {_exp.occ_a} not found"
            )
            continue

        try:
            # Load base map for canonical unit cell,
            # don't extract selection as we'll use the full map later too.
            if (_exp.protein, _exp.occ_a) not in base_map_cache:
                _base_xmap = protein_config.load_map(_base_map_path)
                if _base_xmap is None:
                    raise ValueError(f"Failed to load base map from {_base_map_path}")

                # Extract the region around altloc residues from the base map
                _extracted_base = _base_xmap.extract(
                    _selection_coords, padding=DEFAULT_SELECTION_PADDING
                )
                logger.info(
                    f"Caching base and subselected maps for {_exp.protein} occ_a={_exp.occ_a}"
                )
                base_map_cache[(_exp.protein, _exp.occ_a)] = (_base_xmap, _extracted_base)
            else:
                _base_xmap, _extracted_base = base_map_cache[(_exp.protein, _exp.occ_a)]

            # Validate extraction
            if _extracted_base is None or _extracted_base.array.size == 0:
                raise ValueError(f"Extracted base map from {_base_map_path} is empty")

            # Load refined structure
            _structure = parse(_exp.refined_cif_path, ccd_mirror_path=None)

            # Compute density from refined structure
            atom_array = get_asym_unit_from_structure(_structure)
            _computed_density, _ = compute_density_from_atomarray(
                atom_array, xmap=_base_xmap, em_mode=False, device=_device
            )

            # Create an XMap from the computed density by copying the base xmap
            # and replacing its array with the computed density
            _computed_xmap = copy.deepcopy(_base_xmap)
            _computed_xmap.array = _computed_density.cpu().numpy().squeeze()
            _extracted_computed = _computed_xmap.extract(
                _selection_coords, padding=DEFAULT_SELECTION_PADDING
            )

            # Validate extraction
            if _extracted_computed is None or _extracted_computed.array.size == 0:
                raise ValueError("Extracted computed map is empty")

            # Calculate RSCC on extracted regions
            _exp.rscc = rscc(_extracted_base.array, _extracted_computed.array)
            _exp.base_map_path = _base_map_path

        except Exception as _e:
            logger.error(f"ERROR processing {_exp.exp_dir}: {_e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            _exp.error = _e
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
            "exp_dir",
            "refined_cif_path",
            "base_map_path",
            "error",
            "protein_dir_name",
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

        # We re-use the selection coordinates from the reference structure computed
        # at 0.5 occupancy above.
        _selection_coords = ref_coords[protein_key]

        logger.info(f"\nProcessing {protein_key} single conformer explanatory power:")
        map_path_1occA = protein_config.get_base_map_path_for_occupancy(1.0)
        map_path_1occB = protein_config.get_base_map_path_for_occupancy(0.0)
        if map_path_1occA is None or map_path_1occB is None:
            logger.warning(f"Skipping {protein_key}: pure conformer maps not found")
            continue
        try:
            # Load pure conformer maps--returns canonical unit cell by default,
            # extract selection with padding 0.0
            _extracted_pure_A = protein_config.load_map(
                map_path_1occA, selection_coords=_selection_coords
            )
            _extracted_pure_B = protein_config.load_map(
                map_path_1occB, selection_coords=_selection_coords
            )

            logger.info(
                f"  Pure A reference: {map_path_1occA}\n  Pure B reference: {map_path_1occB}"
            )

            # Calculate correlations for each occupancy
            for _occ_a in OCCUPANCY_LEVELS:  # TODO make configurable
                try:
                    _base_map_path = protein_config.get_base_map_path_for_occupancy(_occ_a)
                    if _base_map_path is None:  # map file not found, will warn.
                        continue

                    logger.info(f"  Processing occ_A={_occ_a}: {_base_map_path.name}")

                    # Load the base map for this occupancy, and do the selection
                    # default padding is zero.
                    _extracted_base = protein_config.load_map(
                        _base_map_path, selection_coords=_selection_coords
                    )

                    if (
                        _extracted_base is None
                        or _extracted_pure_A is None
                        or _extracted_pure_B is None
                    ):
                        raise ValueError("One of the extracted maps is empty")

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
    args = parse_args("Evaluate RSCC on grid search results.")
    main(args)
