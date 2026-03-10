"""
# RSCC Analysis for Grid Search Results
# ported to a Python script by Marcus Collins marcus.collins@astera.org, from a notebook file
# provided by karson.chrispens@ucsf.edu

This script calculates the Real Space Correlation Coefficient (RSCC) between computed maps
from refined structures and reference (ground truth) maps for all trials in the grid
search results.

## Workflow:
1. Scan the `grid_search_results` directory for completed trials
2. For each trial with a `refined.cif`, compute the electron density map
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
from biotite.structure import AtomArrayStack
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.eval.grid_search_eval_utils import parse_args, scan_grid_search_results
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import (
    get_asym_unit_from_structure,
    get_reference_structure_coords,
)
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    remove_atoms_with_any_nan_coords,
)
from sampleworks.utils.density_utils import compute_density_from_atomarray
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch


# TODO consolidate eval script logic: https://github.com/diff-use/sampleworks/issues/93
def main(args: argparse.Namespace):
    workspace_root = Path(args.workspace_root)
    grid_search_dir = workspace_root / "grid_search_results"

    # Protein configurations: base map paths, structure selections, and resolutions
    protein_inputs_dir = args.grid_search_inputs_path or workspace_root
    protein_configs = ProteinConfig.from_csv(protein_inputs_dir, args.protein_configs_csv)

    logger.info(f"Grid search directory: {grid_search_dir}")
    logger.info(f"Proteins configured: {list(protein_configs.keys())}")

    # Test base map path resolution
    logger.debug("Testing base map path resolution:")
    for _, config in protein_configs.items():
        for occ in args.occupancies:
            path = config.get_base_map_path_for_occupancy(occ)  # will warn if not found
            if path:
                logger.debug(f"  {config.protein} occ={occ}: {path}")

    # Scan for trials (look for refined.cif files)
    all_trials = scan_grid_search_results(grid_search_dir, target_filename=args.target_filename)
    logger.info(f"Found {len(all_trials)} trials with refined.cif files")

    if all_trials:
        all_trials.summarize()  # Prints some summary stats, e.g. number of unique proteins

    logger.info("Pre-loading reference structures for each protein for coordinate extraction")
    ref_coords = {}
    for protein_key, protein_config in protein_configs.items():
        # NOTE THAT THIS will be by default include all altlocs, as we use them to create a mask
        # for where to judge the maps' correlation.
        protein_ref_coords = get_reference_structure_coords(protein_config, protein_key)
        if protein_ref_coords is not None:
            for selection in protein_ref_coords.keys():
                ref_coords[(protein_key, selection)] = protein_ref_coords[selection]

    # Calculate RSCC for all trials
    # (BIG) TODO: implement a sliding-window version (global can be achieved with diff't selections.
    logger.info("Calculating RSCC values for all trials...")
    logger.warning(
        "Note: RSCC is computed on the region around altloc residues (defined by selection)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    results = []
    base_map_cache: dict[tuple[str, tuple[tuple[str, float], ...], str], tuple[XMap, XMap]] = {}
    ref_full_structure_cache: dict[tuple[str, tuple[tuple[str, float], ...]], AtomArrayStack] = {}
    # TODO parallelize this loop? It uses GPU, so be careful.
    for i, trial in enumerate(all_trials):
        prev_result_count = len(results)
        if trial.protein in protein_configs:
            protein = trial.protein
        elif trial.protein.upper() in protein_configs:
            protein = trial.protein.upper()
        else:
            logger.warning(f"Skipping protein with no configuration: {trial.protein}")
            continue

        protein_config = protein_configs[protein]
        for selection in protein_config.selection:
            # Check if we have reference coordinates for region extraction
            if (protein, selection) not in ref_coords:
                logger.warning(
                    f"Skipping {trial.protein_dir_name}/{selection}: no reference structure "
                    f"available for {trial.protein}, this may be due to a selection with zero "
                    f"atoms or NaN/Inf coordinates. Check logs above."
                )
                continue

            selection_coords = ref_coords[(protein, selection)]
            base_map_path = protein_config.get_base_map_path_for_occupancy(trial.altloc_occupancies)
            if base_map_path is None:
                logger.warning(
                    f"Skipping {trial.protein_dir_name}: base map for selection {selection} and "
                    f"occupancy {trial.altloc_occupancies} not found"
                )
                continue

            trial_result = trial.__dict__.copy()
            trial_result["selection"] = selection
            trial_result["error"] = None
            try:
                # TODO: this needs to be better unified with what's in generate_synthetic_density
                #
                # Load base map for canonical unit cell,
                # don't overwrite the base map with selection map--we'll use the full map later too.
                if (protein, trial.occ_key, selection) not in base_map_cache:
                    base_xmap = protein_config.load_map(base_map_path)
                    if base_xmap is None:
                        raise ValueError(f"Failed to load base map from {base_map_path}")

                    # Extract the region around altloc residues from the base map, using the
                    # union of boxes around each atom. extracted_base is no longer an XMap
                    _, extracted_base = base_xmap.extract_tight(
                        selection_coords, padding=DEFAULT_SELECTION_PADDING
                    )
                    logger.info(
                        f"Caching base and subselected maps for {protein} "
                        f"altloc_occupancies={trial.altloc_occupancies} selection={selection}"
                    )
                    base_map_cache[(protein, trial.occ_key, selection)] = (
                        base_xmap,
                        extracted_base,
                    )
                else:
                    base_xmap, extracted_base = base_map_cache[(protein, trial.occ_key, selection)]

                # Validate extraction
                if extracted_base is None or extracted_base.shape[0] == 0:
                    raise ValueError(f"Extracted base map from {base_map_path} is empty")

                # Load refined structure
                structure = parse(trial.refined_cif_path, ccd_mirror_path=None)

                # Compute density from refined structure
                atom_array = get_asym_unit_from_structure(structure)
                if not hasattr(atom_array, "coord") or atom_array.coord is None:
                    raise AttributeError("AtomArray | AtomArrayStack is missing coordinates")

                if not hasattr(atom_array, "b_factor"):
                    logger.warning(
                        f"No b-factor array found in {trial.refined_cif_path}, setting to 20."
                    )
                    atom_array.set_annotation("b_factor", np.full(atom_array.coord.shape[-2], 20.0))

                # Lines ~183-245 are to align the refined structure to the reference structure.
                # so that the calculated maps are also aligned, for a correct RSCC calculation
                #
                # Align the refined structure to the reference structure
                # 1. Get the reference structure path and load from cache if available
                if (protein, trial.occ_key) not in ref_full_structure_cache:
                    ref_path = protein_config.get_reference_structure_path(trial.altloc_occupancies)
                    if ref_path is None:
                        raise ValueError(
                            f"Could not find reference structure for "
                            f"occupancy {trial.altloc_occupancies}"
                        )

                    # 2. Load the reference structure with parse() to get only the first altloc
                    ref_structure = parse(ref_path, ccd_mirror_path=None)
                    ref_atom_array = get_asym_unit_from_structure(ref_structure)
                    logger.info(
                        f"Caching reference structure for {protein} "
                        f"altloc_occupancies={trial.altloc_occupancies}"
                    )
                    ref_full_structure_cache[(protein, trial.occ_key)] = ref_atom_array
                else:
                    ref_atom_array = ref_full_structure_cache[(protein, trial.occ_key)]

                # 3. Find the common atoms with non-nan coords between the reference
                #    and the refined structure
                ref_atom_array = remove_atoms_with_any_nan_coords(ref_atom_array)
                atom_array = remove_atoms_with_any_nan_coords(atom_array)
                ref_common, pred_common = filter_to_common_atoms(ref_atom_array, atom_array)

                # 4. Align the refined structure to the reference
                # using weighted_rigid_align_differentiable
                # Convert to torch tensors with batch dimension
                ref_coords_torch = torch.from_numpy(ref_common.coord).float()  # [1, n_atoms, 3]
                pred_coords_torch = torch.from_numpy(pred_common.coord).float()  # [1, n_atoms, 3]

                ref_coords_torch = match_batch(ref_coords_torch, pred_coords_torch.shape[0])
                if (
                    len(ref_coords_torch.shape) != 3
                    or ref_coords_torch.shape[1] != pred_coords_torch.shape[1]
                ):
                    logger.error(
                        f"Shape error: ref_coords_torch: {ref_coords_torch.shape}, "
                        f"pred_coords_torch: {pred_coords_torch.shape}"
                    )
                    raise ValueError(
                        "ref_coords_torch and pred_coords_torch must have the same shape"
                    )

                # Create uniform weights and mask for all common atoms
                n_atoms = ref_coords_torch.shape[1]
                weights = torch.ones(1, n_atoms)
                mask = torch.ones(1, n_atoms)

                # Align predicted to reference and get the transform
                _, transform = weighted_rigid_align_differentiable(
                    true_coords=pred_coords_torch,  # coords to align
                    pred_coords=ref_coords_torch,  # target coords
                    weights=weights,
                    mask=mask,
                    return_transforms=True,
                    allow_gradients=False,
                )

                # 5. Apply the transform to the entire refined structure (atom_array)
                atom_array_coords_torch = torch.from_numpy(atom_array.coord)
                aligned_coords_torch = apply_forward_transform(
                    atom_array_coords_torch, transform, rotation_only=False
                )
                atom_array.coord = aligned_coords_torch.numpy()

                # Compute density from the aligned refined structure
                computed_density, _ = compute_density_from_atomarray(
                    atom_array, xmap=base_xmap, em_mode=False, device=device
                )

                # Create an XMap from the computed density by copying the base xmap
                # and replacing its array with the computed density
                computed_xmap = copy.deepcopy(base_xmap)
                computed_xmap.array = computed_density.cpu().numpy().squeeze()
                _, extracted_computed = computed_xmap.extract_tight(
                    selection_coords, padding=DEFAULT_SELECTION_PADDING
                )

                # Validate extraction
                if extracted_computed is None or extracted_computed.shape[0] == 0:
                    raise ValueError("Extracted computed map is empty")

                # Calculate RSCC on extracted regions
                # TODO: don't alter the input object trial, just get a copy of it as a dict.
                trial_result["rscc"] = rscc(extracted_base, extracted_computed)
                trial_result["base_map_path"] = base_map_path

            except Exception as e:
                logger.error(f"ERROR processing {trial.trial_dir}: {e}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                trial_result["error"] = e
                trial_result["rscc"] = np.nan  # this is the default, but better to be explicit.
                trial_result["base_map_path"] = base_map_path

            results.append(trial_result)

        if (i + 1) % 10 == 0 or i == 0:
            if len(results) > prev_result_count:
                latest_result = results[-1]
                logger.debug(
                    f"  [{i + 1}/{len(all_trials)}] {latest_result.get('protein_dir_name', '?')} / "
                    f"{latest_result.get('model', '?')} / {latest_result.get('scaler', '?')} / "
                    f"ens{latest_result.get('ensemble_size', '?')}_gw"
                    f"{latest_result.get('guidance_weight', '?')}: "
                    f"RSCC = {latest_result.get('rscc', float('nan')):.4f}"
                )
            else:
                logger.debug(f"  [{i + 1}/{len(all_trials)}] did not add new result.")

    logger.info(f"\nCompleted RSCC calculation for {len(results)} trials")

    # Create DataFrame from results
    df = pd.DataFrame(results)
    df.to_csv(grid_search_dir / "rscc_results.csv", index=False)

    if not df.empty:
        # Remove error column for display if present
        drop_cols = [
            "trial_dir",
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


if __name__ == "__main__":
    args = parse_args("Evaluate RSCC on grid search results.")
    main(args)
