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
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

# Import local modules for density calculation
from atomworks.io.parser import parse
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING
from sampleworks.eval.grid_search_eval_utils import parse_eval_args, setup_evaluation_parameters
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import (
    get_asym_unit_from_structure,
    get_reference_structure_coords,
)
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    remove_atoms_with_any_nan_coords,
)
from sampleworks.utils.density_utils import (
    build_density_transformer,
    run_density_transformer,
)
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch


OccKey = tuple[tuple[str, float], ...]
ProteinOccKey = tuple[str, OccKey]
SelectionKey = tuple[str, OccKey, str]


# TODO consolidate eval script logic: https://github.com/diff-use/sampleworks/issues/93
def main(args: argparse.Namespace):
    all_trials, protein_configs = setup_evaluation_parameters(args)

    logger.info("Pre-loading reference structures for each protein for coordinate extraction")
    ref_coords: dict[tuple[str, str], np.ndarray] = {}
    for protein_key, protein_config in protein_configs.items():
        # NOTE THAT THIS will be by default include all altlocs, as we use them to create a mask
        # for where to judge the maps' correlation.
        protein_ref_coords = get_reference_structure_coords(protein_config, protein_key)
        if protein_ref_coords is not None:
            for selection in protein_ref_coords.keys():
                ref_coords[(protein_key, selection)] = protein_ref_coords[selection]

    # Calculate RSCC for all trials
    logger.info("Calculating RSCC values for all trials...")
    logger.warning(
        "Note: RSCC is computed on the region around altloc residues (defined by selection)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Sort so all trials sharing a (protein, occ_key) are contiguous, which lets us reuse
    # loaded base maps, structures, and transformers.
    sorted_trials = sorted(all_trials, key=lambda t: (t.protein, t.occ_key))

    results: list[dict] = []
    base_map_cache: dict[ProteinOccKey, XMap] = {}
    transformer_cache: dict[ProteinOccKey, tuple[DifferentiableTransformer, XMap_torch]] = {}
    ref_full_structure_cache: dict[ProteinOccKey, AtomArray | AtomArrayStack] = {}
    extracted_base_cache: dict[SelectionKey, np.ndarray] = {}
    current_group: ProteinOccKey | None = None

    # TODO parallelize this loop? It uses GPU, so be careful.
    for i, trial in enumerate(sorted_trials):
        prev_result_count = len(results)
        if trial.protein in protein_configs:
            protein = trial.protein
        elif trial.protein.upper() in protein_configs:
            protein = trial.protein.upper()
        else:
            logger.warning(f"Skipping protein with no configuration: {trial.protein}")
            continue

        protein_config = protein_configs[protein]

        base_map_path = protein_config.get_base_map_path_for_occupancy(trial.altloc_occupancies)
        if base_map_path is None:
            logger.warning(
                f"Skipping {trial.protein_dir_name}: base map for occupancy "
                f"{trial.altloc_occupancies} not found"
            )
            continue

        has_valid_selection = False
        for selection in protein_config.selection:
            # Check if we have reference coordinates for region extraction
            if (protein, selection) not in ref_coords:
                logger.warning(
                    f"Skipping {trial.protein_dir_name}/{selection}: no reference structure "
                    f"available for {trial.protein}, this may be due to a selection with zero "
                    f"atoms or NaN/Inf coordinates. Check logs above."
                )
            else:
                has_valid_selection = True
        if not has_valid_selection:
            continue

        # clear caches when we move to a new (protein, occ_key) group
        group_key: ProteinOccKey = (protein, trial.occ_key)
        if current_group is not None and group_key != current_group:
            base_map_cache.clear()
            transformer_cache.clear()
            ref_full_structure_cache.clear()
            extracted_base_cache.clear()
        current_group = group_key

        # Load base map + transformer + reference once per (protein, occ_key)
        # parse refined, align, and compute density once per trial.
        # TODO: this needs to be better unified with what's in generate_synthetic_density
        #
        # Load base map for canonical unit cell,
        # don't overwrite the base map with selection map--we'll use the full map later too.
        try:
            base_xmap = base_map_cache.get(group_key)
            if base_xmap is None:
                base_xmap = protein_config.load_map(base_map_path)
                if base_xmap is None:
                    raise ValueError(f"Failed to load base map from {base_map_path}")
                base_map_cache[group_key] = base_xmap

            transformer_entry = transformer_cache.get(group_key)
            if transformer_entry is None:
                transformer, xmap_torch = build_density_transformer(
                    base_xmap, em_mode=False, device=device
                )
                transformer_cache[group_key] = (transformer, xmap_torch)
            else:
                transformer, _ = transformer_entry

            # Align the refined structure to the reference structure so that the calculated
            # maps are also aligned, for a correct RSCC calculation.
            # 1. Get the reference structure path and load from cache if available.
            ref_atom_array = ref_full_structure_cache.get(group_key)
            if ref_atom_array is None:
                ref_path = protein_config.get_reference_structure_path(trial.altloc_occupancies)
                if ref_path is None:
                    raise ValueError(
                        f"Could not find reference structure for "
                        f"occupancy {trial.altloc_occupancies}"
                    )
                # 2. Load the reference structure with parse() to get only the first altloc
                ref_structure = parse(ref_path, ccd_mirror_path=None)
                ref_atom_array = get_asym_unit_from_structure(ref_structure)
                ref_atom_array = remove_atoms_with_any_nan_coords(ref_atom_array)
                ref_full_structure_cache[group_key] = ref_atom_array

            structure = parse(trial.refined_cif_path, ccd_mirror_path=None)
            atom_array = get_asym_unit_from_structure(structure)
            if not hasattr(atom_array, "coord") or atom_array.coord is None:
                raise AttributeError("AtomArray | AtomArrayStack is missing coordinates")

            if not hasattr(atom_array, "b_factor"):
                logger.warning(
                    f"No b-factor array found in {trial.refined_cif_path}, setting to 20."
                )
                atom_array.set_annotation("b_factor", np.full(atom_array.coord.shape[-2], 20.0))

            atom_array = remove_atoms_with_any_nan_coords(atom_array)
            # 3. Find the common atoms with non-nan coords between the reference
            #    and the refined structure.
            ref_common, pred_common = filter_to_common_atoms(ref_atom_array, atom_array)

            # 4. Align the refined structure to the reference
            # using weighted_rigid_align_differentiable.
            # Convert to torch tensors with batch dimension.
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
                raise ValueError("ref_coords_torch and pred_coords_torch must have the same shape")

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
            computed_density = run_density_transformer(transformer, atom_array, device)
            # Shallow-copy the base xmap so .array can be rebound without touching the cache.
            # XMap.extract_tight reads self.array live, so the two wrappers stay independent.
            computed_xmap = copy.copy(base_xmap)
            computed_xmap.array = computed_density.cpu().numpy()
        except Exception as e:
            logger.error(f"ERROR processing trial {trial.trial_dir}: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            for selection in protein_config.selection:
                if (protein, selection) not in ref_coords:
                    continue
                row = asdict(trial)
                row.update(
                    selection=selection,
                    error=e,
                    rscc=np.nan,
                    base_map_path=base_map_path,
                )
                results.append(row)
            continue

        # Per selection, extract base region (cache) + computed region, compute RSCC
        for selection in protein_config.selection:
            if (protein, selection) not in ref_coords:
                continue
            sel_coords = ref_coords[(protein, selection)]
            row = asdict(trial)
            row.update(selection=selection, error=None, base_map_path=base_map_path)
            try:
                sel_key: SelectionKey = (protein, trial.occ_key, selection)
                extracted_base = extracted_base_cache.get(sel_key)
                if extracted_base is None:
                    _, extracted_base = base_xmap.extract_tight(
                        sel_coords, padding=DEFAULT_SELECTION_PADDING
                    )
                    if extracted_base is None or extracted_base.shape[0] == 0:
                        raise ValueError(f"Extracted base map empty for selection {selection}")
                    extracted_base_cache[sel_key] = extracted_base

                _, extracted_computed = computed_xmap.extract_tight(
                    sel_coords, padding=DEFAULT_SELECTION_PADDING
                )

                # Validate extraction
                if extracted_computed is None or extracted_computed.shape[0] == 0:
                    raise ValueError("Extracted computed map is empty")

                # Calculate RSCC on extracted regions
                row["rscc"] = rscc(extracted_base, extracted_computed)
            except Exception as e:
                logger.error(f"ERROR processing {trial.trial_dir} selection {selection}: {e}")
                row["error"] = e
                row["rscc"] = np.nan  # this is the default, but better to be explicit.
            results.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            if len(results) > prev_result_count:
                latest_result = results[-1]
                logger.debug(
                    f"  [{i + 1}/{len(sorted_trials)}] "
                    f"{latest_result.get('protein_dir_name', '?')} / "
                    f"{latest_result.get('model', '?')} / {latest_result.get('scaler', '?')} / "
                    f"ens{latest_result.get('ensemble_size', '?')}_gw"
                    f"{latest_result.get('guidance_weight', '?')}: "
                    f"RSCC = {latest_result.get('rscc', float('nan')):.4f}"
                )
            else:
                logger.debug(f"  [{i + 1}/{len(sorted_trials)}] did not add new result.")

    logger.info(f"\nCompleted RSCC calculation for {len(results)} trials")

    # Create DataFrame from results
    df = pd.DataFrame(results)
    df.to_csv(args.grid_search_results_path / "rscc_results.csv", index=False)

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
    args = parse_eval_args("Evaluate RSCC on grid search results.")
    main(args)
