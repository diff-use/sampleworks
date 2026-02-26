# NOTE: ty uses pyproject overrides (see [tool.ty.overrides] in pyproject.toml)
# for file-level suppression of optional attribute access diagnostics.
# We access a bunch of attributes from atoms in AtomArrays which by construction exist,
# but static analysis can't always prove that.
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, BadStructureError, index_distance
from biotite.structure.io.pdbx import CIFFile, get_structure
from loguru import logger
from peppr.bounds import get_distance_bounds
from sampleworks.eval.grid_search_eval_utils import parse_args, scan_grid_search_results
from scipy.special import comb
from tqdm import tqdm


# The following two methods, bond_length_violations and bond_angle_violations, are
# modified from https://github.com/aivant/peppr/blob/main/src/peppr/metric.py, MIT license
# at commit eae9c39
def bond_length_violations(pose: AtomArray, tolerance: float = 0.1) -> tuple[float, pd.DataFrame]:
    """
    Calculate the percentage of bonds that are outside acceptable ranges.

    Parameters
    ----------
    pose : AtomArray
        The structure to evaluate.
    tolerance : float, optional,
        relative tolerance for deviation from ideal values, by default 0.1

    Returns
    -------
    invalid fraction: float
        Percentage of bonds outside acceptable ranges (0.0 to 1.0).
    outlier_info : pd.DataFrame
        DataFrame containing information about the outliers, including atom indices and distances.
    """
    if pose.array_length() == 0:
        return np.nan, pd.DataFrame()

    try:
        bounds = get_distance_bounds(pose)  # this fetches values from RDKit
    except BadStructureError:
        return np.nan, pd.DataFrame()

    if not pose.bonds:
        logger.error(
            "Models must have bonds, use "
            "`biotite.structure.io.pdbx.get_structure(..., include_bonds=True)`"
        )
        return np.nan, pd.DataFrame()

    bond_indices = np.sort(pose.bonds.as_array()[:, :2], axis=1)
    if len(bond_indices) == 0:
        return np.nan, pd.DataFrame()

    bond_lengths = index_distance(pose, bond_indices)
    # The bounds matrix has the lower bounds in the lower triangle
    # and the upper bounds in the upper triangle
    lower_bounds = bounds[bond_indices[:, 1], bond_indices[:, 0]]
    upper_bounds = bounds[bond_indices[:, 0], bond_indices[:, 1]]
    invalid_mask = (bond_lengths < lower_bounds * (1 - tolerance)) | (
        bond_lengths > upper_bounds * (1 + tolerance)
    )

    outlier_info = []
    for bond, bond_length in zip(bond_indices[invalid_mask], bond_lengths[invalid_mask]):
        atom1 = pose[bond[0]]
        atom2 = pose[bond[1]]
        lower_bound = bounds[bond[1], bond[0]].item()
        upper_bound = bounds[bond[0], bond[1]].item()
        outlier_info.append(
            {
                "atom1_residue_id": atom1.res_id.item(),
                "atom1_residue_name": atom1.res_name.item(),
                "atom1_chain_id": atom1.chain_id.item(),
                "atom1_atom_id": atom1.atom_name.item(),
                "atom2_residue_id": atom2.res_id.item(),
                "atom2_residue_name": atom2.res_name.item(),
                "atom2_chain_id": atom2.chain_id.item(),
                "atom2_atom_id": atom2.atom_name.item(),
                "bond_length": bond_length.item(),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    # todo modify to return the actual list of violators

    invalid_fraction = float(
        np.count_nonzero(invalid_mask) / np.count_nonzero(np.isfinite(lower_bounds))
    )

    return invalid_fraction, pd.DataFrame(outlier_info)


def bond_angle_violations(pose: AtomArray, tolerance: float = 0.1) -> tuple[float, pd.DataFrame]:
    """
    Calculate the percentage of bonds that are outside acceptable ranges.
    Note that the method is actually to compute distance violations between two
    most commonly unbonded atoms. That is, given an atom A which is bonded to two atoms B and C,
    the method computes the distance between B and C and compares it to the ideal distance between
    B and C. If the distance is outside the acceptable range, the angle is implicitly an outlier.
    However, as a result, we don't report the bond angle, and it is possible that one or both of the
    bond lengths is an outlier, without the angle itself being an outlier.

    Parameters
    ----------
    pose : AtomArray
        The structure to evaluate.
    tolerance : float, optional,
        relative tolerance for deviation from ideal values, by default 0.1

    Returns
    -------
    invalid_fraction : float
        the fraction of bond angles outside the acceptable range (0.0 to 1.0)
    outlier_info : pd.DataFrame
        DataFrame containing information about the outliers, including atom indices and distances.
    """
    if pose.array_length() == 0:
        return np.nan, pd.DataFrame()

    try:
        bounds = get_distance_bounds(pose)
    except BadStructureError:
        return np.nan, pd.DataFrame()

    if not pose.bonds:
        logger.error(
            "Models must have bonds, use "
            "`biotite.structure.io.pdbx.get_structure(..., include_bonds=True)`"
        )
        return np.nan, pd.DataFrame()

    # in the original, bonds were fetched from the reference structure, but we don't have one here.
    all_bonds, _ = pose.bonds.get_all_bonds()
    # For a bond angle 'ABC', this list contains the atom indices for 'A' and 'C'
    bond_indices = []
    center_indices = []
    for center_index, bonded_indices in enumerate(all_bonds):
        # Remove padding values
        bonded_indices = bonded_indices[bonded_indices != -1]
        bond_indices.extend(itertools.combinations(bonded_indices, 2))
        center_indices.extend([center_index] * int(comb(len(bonded_indices), 2, exact=True)))

    if len(bond_indices) == 0:
        return np.nan, pd.DataFrame()

    bond_indices = np.sort(bond_indices, axis=1)
    center_indices = np.array(center_indices)

    bond_lengths = index_distance(pose, bond_indices)
    # The bounds matrix has the lower bounds in the lower triangle
    # and the upper bounds in the upper triangle
    # NOTE that the bounds matrix is computed s.t. there are bounds even for non-bonded atoms
    # so what we're really doing is checking if the atom-atom distance spanned by two bonds
    # with one atom in common (the open end of the triangle) is within the bounds
    lower_bounds = bounds[bond_indices[:, 1], bond_indices[:, 0]]
    upper_bounds = bounds[bond_indices[:, 0], bond_indices[:, 1]]
    invalid_mask = (bond_lengths < lower_bounds * (1 - tolerance)) | (
        bond_lengths > upper_bounds * (1 + tolerance)
    )

    outlier_info = []
    for bond, bond_length, center_index in zip(
        bond_indices[invalid_mask], bond_lengths[invalid_mask], center_indices[invalid_mask]
    ):
        center_atom = pose[center_index]
        atom1 = pose[bond[0]]
        atom2 = pose[bond[1]]
        lower_bound = bounds[bond[1], bond[0]].item()
        upper_bound = bounds[bond[0], bond[1]].item()
        outlier_info.append(
            {
                "atom1_residue_id": atom1.res_id.item(),
                "atom1_residue_name": atom1.res_name.item(),
                "atom1_chain_id": atom1.chain_id.item(),
                "atom1_atom_id": atom1.atom_name.item(),
                "atom2_residue_id": atom2.res_id.item(),
                "atom2_residue_name": atom2.res_name.item(),
                "atom2_chain_id": atom2.chain_id.item(),
                "atom2_atom_id": atom2.atom_name.item(),
                "center_atom_residue_id": center_atom.res_id.item(),
                "center_atom_residue_name": center_atom.res_name.item(),
                "center_atom_chain_id": center_atom.chain_id.item(),
                "center_atom_atom_id": center_atom.atom_name.item(),
                "bond_length": bond_length.item(),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    invalid_fraction = float(
        np.count_nonzero(invalid_mask) / np.count_nonzero(np.isfinite(lower_bounds))
    )
    return invalid_fraction, pd.DataFrame(outlier_info)


def main(args: argparse.Namespace):
    workspace_root = Path(args.workspace_root)
    # TODO make more general: https://github.com/diff-use/sampleworks/issues/93
    grid_search_dir = workspace_root / "grid_search_results"
    logger.info(f"Grid search directory: {grid_search_dir}")

    # Scan for experiments (look for refined.cif files)
    all_experiments = scan_grid_search_results(grid_search_dir)
    logger.info(f"Found {len(all_experiments)} experiments with refined.cif files")

    if all_experiments:
        all_experiments.summarize()  # Prints some summary stats, e.g. number of unique proteins
    else:
        logger.error("No experiments found in grid search directory. Exiting with status 1.")
        sys.exit(1)

    all_bond_length_outliers = []
    all_bond_angle_outliers = []
    all_bond_length_violation_fractions = []
    all_bond_angle_violation_fractions = []
    # TODO parallelize this with joblib
    for exp in tqdm(all_experiments):
        # get the refined cif file
        ciffile = CIFFile.read(exp.refined_cif_path)
        structures = get_structure(ciffile, include_bonds=True)
        structures = ensure_atom_array_stack(structures)
        for model_n, s in enumerate(structures):
            bond_angle_violation_fraction, bond_angle_outliers = bond_angle_violations(s)
            bond_length_violation_fraction, bond_length_outliers = bond_length_violations(s)
            bond_angle_outliers["model"] = str(exp.refined_cif_path)
            bond_angle_outliers["model_n"] = model_n
            bond_length_outliers["model"] = str(exp.refined_cif_path)
            bond_length_outliers["model_n"] = model_n

            all_bond_length_outliers.append(bond_length_outliers)
            all_bond_angle_outliers.append(bond_angle_outliers)

            all_bond_length_violation_fractions.append(
                {
                    "outlier_fraction": bond_length_violation_fraction,
                    "model": str(exp.refined_cif_path),
                    "model_n": model_n,
                }
            )
            all_bond_angle_violation_fractions.append(
                {
                    "outlier_fraction": bond_angle_violation_fraction,
                    "model": str(exp.refined_cif_path),
                    "model_n": model_n,
                }
            )

    pd.concat(all_bond_length_outliers).to_csv(
        grid_search_dir / "bond_length_outliers.csv", index=False
    )
    pd.concat(all_bond_angle_outliers).to_csv(
        grid_search_dir / "bond_angle_outliers.csv", index=False
    )
    pd.DataFrame(all_bond_length_violation_fractions).to_csv(
        grid_search_dir / "bond_length_violation_fractions.csv", index=False
    )
    pd.DataFrame(all_bond_angle_violation_fractions).to_csv(
        grid_search_dir / "bond_angle_violation_fractions.csv", index=False
    )


if __name__ == "__main__":
    args = parse_args("Evaluate bond angle and length outliers on grid search results.")
    main(args)
