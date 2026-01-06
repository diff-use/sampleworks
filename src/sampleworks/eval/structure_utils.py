import re
import traceback
from typing import cast

import numpy as np
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from sampleworks.eval.eval_dataclasses import ProteinConfig


def parse_selection_string(selection: str) -> tuple[str | None, int | None, int | None]:
    """Parse a selection string like 'chain A and resi 326-339'.

    Parameters
    ----------
    selection : str Selection string

    Returns
    -------
    tuple (chain_id, resi_start, resi_end)
    """
    # Parse "chain X and resi N-M" format and generalizations of that.
    chain = re.search(r"chain\s+(\w+)", selection, re.IGNORECASE)
    if chain is not None:
        chain = chain.group(1).upper()

    residues = re.search(r"resi\s+(\d+)-(\d+)", selection, re.IGNORECASE)
    if residues is not None:
        resi_start = int(residues.group(1))
        resi_end = int(residues.group(2))
    else:
        resi_start = resi_end = None

    return chain, resi_start, resi_end


def extract_selection_coordinates(
    atom_array: AtomArray | AtomArrayStack, selection: str
) -> np.ndarray:
    """
    Extract coordinates for atoms matching a selection from an atomworks structure.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack Atomworks parsed structure
    selection : str Selection string like 'chain A and resi 326-339'

    Returns
    -------
    np.ndarray Coordinates of selected atoms, shape (n_atoms, 3)

    Raises
    ------
    RuntimeError: If no atoms match the selection or coordinates are invalid
    TypeError: If the "asym_unit" in `structure` is not an AtomArray or AtomArrayStack
    """
    # TODO: we will need to handle other kinds of selections later, like radius around a point.
    #   surely biotite has this capability?
    if isinstance(atom_array, AtomArrayStack):
        working_array = cast(AtomArray, atom_array[0])
    else:
        working_array = atom_array

    chain_id, resi_start, resi_end = parse_selection_string(selection)

    # Create the selection mask, don't rely on len(atom_array) in case it is the ensemble size
    mask = np.ones(len(working_array), dtype=bool)

    if chain_id is not None:
        mask &= working_array.chain_id == chain_id

    if resi_start is not None:
        res_ids = cast(np.ndarray, working_array.res_id)
        if resi_end is not None:
            # Explicitly check for None to satisfy pyright
            start: int = resi_start
            end: int = resi_end
            mask &= (res_ids >= start) & (res_ids <= end)
        else:
            start = resi_start
            mask &= res_ids == start

    selected_coords = cast(np.ndarray, working_array.coord)[mask]

    # VALIDATION
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No atoms matched selection: '{selection}'. "
            f"Chain ID: {chain_id}, Residue range: {resi_start}-{resi_end}. "
            f"Total atoms in structure: {len(atom_array)}"
        )

    # TODO? if there are missing atoms, what happens to the computed density?
    #  Do we need to be careful here?
    # Filter out atoms with NaN or Inf coordinates (common in alt conf structures)
    finite_mask = np.isfinite(selected_coords).all(axis=1)
    if not finite_mask.all():
        n_invalid = (~finite_mask).sum()
        n_total = len(selected_coords)
        logger.warning(
            f"Filtered {n_invalid} atoms with NaN/Inf coordinates from "
            f"selection '{selection}' ({n_total - n_invalid} valid atoms remaining)"
        )
        selected_coords = selected_coords[finite_mask]

    # Check if we have any valid coordinates left
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No valid (finite) coordinates after filtering NaN/Inf from selection: '{selection}'"
        )

    return selected_coords


def get_asym_unit_from_structure(
    structure: dict, atom_array_index: int | None = None
) -> AtomArray | AtomArrayStack:
    """
    Extract the AtomArray from a structure dictionary, handling AtomArrayStack if present.
    optionally specify the index of the AtomArray in the stack (e.g. for NMR models).
    """
    atom_array = structure["asym_unit"]
    if atom_array_index and isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array.get_array(atom_array_index)
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(f"Unexpected atom array type: {type(atom_array)}")
    return atom_array


# TODO: this assumes a single preconfigured selection of the structure--to generalize the
#  selection, we'll need to pull that out, or I guess modify it in the ProteinConfig instance
#  and then fetch the coordinates again?
# TODO: add option to override the selection string?
# TODO: rename--this really returns the coordinates, not a complete structure object.
def get_reference_structure(
    protein_config: ProteinConfig, protein_key: str, occ_list: tuple[float, ...] = (0.0, 1.0)
) -> np.ndarray | None:
    protein_ref_coords_list = []
    for occ in occ_list:
        ref_path = protein_config.get_reference_structure_path(occ)  # will warn if not found
        if ref_path:  # if not None, it is already a validated Path object
            try:
                ref_struct = load_any(ref_path, altloc="all")
                # TODO: enumerate actual exceptions this can raise.
                coords = extract_selection_coordinates(ref_struct, protein_config.selection)
                if not len(coords):
                    logger.warning(
                        f"  No atoms in selection '{protein_config.selection}' for {protein_key}"
                    )
                elif not np.isfinite(coords).all():
                    logger.warning(
                        f"  NaN/Inf coordinates in selection "
                        f"'{protein_config.selection}' for {protein_key}"
                    )
                else:
                    protein_ref_coords_list.append(coords)
                    logger.info(
                        f"  Loaded reference structure for {protein_key}: "
                        f"{len(coords)} atoms in selection '{protein_config.selection}'"
                    )
            except Exception as _e:
                _selection = protein_config.selection if protein_config.selection else "(none)"
                logger.error(
                    f"  ERROR: Failed to load reference structure for {protein_key}: {_e}\n"
                    f"    Path: {ref_path}\n"
                    f"    Selection: {_selection}\n"
                    f"    Traceback: {traceback.format_exc()}"
                )

    if not protein_ref_coords_list:
        logger.error(f"No reference structures found for {protein_key}")
        return None

    return np.vstack(protein_ref_coords_list)
