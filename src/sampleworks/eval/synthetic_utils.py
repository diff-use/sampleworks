"""Shared utilities for synthetic structure factor and density generation."""

import math
import traceback
from pathlib import Path

from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from loguru import logger
from sampleworks.eval.structure_utils import apply_selection
from sampleworks.utils.atom_array_utils import (
    AltlocInfo,
    detect_altlocs,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    remove_hydrogens,
)


def validate_occupancy_values(occupancy_values: list[float]) -> None:
    """Raise ValueError if any value is outside [0.0, 1.0] or values don't sum to 1.0."""
    for i, v in enumerate(occupancy_values):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Occupancy value {v} at index {i} is out of range [0.0, 1.0]")
    if occupancy_values and not math.isclose(sum(occupancy_values), 1.0):
        raise ValueError(f"Occupancy values must sum to 1.0, got {sum(occupancy_values)}")


def assign_occupancies(
    atom_array: AtomArray,
    altloc_info: AltlocInfo,
    occupancy_mode: str,
    occupancy_values: list[float] | None = None,
) -> AtomArray:
    """Assign occupancy values to atoms based on their altloc membership.

    Parameters
    ----------
    atom_array
        Structure to modify
    altloc_info
        Detected altloc information from detect_altlocs()
    occupancy_mode
        Assignment mode: 'default' (no change), 'uniform' (1/n_altlocs each),
        or 'custom' (user-specified values)
    occupancy_values
        For 'custom' mode: list of occupancy values [0.0-1.0] assigned to altlocs
        in sorted order (e.g., [0.3, 0.7] assigns 0.3 to altloc 'A', 0.7 to 'B').
        If fewer values than altlocs, remaining altlocs get occupancy 0.

    Returns
    -------
    AtomArray
        Modified structure with updated occupancies

    Raises
    ------
    ValueError
        If 'custom' mode is requested but no altlocs exist, or if occupancy_values
        is None in custom mode, or if any occupancy value is outside [0.0, 1.0]
    """
    if occupancy_mode == "default":
        return atom_array

    if not altloc_info.altloc_ids:
        if occupancy_mode == "custom":
            raise ValueError(
                "Custom occupancy mode was requested, but the structure has no altlocs."
            )
        logger.warning("No altlocs detected, using default occupancies")
        return atom_array

    result = atom_array.copy()
    occupancy = result.occupancy

    if occupancy_mode == "uniform":
        n_altlocs = len(altloc_info.altloc_ids)
        uniform_occ = 1.0 / n_altlocs
        for altloc in altloc_info.altloc_ids:
            occupancy[altloc_info.atom_masks[altloc]] = uniform_occ

    elif occupancy_mode == "custom":
        if occupancy_values is None:
            raise ValueError("occupancy_values required for custom mode")
        n_altlocs = len(altloc_info.altloc_ids)
        if len(occupancy_values) > n_altlocs:
            logger.warning(
                f"Got {len(occupancy_values)} occupancy values but structure "
                f"has {n_altlocs} altlocs. Extra values will be ignored."
            )
            occupancy_values = occupancy_values[:n_altlocs]
        elif len(occupancy_values) < n_altlocs:
            logger.warning(
                f"Got {len(occupancy_values)} occupancy values but structure "
                f"has {n_altlocs} altlocs. Missing values are automatically set to 0."
            )
            occupancy_values = occupancy_values + [0.0] * (n_altlocs - len(occupancy_values))
        validate_occupancy_values(occupancy_values)
        for altloc, occupancy_value in zip(sorted(altloc_info.altloc_ids), occupancy_values):
            occupancy[altloc_info.atom_masks[altloc]] = occupancy_value

    return result


def load_structure_for_synthetic_reward(
    structure_path: Path,
    occupancy_mode: str,
    occupancy_values: list[float],
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    selection: str | None = None,
) -> AtomArray | None:
    """Load and prepare a structure for synthetic reward generation.

    Handles loading, optional atom selection, stripping of unwanted atom classes,
    and occupancy assignment. Returns None on load or selection errors (logged);
    raises ValueError on invalid occupancy_mode or occupancy assignment errors (logged
    before raising).

    Parameters
    ----------
    structure_path
        Absolute path to the structure file
    occupancy_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform'
        (1/n_altlocs each), or 'custom' (use occupancy_values)
    occupancy_values
        Occupancy values for custom mode. If non-empty and occupancy_mode is not
        'custom', occupancy_mode is overridden with a warning.
    strip_hydrogens
        If True, remove hydrogen atoms
    strip_waters
        If True, remove water molecules
    strip_ligands
        If True, keep only polymer amino-acid atoms
    selection
        Optional atom selection string. If None, the full structure is used.

    Returns
    -------
    AtomArray | None
        Prepared structure, or None if loading or selection failed
    """
    if not structure_path.exists():
        logger.error(f"Structure not found: {structure_path}")
        return None

    try:
        atom_array = load_structure_with_altlocs(structure_path)
    except Exception as e:
        logger.error(
            f"Failed to load {structure_path} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return None

    try:
        atom_array = apply_selection(atom_array, selection)
    except ValueError as e:
        logger.error(f"Selection error for {structure_path}: {e}")
        return None

    # This is currently a sort of hacky way to remove ligands by keeping only polymer atoms
    # TODO: there's probably a more robust way to do this
    atom_array = remove_hydrogens(atom_array) if strip_hydrogens else atom_array
    atom_array = remove_waters(atom_array) if strip_waters else atom_array
    atom_array = keep_polymer(keep_amino_acids(atom_array)) if strip_ligands else atom_array
    assert isinstance(atom_array, AtomArray)
    altloc_info = detect_altlocs(atom_array)
    if occupancy_values:
        if occupancy_mode != "custom":
            logger.warning(
                f"Custom occupancy values provided for {structure_path}, "
                f"but occupancy_mode is '{occupancy_mode}'. Using 'custom' mode."
            )
            occupancy_mode = "custom"
        try:
            atom_array = assign_occupancies(
                atom_array, altloc_info, occupancy_mode, occupancy_values
            )
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {structure_path}: {e}")
            raise
    elif occupancy_mode in {"uniform", "default"}:
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, occupancy_mode)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {structure_path}: {e}")
            raise
    else:
        logger.error(f"Invalid occupancy mode '{occupancy_mode}' for {structure_path}")
        raise ValueError(f"Invalid occupancy mode '{occupancy_mode}'")

    return atom_array
