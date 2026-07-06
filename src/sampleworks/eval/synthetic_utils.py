"""Shared utilities for synthetic structure factor and density generation."""

import math
import traceback
from pathlib import Path

import gemmi
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from loguru import logger
from sampleworks.eval.structure_utils import apply_selection
from sampleworks.utils.atom_array_utils import (
    AltlocInfo,
    BLANK_ALTLOC_IDS,
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


def atomarray_to_gemmi(
    atom_array: AtomArray,
    unit_cell: gemmi.UnitCell | None = None,
    space_group: str | None = None,
) -> gemmi.Structure:
    """Convert a biotite AtomArray to a gemmi.Structure for SFcalculator.

    Anisotropic B-factors are set to zero since biotite does not store them.
    Blank altloc labels are converted from biotite's '' to gemmi's '\\x00'. If
    the atom array has no ``altloc_id`` annotation (e.g. arrays reconstructed by
    a model wrapper), all altlocs default to blank.

    Parameters
    ----------
    atom_array
        Input structure with occupancy and b_factor annotations
    unit_cell
        Crystallographic unit cell for the structure. If None, gemmi defaults
        to (1.0, 1.0, 1.0, 90.0, 90.0, 90.0) in units of Angstroms and degrees.
    space_group
        Space group (in Hermann-Mauguin string format) for the structure. If
        empty or invalid, SFcalculator defaults to P1.

    Returns
    -------
    gemmi.Structure
        Structure ready to be wrapped by SFC_Torch.io.PDBParser

    Notes
    -----
    Built directly rather than via ``SFC_Torch.io.array2hier``: that helper packs atom
    identity into a hyphen-delimited ``cra_name`` string, sets only the author seqid, and
    keys residue boundaries on ``res_id`` alone (merging residues that share a res_id across
    a chain boundary). Building the hierarchy here from the AtomArray annotations avoids the
    string round-trip and lets us set the author seqid, ``label_seq`` (so ``label_seq_id``
    survives a write -> atomworks reload; otherwise res_id collapses to -1), and ``subchain``
    (a <=1-char ``label_asym_id`` that SFcalculator's PDB-header step accepts) at construction.
    """
    n = len(atom_array)
    # altloc_id is not a mandatory biotite annotation; default to blank when absent.
    # gemmi uses '\x00' for blank altloc.
    if "altloc_id" in atom_array.get_annotation_categories():
        atom_altloc = ["\x00" if a in BLANK_ALTLOC_IDS else a for a in atom_array.altloc_id]
    else:
        atom_altloc = ["\x00"] * n

    model = gemmi.Model("1")  # numeric name -> valid mmCIF pdbx_PDB_model_num
    current_chain: gemmi.Chain | None = None
    current_res: gemmi.Residue | None = None
    prev_key: tuple[str, int] | None = None  # (chain_id, res_id) of the previous atom

    for i in range(n):
        chain_id = str(atom_array.chain_id[i])
        res_id = int(atom_array.res_id[i])
        key = (chain_id, res_id)

        if current_chain is None or chain_id != current_chain.name:
            if current_chain is not None:
                assert current_res is not None  # a chain always holds >=1 residue here
                current_chain.add_residue(current_res)
                model.add_chain(current_chain)
            current_chain = gemmi.Chain(chain_id)
            prev_key = None  # force a fresh residue for the new chain

        # New residue on any (chain_id, res_id) change. Keying on the pair (not res_id
        # alone) keeps residues that share a res_id across a chain boundary separate.
        if key != prev_key:
            if prev_key is not None:
                assert current_res is not None  # prev_key set implies a residue exists
                current_chain.add_residue(current_res)
            current_res = gemmi.Residue()
            current_res.name = str(atom_array.res_name[i])
            current_res.seqid = gemmi.SeqId(str(res_id))  # author numbering
            current_res.label_seq = res_id  # label numbering -> label_seq_id on write
            current_res.subchain = chain_id  # label_asym_id == chain name
            prev_key = key

        assert current_res is not None  # created on the first atom / every chain switch
        atom = gemmi.Atom()
        atom.name = str(atom_array.atom_name[i])
        atom.element = gemmi.Element(str(atom_array.element[i]))
        atom.pos = gemmi.Position(*atom_array.coord[i])  # coord[i]: (3,) float
        atom.b_iso = float(atom_array.b_factor[i])
        atom.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)  # biotite stores no anisotropic B
        atom.occ = float(atom_array.occupancy[i])
        atom.altloc = atom_altloc[i]
        current_res.add_atom(atom)

    if current_chain is not None:  # flush the trailing residue + chain
        assert current_res is not None  # non-empty input -> a residue was built
        current_chain.add_residue(current_res)
        model.add_chain(current_chain)

    structure = gemmi.Structure()
    structure.add_model(model)
    structure.setup_entities()  # SFcalculator/PDBParser expects entities assigned
    if unit_cell is not None:
        structure.cell = unit_cell
    if space_group is not None:
        structure.spacegroup_hm = space_group
    return structure
