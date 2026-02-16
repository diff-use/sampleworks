import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io.pdbx import CIFFile, set_structure
from loguru import logger


BACKBONE_ATOM_TYPES = ["C", "CA", "N", "O"]
BLANK_ALTLOC_IDS = {"", ".", " ", "?"}


@dataclass
class AltlocInfo:
    """Information about alternate conformations (altlocs) in a structure.

    Attributes
    ----------
    altloc_ids
        Sorted list of altloc identifiers (e.g., ['A', 'B'])
    atom_masks
        Dictionary mapping each altloc ID to a boolean mask indicating which atoms
        belong to that altloc
    """

    altloc_ids: list[str]
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]


def load_structure_with_altlocs(path: Path) -> AtomArray:
    """Load a structure file with alternate conformations and occupancy data.

    Takes the first model if multiple models are present.

    Parameters
    ----------
    path
        Path to the structure file (PDB, mmCIF, etc.)

    Returns
    -------
    AtomArray
        Loaded structure with occupancy and B-factor data
    """
    # Currently, we need to specify extra_fields=["occupancy"] to load altlocs properly
    atom_array = load_any(path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(atom_array, AtomArrayStack):
        atom_array = cast(AtomArray, atom_array[0])
    return cast(AtomArray, atom_array)


def save_structure_to_cif(
    atom_array: AtomArray | AtomArrayStack,
    output_path: str | Path,
    handle_nan: bool = True,
    frame_indices: int | list[int] | None = None,
) -> Path:
    """
    Save an AtomArray or AtomArrayStack to CIF file using biotite.

    Supports saving single-model or multimodel CIF files. If a stack is provided
    and multiple frames are selected, each frame becomes a separate model in the
    output CIF file.

    Parameters
    ----------
    atom_array: AtomArray | AtomArrayStack
        The structure to save.
    output_path: str | Path
        Output file path. Parent directories will be created if needed.
    handle_nan: bool, default=True
        If True, filters out atoms with NaN coordinates and warns.
        If False, passes structure as-is (may cause errors if NaN present).
    frame_indices: int | list[int] | None, default=None
        Which frames to save:
        - None: Save all frames (if AtomArrayStack) or single frame (if AtomArray)
        - int: Extract single frame at this index (saves single-model CIF)
        - list[int]: Extract multiple frames (saves multimodel CIF)

    Returns
    -------
    Path
        The path to the saved CIF file.

    Examples
    --------
    >>> from atomworks.io import parse
    >>> structure = parse("structure.pdb", ccd_mirror_path=None)
    >>> atom_array = structure["asym_unit"]
    >>>
    >>> save_structure_to_cif(atom_array, "output.cif")
    PosixPath('output.cif')
    >>>
    >>> save_structure_to_cif(
    ...     atom_array_stack,
    ...     "frame5.cif",
    ...     frame_indices=5
    ... )
    PosixPath('frame5.cif')
    >>>
    >>> save_structure_to_cif(
    ...     atom_array_stack,
    ...     "multimodel.cif",
    ...     frame_indices=[0, 10, 20, 30]
    ... )
    PosixPath('multimodel.cif')

    Notes
    -----
    - NaN coordinates will cause errors in downstream processing (e.g., CellList
      creation), so handle_nan=True is recommended.
    - When saving multimodel structures, all frames must have the same number of
      atoms and identical annotation arrays (chain_id, res_id, etc.).
    - Biotite's set_structure() automatically preserves all annotations present
      on the atom array. The extra_fields parameter is provided for API
      compatibility but annotations are saved by default.

    See Also
    --------
    biotite.structure.io.pdbx.set_structure: Underlying save function
    biotite.structure.stack: Create AtomArrayStack from multiple frames
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    structure_to_save: AtomArray | AtomArrayStack

    if frame_indices is not None:
        if not isinstance(atom_array, AtomArrayStack):
            raise ValueError("frame_indices can only be used with AtomArrayStack input")

        if isinstance(frame_indices, int):
            if frame_indices >= len(atom_array) or frame_indices < -len(atom_array):
                raise ValueError(
                    f"frame_indices {frame_indices} out of bounds for AtomArrayStack "
                    f"with {len(atom_array)} frames"
                )
            structure_to_save = cast(AtomArray, atom_array[frame_indices])
        else:
            for idx in frame_indices:
                if idx >= len(atom_array) or idx < -len(atom_array):
                    raise ValueError(
                        f"frame_indices contains {idx} which is out of bounds for "
                        f"AtomArrayStack with {len(atom_array)} frames"
                    )
            frames = [cast(AtomArray, atom_array[i]) for i in frame_indices]
            structure_to_save = stack(frames)
    else:
        structure_to_save = atom_array

    if handle_nan:
        if isinstance(structure_to_save, AtomArrayStack):
            coords: Any = structure_to_save.coord
            nan_mask = np.isnan(coords).any(axis=-1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                warnings.warn(
                    f"Found {n_nan} atom positions with NaN coordinates "
                    "across all frames. Filtering them out before saving.",
                    UserWarning,
                    stacklevel=2,
                )
                clean_mask = ~nan_mask.any(axis=0)
                structure_to_save = cast(AtomArrayStack, structure_to_save[:, clean_mask])
        else:
            coords: Any = structure_to_save.coord
            nan_mask = np.isnan(coords).any(axis=-1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                warnings.warn(
                    f"Found {n_nan} atoms with NaN coordinates. Filtering them out before saving.",
                    UserWarning,
                    stacklevel=2,
                )
                structure_to_save = cast(AtomArray, structure_to_save[~nan_mask])

    cif_file = CIFFile()
    set_structure(cif_file, structure_to_save)
    cif_file.write(str(output_path))

    return output_path


def find_all_altloc_ids(
        atom_array: AtomArray | AtomArrayStack, altloc_label: str = "altloc_id"
) -> set[str]:
    """
    Find all unique alternate location indicator (altloc) IDs in an AtomArray or AtomArrayStack.
    """
    if hasattr(atom_array, altloc_label):
        altloc_ids = np.unique(getattr(atom_array, altloc_label))  # pyright: ignore (reportArgumentType)
    else:
        raise AttributeError("atom_array must have altloc annotation, defaults to 'altloc_id'; "
                             f"you provided {altloc_label}")

    return set(altloc_ids.tolist()) - BLANK_ALTLOC_IDS


def detect_altlocs(atom_array: AtomArray) -> AltlocInfo:
    """Detect alternate conformations in a structure.

    Identifies all non-default altloc IDs and creates boolean masks for each.
    Default values ("", ".", " ") are excluded from the detected altlocs.

    Parameters
    ----------
    atom_array
        The input AtomArray containing altloc_id annotations.

    Returns
    -------
    AltlocInfo
        Dataclass containing:
        - altloc_ids: Sorted list of detected altloc IDs
        - atom_masks: Dictionary mapping each altloc ID to a boolean mask of atoms

    """
    # find_all_altloc_ids has error checking
    try:
        altloc_ids = sorted(find_all_altloc_ids(atom_array))
    except AttributeError:
        logger.warning("Structure has no altloc_id annotation; assuming no alternate locations")
        return AltlocInfo(altloc_ids=[], atom_masks={})

    altloc_arr = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.altloc_id)
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {}
    for altloc in altloc_ids:
        atom_masks[altloc] = altloc_arr == altloc

    return AltlocInfo(altloc_ids=altloc_ids, atom_masks=atom_masks)


def map_altlocs_to_stack(
    atom_array: AtomArray | AtomArrayStack,
    selection: str | None = None,
    return_full_array: bool = True
) -> tuple[AtomArrayStack, np.ndarray, np.ndarray]:
    """
    Map alternate location indicators (altloc) to separate structures in a new AtomArrayStack.

    Note: This will raise an error if you pass an AtomArrayStack containing multiple structures

    Parameters:
        atom_array: AtomArray or AtomArrayStack to map altlocs from.
        selection: str
            Optional selection string to apply to the atom array before mapping altlocs.
            If the return_full_array is also True, this selection applies _only_ to the altlocs.
        return_full_array: bool
            If True, return the full AtomArrayStack with all atoms, even those with no altlocs.
            If False, only return structures with altlocs.

    Returns:
        Tuple containing:
            - AtomArrayStack: The new stack with separate structures for each altloc.
            - np.ndarray: Array of altloc IDs, corresponding to the order
                 of the structures in the stack.
            - np.ndarray: Array of occupancies for each atom in each stack,
                 corresponding to the order of structures in the stack
    """
    if isinstance(atom_array, AtomArrayStack):
        if len(atom_array) > 1:
            raise ValueError("Cannot map altlocs with multiple structures each containing altlocs")
        atom_array = atom_array[0]  # pyright: ignore (reportAssignmentType)

    altloc_ids = sorted(list(find_all_altloc_ids(atom_array)))
    if selection is not None and return_full_array:
        # in this case, we select atoms with no altlocs, and atoms in the selection that do have altlocs
        # construct the query (note np.isin doesn't seem to list sets, hence conversion to list.
        no_altloc_mask = np.isin(atom_array.altloc_id, list(BLANK_ALTLOC_IDS))
        altloc_mask = np.isin(atom_array.altloc_id, altloc_ids)
        selection_mask = atom_array.mask(selection)
        mask = np.logical_or(no_altloc_mask, np.logical_and(altloc_mask, selection_mask))
        atom_array = atom_array[mask]
    elif selection is not None:
        atom_array = atom_array.query(selection)

    # The available altloc ids might have changed if one or more are missing from the selection.
    altloc_ids = sorted(list(find_all_altloc_ids(atom_array)))
    altloc_list = [
        select_altloc(atom_array, altloc_id, return_full_array=return_full_array) for altloc_id in altloc_ids
    ]
    # ensure that each structure has the same number of atoms
    # it's critical that we've updated the altloc id list above, or this will return only
    # residues with no altlocs in case one or more altloc ids is/are missing from the selection.
    atom_arrays = filter_to_common_atoms(*altloc_list)
    altloc_ids = np.vstack([r.altloc_id for r in atom_arrays])  # pyright: ignore
    occupancies = np.vstack([r.occupancy for r in atom_arrays])  # pyright: ignore

    # remove those annotations or we cannot stack arrays.
    for array in atom_arrays:
        array.del_annotation("occupancy")
        array.del_annotation("altloc_id")

    # filter_to_common_atoms() returns a tuple of AtomArrayStack, so we need to take the first
    # (there will only be one structure in each)
    output_atom_array_stack = stack([a[0] for a in atom_arrays])

    return output_atom_array_stack, altloc_ids, occupancies


def select_altloc(
    atom_array: AtomArray | AtomArrayStack, altloc_id: str, return_full_array: bool = False
) -> AtomArray | AtomArrayStack:
    """Select atoms with a specific alternate location indicator (altloc).

    Parameters:
        atom_array (AtomArray | AtomArrayStack): The input atom array.
        altloc_id (str): The alternate location indicator to filter by.
        return_full_array (bool, optional):
            If True, return the full atom array with, selecting atoms with either the
            specified altloc or an empty/default altloc.

    Returns:
        AtomArray: A new atom array containing only atoms with the specified altloc.
        if return_full_array is False, AtomArrayStack:
            A new atom array stack containing only atoms with the specified altloc.
        if return_full_array is True, AtomArrayStack:
            A new atom array stack containing atoms with either the specified altloc or
            an empty/default altloc (period, space, ?, or empty string). It is possible that
            an alternate altloc may be, e.g.,  disordered and therefore missing from the structure.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    if not (hasattr(atom_array, "altloc_id") and hasattr(atom_array, "occupancy")):
        raise AttributeError("atom_array must have `altloc_id` and `occupancy` annotations")

    if return_full_array:
        mask = np.isin(
            atom_array.altloc_id,  # pyright: ignore (reportArgumentType)
            list({altloc_id,}.union(BLANK_ALTLOC_IDS)),
        )
    else:
        mask = atom_array.altloc_id == altloc_id

    if isinstance(atom_array, AtomArrayStack):
        return cast(AtomArrayStack, atom_array[:, mask])
    else:
        return cast(AtomArray, atom_array[mask])


def select_non_hetero(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Select atoms with the hetero flag set to False.

    The hetero flag in biotite indicates whether an atom is a heteroatom
    (typically ligands, waters, or other non-standard residues). This function
    filters to return only standard protein/nucleic acid atoms.

    Parameters:
        atom_array (AtomArray | AtomArrayStack): The input atom array or stack.

    Returns:
        AtomArray | AtomArrayStack: A new array/stack with hetero=False atoms.

    Raises:
        TypeError: If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    hetero = atom_array.hetero
    if hetero is None:
        raise AttributeError("atom_array must have 'hetero' annotation")
    mask = ~hetero

    if isinstance(atom_array, AtomArrayStack):
        return cast(AtomArrayStack, atom_array[:, mask])
    else:
        return cast(AtomArray, atom_array[mask])


def remove_hydrogens(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Remove hydrogen atoms from an AtomArray or AtomArrayStack.

    Filters out all atoms where the element annotation is "H" (hydrogen).

    Parameters:
        atom_array (AtomArray | AtomArrayStack): The input atom array or stack.

    Returns:
        AtomArray | AtomArrayStack: A new array/stack with hydrogen atoms removed.

    Raises:
        TypeError: If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    element = atom_array.element
    if element is None:
        raise AttributeError("atom_array must have 'element' annotation")
    mask = element != "H"

    if isinstance(atom_array, AtomArrayStack):
        return cast(AtomArrayStack, atom_array[:, mask])
    else:
        return cast(AtomArray, atom_array[mask])


def select_backbone(atom_array: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
    """Select only backbone atoms from an AtomArray or AtomArrayStack.

    Returns atoms with atom_name in ["C", "CA", "N", "O"], which are the
    standard protein backbone atoms.

    Parameters:
        atom_array (AtomArray | AtomArrayStack): The input atom array or stack.

    Returns:
        AtomArray | AtomArrayStack: A new array/stack containing only backbone atoms.

    Raises:
        TypeError: If input is not an AtomArray or AtomArrayStack.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    backbone_atoms = np.array(BACKBONE_ATOM_TYPES)
    atom_name = atom_array.atom_name
    if atom_name is None:
        raise AttributeError("atom_array must have 'atom_name' annotation")
    mask = np.isin(atom_name, backbone_atoms)

    if isinstance(atom_array, AtomArrayStack):
        return cast(AtomArrayStack, atom_array[:, mask])
    else:
        return cast(AtomArray, atom_array[mask])


def make_atom_id(arr: AtomArray | AtomArrayStack) -> np.ndarray:
    """Create a unique identifier for each atom."""
    chain_id = cast(np.ndarray, arr.chain_id)
    res_id = cast(np.ndarray, arr.res_id)
    atom_name = cast(np.ndarray, arr.atom_name)
    return np.array(
        [f"{chain}_{res}_{atom}" for chain, res, atom in zip(chain_id, res_id, atom_name)]
    )


def filter_to_common_atoms(
    *arrays: AtomArray | AtomArrayStack
) -> tuple[AtomArrayStack, ...]:
    """Filter multiple AtomArrays/AtomArrayStacks to only include common atoms.

    Creates unique identifiers for each atom based on chain_id, res_id, and
    atom_name, then filters all structures to include only atoms present in all.
    The returned arrays are sorted to ensure atoms are in matching order.

    Parameters:
        *arrays (AtomArray | AtomArrayStack): Two or more atom arrays or stacks
            to filter.

    Returns:
        tuple[AtomArray | AtomArrayStack, ...]:
            Filtered versions of all input arrays containing only common atoms
            in matching order. The tuple length matches the number of inputs.

    Raises:
        TypeError: If inputs are not AtomArray or AtomArrayStack.
        ValueError: If fewer than two arrays are provided.
        RuntimeError: If no common atoms are found across all structures.

    Examples:
        >>> # Two arrays (original behavior)
        >>> array1 = AtomArray(...)  # 100 atoms
        >>> array2 = AtomArray(...)  # 95 atoms, 90 overlap with array1
        >>> filtered1, filtered2 = filter_to_common_atoms(array1, array2)
        >>> len(filtered1)  # 90
        >>> len(filtered2)  # 90

        >>> # Three or more arrays
        >>> arr1 = AtomArray(...)  # 100 atoms
        >>> arr2 = AtomArray(...)  # 95 atoms
        >>> arr3 = AtomArray(...)  # 98 atoms
        >>> f1, f2, f3 = filter_to_common_atoms(arr1, arr2, arr3)
        >>> # All have the same atoms present in all three
    """
    if len(arrays) < 2:
        raise ValueError(f"At least two arrays must be provided, got {len(arrays)}")

    # Validate all inputs are correct types
    for i, array in enumerate(arrays):
        if not isinstance(array, (AtomArray, AtomArrayStack)):
            raise TypeError(
                f"Array at position {i} must be AtomArray or AtomArrayStack, got {type(array)}"
            )

    # Get atom identifiers for all structures
    all_ids = [make_atom_id(array) for array in arrays]

    # Find common atom IDs across all structures
    common_ids = all_ids[0]
    for ids in all_ids[1:]:
        common_ids = np.intersect1d(common_ids, ids)

    if len(common_ids) == 0:
        raise RuntimeError(f"No common atoms found across all {len(arrays)} structures")

    # Filter and sort each array
    filtered_arrays: list[AtomArrayStack] = []

    for array, ids in zip(arrays, all_ids):
        # Create mask for common atoms
        mask = np.isin(ids, common_ids)
        array = ensure_atom_array_stack(array)

        # Filter array
        filtered_array = array[:, mask]

        # Sort by atom ID to ensure matching order
        sort_idx = np.argsort(make_atom_id(filtered_array))  # pyright: ignore (reportArgumentType)
        filtered_array = cast(AtomArrayStack, filtered_array[:, sort_idx])  # pyright: ignore

        filtered_arrays.append(filtered_array)

    return tuple(filtered_arrays)
