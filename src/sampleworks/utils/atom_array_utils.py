import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io.pdbx import CIFFile, set_structure



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
                structure_to_save = cast(
                    AtomArrayStack, structure_to_save[:, clean_mask]
                )
        else:
            coords: Any = structure_to_save.coord
            nan_mask = np.isnan(coords).any(axis=-1)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                warnings.warn(
                    f"Found {n_nan} atoms with NaN coordinates. "
                    "Filtering them out before saving.",
                    UserWarning,
                    stacklevel=2,
                )
                structure_to_save = cast(AtomArray, structure_to_save[~nan_mask])

    cif_file = CIFFile()
    set_structure(cif_file, structure_to_save)
    cif_file.write(str(output_path))

    return output_path


def select_altloc(
        atom_array: AtomArray | AtomArrayStack,
        altloc_id: str,
        return_full_array: bool = False
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
            an empty/default altloc.
    """
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(
            f"Unexpected type: {type(atom_array)}, can only accept AtomArray or AtomArrayStack"
        )

    if not (hasattr(atom_array, "altloc_id") and hasattr(atom_array, "occupancy")):
        raise AttributeError("atom_array must have `altloc_id` and `occupancy` annotations")

    if return_full_array:
        mask = (atom_array.altloc_id == altloc_id) | (atom_array.occupancy == 1.0)
    else:
        mask = atom_array.altloc_id == altloc_id

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


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

    mask = ~atom_array.hetero

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


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

    mask = atom_array.element != "H"

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


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

    backbone_atoms = np.array(["C", "CA", "N", "O"])
    mask = np.isin(atom_array.atom_name, backbone_atoms)

    if isinstance(atom_array, AtomArrayStack):
        return atom_array[:, mask]
    else:
        return atom_array[mask]


def make_atom_id(arr: AtomArray | AtomArrayStack) -> np.ndarray:
    """Create a unique identifier for each atom."""
    return np.array([
        f"{chain}_{res}_{atom}"
        for chain, res, atom in zip(arr.chain_id, arr.res_id, arr.atom_name)
    ])


def filter_to_common_atoms(
    array1: AtomArray | AtomArrayStack,
    array2: AtomArray | AtomArrayStack,
) -> tuple[AtomArray | AtomArrayStack, AtomArray | AtomArrayStack]:
    """Filter two AtomArrays/AtomArrayStacks to only include common atoms.

    Creates unique identifiers for each atom based on chain_id, res_id, and
    atom_name, then filters both structures to include only atoms present in both.
    The returned arrays are sorted to ensure atoms are in matching order.

    Parameters:
        array1 (AtomArray | AtomArrayStack): First atom array or stack.
        array2 (AtomArray | AtomArrayStack): Second atom array or stack.

    Returns:
        tuple[AtomArray | AtomArrayStack, AtomArray | AtomArrayStack]:
            Filtered versions of array1 and array2 containing only common atoms
            in matching order.

    Raises:
        TypeError: If inputs are not AtomArray or AtomArrayStack.
        RuntimeError: If no common atoms are found between the two structures.

    Examples:
        >>> array1 = AtomArray(...)  # 100 atoms
        >>> array2 = AtomArray(...)  # 95 atoms, 90 overlap with array1
        >>> filtered1, filtered2 = filter_to_common_atoms(array1, array2)
        >>> len(filtered1)  # 90
        >>> len(filtered2)  # 90
    """
    if not (
        isinstance(array1, (AtomArray, AtomArrayStack))
        and isinstance(array2, (AtomArray, AtomArrayStack))
    ):
        raise TypeError(
            f"array1 and array2 must be AtomArray or AtomArrayStack, "
            f"got {type(array1)} and {type(array2)}"
        )

    # Get atom identifiers for both structures
    ids1 = make_atom_id(array1)
    ids2 = make_atom_id(array2)

    # Find common atom IDs
    common_ids = np.intersect1d(ids1, ids2)

    if len(common_ids) == 0:
        raise RuntimeError("No common atoms found between the two structures")

    # Create masks for common atoms
    mask1 = np.isin(ids1, common_ids)
    mask2 = np.isin(ids2, common_ids)

    # Filter arrays
    if isinstance(array1, AtomArrayStack):
        filtered_array1 = array1[:, mask1]
    else:
        filtered_array1 = array1[mask1]

    if isinstance(array2, AtomArrayStack):
        filtered_array2 = array2[:, mask2]
    else:
        filtered_array2 = array2[mask2]

    # Sort by atom ID to ensure matching order
    sort_idx1 = np.argsort(make_atom_id(filtered_array1))
    sort_idx2 = np.argsort(make_atom_id(filtered_array2))

    if isinstance(filtered_array1, AtomArrayStack):
        filtered_array1 = filtered_array1[:, sort_idx1]
    else:
        filtered_array1 = filtered_array1[sort_idx1]

    if isinstance(filtered_array2, AtomArrayStack):
        filtered_array2 = filtered_array2[:, sort_idx2]
    else:
        filtered_array2 = filtered_array2[sort_idx2]

    return filtered_array1, filtered_array2
