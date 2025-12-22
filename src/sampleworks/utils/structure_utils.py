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
