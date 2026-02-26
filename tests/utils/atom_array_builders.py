"""Shared AtomArray builders for tests."""

from collections.abc import Sequence

import numpy as np
from biotite.structure import array, Atom, AtomArray


def build_test_atom_array(
    *,
    n_atoms: int | None = None,
    chain_ids: Sequence[str] | None = None,
    res_ids: Sequence[int] | None = None,
    atom_names: Sequence[str] | None = None,
    coords: np.ndarray | None = None,
    res_name: str = "ALA",
    element: str = "C",
    with_occupancy: bool = True,
    occupancy_value: float = 1.0,
    b_factor_value: float = 20.0,
) -> AtomArray:
    """Build a minimal AtomArray with configurable annotations.

    Parameters
    ----------
    n_atoms
        Number of atoms when explicit annotation lists are not provided.
    chain_ids
        Per-atom chain IDs.
    res_ids
        Per-atom residue IDs.
    atom_names
        Per-atom atom names.
    coords
        Optional coordinate array with shape ``(n_atoms, 3)``.
    res_name
        Residue name assigned to all atoms.
    element
        Element symbol assigned to all atoms.
    with_occupancy
        Whether to attach an occupancy annotation.
    occupancy_value
        Occupancy value used when ``with_occupancy=True``.
    b_factor_value
        B-factor value assigned to all atoms.

    Returns
    -------
    AtomArray
        Configured atom array for tests.
    """
    if chain_ids is not None and res_ids is not None and atom_names is not None:
        if not (len(chain_ids) == len(res_ids) == len(atom_names)):
            raise ValueError("chain_ids, res_ids, and atom_names must have matching lengths")
        n = len(chain_ids)
    elif n_atoms is not None:
        n = n_atoms
        chain_ids = ["A"] * n
        res_ids = list(range(1, n + 1))
        atom_names = [f"C{i}" for i in range(n)]
    else:
        raise ValueError("Provide either n_atoms or all of chain_ids/res_ids/atom_names")

    if coords is None:
        coords = np.zeros((n, 3), dtype=np.float32)
    if coords.shape != (n, 3):
        raise ValueError(f"coords must have shape ({n}, 3), got {coords.shape}")

    atoms = []
    for i, (chain_id, res_id, atom_name) in enumerate(zip(chain_ids, res_ids, atom_names)):
        atoms.append(
            Atom(
                coord=np.asarray(coords[i], dtype=np.float32),
                chain_id=str(chain_id),
                res_id=int(res_id),
                res_name=res_name,
                atom_name=str(atom_name),
                element=element,
            )
        )

    arr = array(atoms)
    if with_occupancy:
        arr.set_annotation("occupancy", np.full(n, occupancy_value, dtype=np.float32))
    arr.set_annotation("b_factor", np.full(n, b_factor_value, dtype=np.float32))
    return arr
