"""Fixtures for atom_array_utils tests."""

import numpy as np
import pytest
from biotite.structure import array, Atom, AtomArray


def _build_atom_array(
    chain_ids: list[str],
    res_ids: list[int],
    atom_names: list[str],
) -> AtomArray:
    """Build a minimal AtomArray from parallel annotation lists."""
    atoms = []
    for c, r, a in zip(chain_ids, res_ids, atom_names):
        atoms.append(
            Atom(
                coord=np.zeros(3, dtype=np.float32),
                chain_id=c,
                res_id=r,
                res_name="ALA",
                atom_name=a,
                element="C",
            )
        )
    return array(atoms)


@pytest.fixture
def backbone_two_residues() -> AtomArray:
    """
    Return an AtomArray representing two sequential residues on chain "A" with backbone atoms.
    
    The first residue has res_id 1 and atoms "N" and "CA"; the second residue has res_id 2 and atom "N".
    
    Returns:
        AtomArray: An AtomArray with three atoms on chain "A" with res_ids [1, 1, 2] and atom_names ["N", "CA", "N"].
    """
    return _build_atom_array(["A", "A", "A"], [1, 1, 2], ["N", "CA", "N"])


@pytest.fixture
def backbone_two_residues_offset() -> AtomArray:
    """
    Construct an AtomArray with a two-residue backbone layout using residue IDs 100 and 101 on chain "A".
    
    Returns:
        AtomArray: Three-atom array with chain_ids ["A", "A", "A"], res_ids [100, 100, 101], and atom_names ["N", "CA", "N"].
    """
    return _build_atom_array(["A", "A", "A"], [100, 100, 101], ["N", "CA", "N"])


@pytest.fixture
def negative_zero_res_id_pair() -> tuple[AtomArray, AtomArray]:
    """
    Create two AtomArray instances that differ only by residue numbering.
    
    Both arrays contain two atoms on chain "A" with atom names ["N", "N"]. The first array has residue IDs -5 and -4; the second has residue IDs 0 and 1.
    
    Returns:
        pair (tuple[AtomArray, AtomArray]): A tuple where the first element is the AtomArray with residue IDs -5 and -4, and the second element is the AtomArray with residue IDs 0 and 1.
    """
    return (
        _build_atom_array(["A", "A"], [-5, -4], ["N", "N"]),
        _build_atom_array(["A", "A"], [0, 1], ["N", "N"]),
    )


@pytest.fixture
def two_chain_array() -> AtomArray:
    """
    Return an AtomArray containing one 'N' atom on chain 'A' and one 'N' atom on chain 'B' with the same residue id.
    
    Returns:
        AtomArray: Two-atom AtomArray with chain_ids ["A", "B"], res_ids [1, 1], and atom_names ["N", "N"].
    """
    return _build_atom_array(["A", "B"], [1, 1], ["N", "N"])


@pytest.fixture
def model_struct_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """
    Model and structure AtomArrays that illustrate a residue-numbering offset between model and structure.
    
    Model contains two residues (0 and 1) with backbone and a sidechain atom: residue 0 has atoms N, CA, CB; residue 1 has N, CA. Struct contains backbone-only residues with an offset in numbering: residue 5 has N, CA; residue 6 has N.
    
    Returns:
        tuple[AtomArray, AtomArray]: (model, struct) where `model` is the AtomArray described above and `struct` is the backbone-only AtomArray with offset residue numbering.
    """
    model = _build_atom_array(
        ["A", "A", "A", "A", "A"],
        [0, 0, 0, 1, 1],
        ["N", "CA", "CB", "N", "CA"],
    )
    struct = _build_atom_array(
        ["A", "A", "A"],
        [5, 5, 6],
        ["N", "CA", "N"],
    )
    return model, struct


@pytest.fixture
def single_atom_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """
    Provide two single-atom AtomArray instances that differ only by residue numbering.
    
    Returns:
        pair (tuple[AtomArray, AtomArray]): A tuple of two AtomArray objects. The first contains a single atom on chain "A" with res_id 0 and atom_name "N". The second contains a single atom on chain "A" with res_id 5 and atom_name "N".
    """
    return (
        _build_atom_array(["A"], [0], ["N"]),
        _build_atom_array(["A"], [5], ["N"]),
    )


@pytest.fixture
def model_struct_extra_element() -> tuple[AtomArray, AtomArray]:
    """
    Builds a pair of AtomArray objects where the structure contains an extra atom ('SE') and uses a different residue numbering.
    
    Returns:
        tuple[AtomArray, AtomArray]: (model, struct) where
            - model: chain "A", residues [0, 0], atom_names ["N", "CA"]
            - struct: chain "A", residues [1, 1, 1], atom_names ["N", "CA", "SE"]
    """
    return (
        _build_atom_array(["A", "A"], [0, 0], ["N", "CA"]),
        _build_atom_array(["A", "A", "A"], [1, 1, 1], ["N", "CA", "SE"]),
    )


@pytest.fixture
def multi_chain_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """
    Create a model/structure pair demonstrating per-chain residue numbering offsets.
    
    The returned tuple contains two AtomArray objects:
    - model: chains "A" and "B" each contain backbone atoms "N" and "CA" with residue IDs [0, 0, 0, 0].
    - struct: chain "A" contains "N" and "CA" with residue ID 1, and chain "B" contains only "N" with residue ID 1.
    
    Returns:
        tuple[AtomArray, AtomArray]: (model, struct) pair illustrating multi-chain numbering offsets.
    """
    return (
        _build_atom_array(["A", "A", "B", "B"], [0, 0, 0, 0], ["N", "CA", "N", "CA"]),
        _build_atom_array(["A", "A", "B"], [1, 1, 1], ["N", "CA", "N"]),
    )


@pytest.fixture
def model_struct_partial_atom_names() -> tuple[AtomArray, AtomArray]:
    """Model and struct sharing only some atom names (CA and C overlap).

    Model: single residue with [N, CA, C, CB]
    Structure: single residue (offset) with [CA, C, O]
    """
    return (
        _build_atom_array(["A", "A", "A", "A"], [0, 0, 0, 0], ["N", "CA", "C", "CB"]),
        _build_atom_array(["A", "A", "A"], [7, 7, 7], ["CA", "C", "O"]),
    )