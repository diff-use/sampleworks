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
    """Chain A, 2 residues with backbone atoms: res 1 is [N, CA], res 2 is [N]."""
    return _build_atom_array(["A", "A", "A"], [1, 1, 2], ["N", "CA", "N"])


@pytest.fixture
def backbone_two_residues_offset() -> AtomArray:
    """Same atom layout as backbone_two_residues, res_ids offset to 100 and 101."""
    return _build_atom_array(["A", "A", "A"], [100, 100, 101], ["N", "CA", "N"])


@pytest.fixture
def negative_zero_res_id_pair() -> tuple[AtomArray, AtomArray]:
    """Two 2-atom arrays: one with negative res_ids [-5, -4], one 0 indexed [0, 1]."""
    return (
        _build_atom_array(["A", "A"], [-5, -4], ["N", "N"]),
        _build_atom_array(["A", "A"], [0, 1], ["N", "N"]),
    )


@pytest.fixture
def two_chain_array() -> AtomArray:
    """Single N atom on chain A and chain B at the same res_id."""
    return _build_atom_array(["A", "B"], [1, 1], ["N", "N"])


@pytest.fixture
def model_struct_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """Model with backbone + sidechain (0-based) and structure with backbone only (offset).

    Model: res 0 is [N, CA, CB], res 1 is [N, CA]
    Struct: res 5 is [N, CA], res 6 is [N]
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
    """Two single atom arrays with different res_ids (0 vs 5)."""
    return (
        _build_atom_array(["A"], [0], ["N"]),
        _build_atom_array(["A"], [5], ["N"]),
    )


@pytest.fixture
def model_struct_extra_element() -> tuple[AtomArray, AtomArray]:
    """Model [N, CA] and struct [N, CA, SE] where struct has an extra atom type.

    Model: res 0.  Struct: res 1 (different numbering).
    """
    return (
        _build_atom_array(["A", "A"], [0, 0], ["N", "CA"]),
        _build_atom_array(["A", "A", "A"], [1, 1, 1], ["N", "CA", "SE"]),
    )


@pytest.fixture
def multi_chain_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """Multi-chain offsets.

    Model: A is [N, CA], B is [N, CA]
    Struct: A is [N, CA], B is [N]
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
