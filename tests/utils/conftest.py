"""Fixtures for tests under ``tests/utils``."""

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack

from tests.utils.atom_array_builders import build_test_atom_array


@pytest.fixture
def simple_atom_array() -> AtomArray:
    """Small AtomArray with valid coords, elements, occupancy, and B-factors."""
    atom_array = AtomArray(5)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture
def simple_atom_array_stack() -> AtomArrayStack:
    """AtomArrayStack with two models."""
    arrays = []
    for i in range(2):
        atom_array = AtomArray(3)
        base_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_array.coord = base_coords + i * 0.1
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0]))
        arrays.append(atom_array)
    return stack(arrays)


@pytest.fixture
def basic_atom_array_altloc() -> AtomArray:
    """AtomArray with mixed altloc IDs and occupancies."""
    atom_array = AtomArray(5)
    atom_array.coord = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    atom_array.set_annotation("chain_id", np.array(["A", "A", "A", "A", "A"]))
    atom_array.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))
    return atom_array


@pytest.fixture
def atom_array_stack_altloc() -> AtomArrayStack:
    """AtomArrayStack with mixed altloc IDs."""
    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array_stack.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))
    atom_array_stack.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array_stack.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))
    return atom_array_stack


@pytest.fixture
def atom_array_missing_altloc_id() -> AtomArray:
    """AtomArray without altloc_id annotation."""
    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("occupancy", np.ones(5))
    return atom_array


@pytest.fixture
def atom_array_missing_occupancy() -> AtomArray:
    """AtomArray without occupancy annotation."""
    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("altloc_id", np.array(["A"] * 5))
    return atom_array


@pytest.fixture
def atom_array_partial_overlap() -> tuple[AtomArray, AtomArray]:
    """Two AtomArrays with three shared atoms by chain/residue/atom identity."""
    array1 = AtomArray(5)
    array1.coord = np.random.rand(5, 3)
    array1.set_annotation("chain_id", np.array(["A"] * 5))
    array1.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    array1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    array1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array1.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    array2 = AtomArray(5)
    array2.coord = np.random.rand(5, 3)
    array2.set_annotation("chain_id", np.array(["A"] * 5))
    array2.set_annotation("res_id", np.array([3, 4, 5, 6, 7]))
    array2.set_annotation("res_name", np.array(["VAL", "LEU", "SER", "THR", "TYR"]))
    array2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array2.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    return array1, array2


@pytest.fixture
def atom_array_stacks_partial_overlap() -> tuple[AtomArrayStack, AtomArrayStack]:
    """Two AtomArrayStacks with shared atom identities."""
    arrays1 = []
    for i in range(2):
        atom_array = AtomArray(4)
        atom_array.coord = np.random.rand(4, 3) + i * 0.1
        arrays1.append(atom_array)
    stack1 = stack(arrays1)
    stack1.set_annotation("chain_id", np.array(["A"] * 4))
    stack1.set_annotation("res_id", np.array([1, 2, 3, 4]))
    stack1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU"]))
    stack1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack1.set_annotation("element", np.array(["C", "C", "C", "C"]))

    arrays2 = []
    for i in range(2):
        atom_array = AtomArray(4)
        atom_array.coord = np.random.rand(4, 3) + i * 0.1
        arrays2.append(atom_array)
    stack2 = stack(arrays2)
    stack2.set_annotation("chain_id", np.array(["A"] * 4))
    stack2.set_annotation("res_id", np.array([2, 3, 4, 5]))
    stack2.set_annotation("res_name", np.array(["GLY", "VAL", "LEU", "SER"]))
    stack2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack2.set_annotation("element", np.array(["C", "C", "C", "C"]))

    return stack1, stack2


@pytest.fixture
def backbone_two_residues() -> AtomArray:
    """Chain A, two residues with atoms: res1=[N,CA], res2=[N]."""
    return build_test_atom_array(
        chain_ids=["A", "A", "A"], res_ids=[1, 1, 2], atom_names=["N", "CA", "N"]
    )


@pytest.fixture
def backbone_two_residues_offset() -> AtomArray:
    """Same layout as ``backbone_two_residues`` with residue IDs shifted to 100/101."""
    return build_test_atom_array(
        chain_ids=["A", "A", "A"], res_ids=[100, 100, 101], atom_names=["N", "CA", "N"]
    )


@pytest.fixture
def two_chain_array() -> AtomArray:
    """Single N atom on chain A and chain B at the same residue ID."""
    return build_test_atom_array(chain_ids=["A", "B"], res_ids=[1, 1], atom_names=["N", "N"])


@pytest.fixture
def model_struct_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """Model and structure with numbering offsets and partial atom overlap."""
    model = build_test_atom_array(
        chain_ids=["A", "A", "A", "A", "A"],
        res_ids=[0, 0, 0, 1, 1],
        atom_names=["N", "CA", "CB", "N", "CA"],
    )
    struct = build_test_atom_array(
        chain_ids=["A", "A", "A"],
        res_ids=[5, 5, 6],
        atom_names=["N", "CA", "N"],
    )
    return model, struct


@pytest.fixture
def single_atom_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """Single-atom arrays with different residue numbering (0 vs 5)."""
    return (
        build_test_atom_array(chain_ids=["A"], res_ids=[0], atom_names=["N"]),
        build_test_atom_array(chain_ids=["A"], res_ids=[5], atom_names=["N"]),
    )


@pytest.fixture
def multi_chain_numbering_offset() -> tuple[AtomArray, AtomArray]:
    """Multi-chain model/structure pair with residue numbering offset."""
    return (
        build_test_atom_array(
            chain_ids=["A", "A", "B", "B"], res_ids=[0, 0, 0, 0], atom_names=["N", "CA", "N", "CA"]
        ),
        build_test_atom_array(
            chain_ids=["A", "A", "B"], res_ids=[1, 1, 1], atom_names=["N", "CA", "N"]
        ),
    )


@pytest.fixture
def model_struct_partial_atom_names() -> tuple[AtomArray, AtomArray]:
    """Model and structure sharing only CA/C atom names."""
    return (
        build_test_atom_array(
            chain_ids=["A", "A", "A", "A"], res_ids=[0, 0, 0, 0], atom_names=["N", "CA", "C", "CB"]
        ),
        build_test_atom_array(
            chain_ids=["A", "A", "A"], res_ids=[7, 7, 7], atom_names=["CA", "C", "O"]
        ),
    )


@pytest.fixture
def reordered_chains() -> tuple[AtomArray, AtomArray]:
    """Model has chains [B, A], struct has chains [A, B] — same atoms, different order."""
    return (
        build_test_atom_array(chain_ids=["B", "A"], res_ids=[1, 1], atom_names=["N", "N"]),
        build_test_atom_array(chain_ids=["A", "B"], res_ids=[1, 1], atom_names=["N", "N"]),
    )
