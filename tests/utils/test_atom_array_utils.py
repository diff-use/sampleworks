"""Tests for atom_array_utils module."""
import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack

from sampleworks.utils.atom_array_utils import select_altloc, filter_to_common_atoms


# Fixtures for creating test data

@pytest.fixture(scope="module")
def basic_atom_array() -> AtomArray:
    """AtomArray with mixed altloc_ids."""
    atom_array = AtomArray(5)
    coord = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ])
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A", "A", "A", "A", "A"]))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.4, 0.5]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_full_occupancy():
    """AtomArray with some atoms having full occupancy."""
    atom_array = AtomArray(7)
    atom_array.coord =np.random.rand(7, 3)
    atom_array.set_annotation("chain_id", np.array(["A"] * 7))
    atom_array.set_annotation("res_id", np.arange(1, 8))
    atom_array.set_annotation("res_name", np.array(["ALA"] * 7))
    atom_array.set_annotation("atom_name", np.array(["CA"] * 7))
    atom_array.set_annotation("element", np.array(["C"] * 7))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B", "D", "E"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 1.0, 0.4, 1.0, 0.7]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack():
    """AtomArrayStack with 3 models."""
    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.arange(1, 6))
    atom_array_stack.set_annotation("res_name", np.array(["ALA"] * 5))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))
    atom_array_stack.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B"]))
    atom_array_stack.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 1.0, 0.4]))

    return atom_array_stack


@pytest.fixture(scope="module")
def atom_array_missing_altloc_id():
    """AtomArray without altloc_id annotation."""
    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("occupancy", np.ones(5))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_missing_occupancy():
    """AtomArray without occupancy annotation."""
    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("altloc_id", np.array(["A"] * 5))
    return atom_array


# Tests

class TestSelectAltlocBasic:
    """Basic functionality tests for select_altloc."""

    def test_select_specific_altloc(self, basic_atom_array):
        """Test selecting atoms with a specific altloc_id."""
        result = select_altloc(basic_atom_array, "A", return_full_array=False)

        assert len(result) == 2
        assert all(result.altloc_id == "A")
        assert list(result.res_name) == ["ALA", "VAL"]

    def test_select_different_altloc(self, basic_atom_array):
        """Test selecting different altloc_id values."""
        result_b = select_altloc(basic_atom_array, "B", return_full_array=False)
        result_c = select_altloc(basic_atom_array, "C", return_full_array=False)

        assert len(result_b) == 2
        assert all(result_b.altloc_id == "B")
        assert len(result_c) == 1
        assert all(result_c.altloc_id == "C")

    def test_select_nonexistent_altloc(self, basic_atom_array):
        """Test selecting altloc that doesn't exist returns empty array."""
        result = select_altloc(basic_atom_array, "Z", return_full_array=False)

        assert len(result) == 0

    def test_preserves_coordinates(self, basic_atom_array):
        """Test that selected atoms preserve their original coordinates."""
        result = select_altloc(basic_atom_array, "A", return_full_array=False)

        np.testing.assert_array_equal(result.coord[0], basic_atom_array.coord[0])
        np.testing.assert_array_equal(result.coord[1], basic_atom_array.coord[2])

    def test_preserves_annotations(self, basic_atom_array):
        """Test that other annotations are preserved."""
        result = select_altloc(basic_atom_array, "A", return_full_array=False)

        assert list(result.chain_id) == ["A", "A"]
        assert list(result.res_id) == [1, 3]
        assert list(result.atom_name) == ["CA", "CA"]


class TestSelectAltlocWithFullArray:
    """Tests for return_full_array=True mode."""

    def test_includes_full_occupancy(self, atom_array_with_full_occupancy):
        """Test that return_full_array=True includes atoms with occupancy=1.0."""
        result = select_altloc(atom_array_with_full_occupancy, "A", return_full_array=True)

        # Should include: 2 "A" atoms + 2 atoms with occupancy=1.0
        assert len(result) == 4
        assert all((result.altloc_id == "A") | (result.occupancy == 1.0))

    def test_all_full_occupancy(self, atom_array_with_full_occupancy):
        """Test selecting when all atoms have occupancy=1.0 in result."""
        # Modify to have all occupancy=1.0
        atom_array = atom_array_with_full_occupancy.copy()
        atom_array.occupancy = np.ones(len(atom_array))

        result = select_altloc(atom_array, "Z", return_full_array=True)

        # Should return all atoms since they all have occupancy=1.0
        assert len(result) == len(atom_array)

    def test_excludes_others_when_false(self, atom_array_with_full_occupancy):
        """Test that return_full_array=False excludes occupancy=1.0 atoms."""
        result = select_altloc(atom_array_with_full_occupancy, "A", return_full_array=False)

        # Should only include the "A" atoms, not occupancy=1.0
        assert len(result) == 2
        assert all(result.altloc_id == "A")


class TestSelectAltlocWithStack:
    """Tests for AtomArrayStack inputs."""

    def test_select_from_stack(self, atom_array_stack):
        """Test selecting altloc from AtomArrayStack."""
        result = select_altloc(atom_array_stack, "A", return_full_array=False)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 3
        assert result.array_length() == 2
        assert all(result[0].altloc_id == "A")
        assert all(result[1].altloc_id == "A")
        assert all(result[2].altloc_id == "A")

    def test_select_from_stack_with_full_array(self, atom_array_stack):
        """Test selecting from stack with return_full_array=True."""
        result = select_altloc(atom_array_stack, "A", return_full_array=True)

        # Should include: 2 "A" atoms + 1 atom with occupancy=1.0
        assert result.stack_depth() == 3
        assert result.array_length() == 3

    def test_stack_preserves_all_models(self, atom_array_stack):
        """Test that all models in stack are preserved after filtering."""
        result = select_altloc(atom_array_stack, "B", return_full_array=False)

        assert result.stack_depth() == atom_array_stack.stack_depth()
        # Each model should have 2 "B" atoms
        for i in range(result.stack_depth()):
            assert len(result[i]) == 2
            assert all(result[i].altloc_id == "B")


class TestSelectAltlocErrors:
    """Tests for error handling."""

    def test_missing_altloc_id_raises_error(self, atom_array_missing_altloc_id):
        """Test that missing altloc_id annotation raises AttributeError."""
        with pytest.raises(AttributeError, match="must have `altloc_id` and `occupancy` annotations"):
            select_altloc(atom_array_missing_altloc_id, "A")

    def test_missing_occupancy_raises_error(self, atom_array_missing_occupancy):
        """Test that missing occupancy annotation raises AttributeError."""
        with pytest.raises(AttributeError, match="must have `altloc_id` and `occupancy` annotations"):
            select_altloc(atom_array_missing_occupancy, "A")

    def test_missing_both_attributes_raises_error(self):
        """Test that missing both altloc_id and occupancy raises AttributeError."""
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3)

        with pytest.raises(AttributeError, match="must have `altloc_id` and `occupancy` annotations"):
            select_altloc(atom_array, "A")

    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises TypeError."""
        invalid_input = "not an atom array"

        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(invalid_input, "A")

    def test_none_input_raises_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(None, "A")


class TestSelectAltlocEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self):
        """Test with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.altloc_id = np.array([], dtype=str)
        atom_array.occupancy = np.array([], dtype=float)

        result = select_altloc(atom_array, "A", return_full_array=False)

        assert len(result) == 0

    def test_all_same_altloc(self):
        """Test when all atoms have the same altloc_id."""
        atom_array = AtomArray(4)
        atom_array.set_annotation("altloc_id", np.array(["A", "A", "A", "A"]))
        atom_array.set_annotation("occupancy", np.array([0.5, 0.6, 0.7, 0.8]))

        result = select_altloc(atom_array, "A", return_full_array=False)

        assert len(result) == 4
        assert all(result.altloc_id == "A")


# Fixtures for filter_to_common_atoms tests

@pytest.fixture(scope="module")
def atom_array_partial_overlap():
    """Two AtomArrays with partial overlap."""
    # Array 1: 5 atoms (residues 1-5)
    array1 = AtomArray(5)
    array1.coord = np.random.rand(5, 3)
    array1.set_annotation("chain_id", np.array(["A"] * 5))
    array1.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    array1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    array1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array1.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    # Array 2: 5 atoms (residues 3-7, overlaps with 3-5 from array1)
    array2 = AtomArray(5)
    array2.coord = np.random.rand(5, 3)
    array2.set_annotation("chain_id", np.array(["A"] * 5))
    array2.set_annotation("res_id", np.array([3, 4, 5, 6, 7]))
    array2.set_annotation("res_name", np.array(["VAL", "LEU", "SER", "THR", "TYR"]))
    array2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array2.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    return array1, array2


@pytest.fixture(scope="module")
def atom_array_stacks_partial_overlap():
    """Two AtomArrayStacks with partial overlap."""
    # Stack 1: 2 models, 4 atoms each
    arrays1 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays1.append(array)
    stack1 = stack(arrays1)
    stack1.set_annotation("chain_id", np.array(["A"] * 4))
    stack1.set_annotation("res_id", np.array([1, 2, 3, 4]))
    stack1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU"]))
    stack1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack1.set_annotation("element", np.array(["C", "C", "C", "C"]))

    # Stack 2: 2 models, 4 atoms each (overlaps with res 2-4 from stack1)
    arrays2 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays2.append(array)
    stack2 = stack(arrays2)
    stack2.set_annotation("chain_id", np.array(["A"] * 4))
    stack2.set_annotation("res_id", np.array([2, 3, 4, 5]))
    stack2.set_annotation("res_name", np.array(["GLY", "VAL", "LEU", "SER"]))
    stack2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack2.set_annotation("element", np.array(["C", "C", "C", "C"]))

    return stack1, stack2


# Tests for filter_to_common_atoms

class TestFilterToCommonAtoms:
    """Tests for filter_to_common_atoms function."""

    def test_filters_to_common_atoms(self, atom_array_partial_overlap):
        """Test that only common atoms are returned."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Should have 3 common atoms (residues 3, 4, 5)
        assert len(filtered1) == 3
        assert len(filtered2) == 3
        assert list(filtered1.res_id) == [3, 4, 5]
        assert list(filtered2.res_id) == [3, 4, 5]

    def test_atoms_in_matching_order(self, atom_array_partial_overlap):
        """Test that returned atoms are in matching order."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Check that atom identifiers match
        assert np.array_equal(filtered1.chain_id, filtered2.chain_id)
        assert np.array_equal(filtered1.res_id, filtered2.res_id)
        assert np.array_equal(filtered1.atom_name, filtered2.atom_name)

    def test_with_identical_arrays(self, basic_atom_array):
        """Test with two identical arrays."""
        filtered1, filtered2 = filter_to_common_atoms(basic_atom_array, basic_atom_array)

        # Should return all atoms
        assert len(filtered1) == len(basic_atom_array)
        assert len(filtered2) == len(basic_atom_array)

    def test_with_no_overlap(self):
        """Test with arrays that have no common atoms."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["A"] * 3))
        array2.set_annotation("res_id", np.array([4, 5, 6]))
        array2.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No common atoms found between the two structures"):
            filter_to_common_atoms(array1, array2)

    def test_with_stacks(self, atom_array_stacks_partial_overlap):
        """Test with AtomArrayStacks."""
        stack1, stack2 = atom_array_stacks_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(stack1, stack2)

        # Should have 3 common atoms (residues 2, 3, 4)
        assert isinstance(filtered1, AtomArrayStack)
        assert isinstance(filtered2, AtomArrayStack)
        assert filtered1.stack_depth() == 2
        assert filtered2.stack_depth() == 2
        assert filtered1.array_length() == 3
        assert filtered2.array_length() == 3

    def test_mixed_array_and_stack(self, basic_atom_array, atom_array_stack):
        """Test with one AtomArray and one AtomArrayStack."""
        # Both have same res_ids 1-5
        filtered1, filtered2 = filter_to_common_atoms(basic_atom_array, atom_array_stack)

        assert isinstance(filtered1, AtomArray)
        assert isinstance(filtered2, AtomArrayStack)
        assert len(filtered1) == 5
        assert filtered2.array_length() == 5

    def test_different_chains_no_overlap(self):
        """Test arrays with different chain IDs have no overlap."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["B"] * 3))
        array2.set_annotation("res_id", np.array([1, 2, 3]))
        array2.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No common atoms found"):
            filter_to_common_atoms(array1, array2)

    def test_different_atom_names_no_overlap(self):
        """Test arrays with different atom names have no overlap."""
        array1 = AtomArray(3)
        array1.coord = np.random.rand(3, 3)
        array1.set_annotation("chain_id", np.array(["A"] * 3))
        array1.set_annotation("res_id", np.array([1, 2, 3]))
        array1.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        array2 = AtomArray(3)
        array2.coord = np.random.rand(3, 3)
        array2.set_annotation("chain_id", np.array(["A"] * 3))
        array2.set_annotation("res_id", np.array([1, 2, 3]))
        array2.set_annotation("atom_name", np.array(["CB", "CB", "CB"]))

        with pytest.raises(RuntimeError, match="No common atoms found"):
            filter_to_common_atoms(array1, array2)

    def test_invalid_type_first_arg(self):
        """Test that invalid first argument raises TypeError."""
        array = AtomArray(3)
        array.coord = np.random.rand(3, 3)

        with pytest.raises(TypeError, match="must be AtomArray or AtomArrayStack"):
            filter_to_common_atoms("not an array", array)

    def test_invalid_type_second_arg(self):
        """Test that invalid second argument raises TypeError."""
        array = AtomArray(3)
        array.coord = np.random.rand(3, 3)

        with pytest.raises(TypeError, match="must be AtomArray or AtomArrayStack"):
            filter_to_common_atoms(array, None)

    def test_preserves_coordinates(self, atom_array_partial_overlap):
        """Test that coordinates are preserved for common atoms."""
        array1, array2 = atom_array_partial_overlap
        original_coords1 = array1.coord.copy()
        original_coords2 = array2.coord.copy()

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Filtered arrays should have coordinates from residues 3, 4, 5
        # which are indices 2, 3, 4 in array1 and 0, 1, 2 in array2
        np.testing.assert_array_equal(filtered1.coord[0], original_coords1[2])
        np.testing.assert_array_equal(filtered2.coord[0], original_coords2[0])
