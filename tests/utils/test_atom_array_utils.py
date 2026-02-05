"""Tests for atom_array_utils module."""

from typing import cast

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.filter import filter_polymer
from biotite.structure.info.groups import amino_acid_names
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    keep_amino_acids,
    keep_polymer,
    remove_hydrogens,
    select_altloc,
)


class TestSelectAltlocBasic:
    """Basic functionality tests for select_altloc."""

    def test_select_specific_altloc(self, basic_atom_array_altloc):
        """Test selecting atoms with a specific altloc_id."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)

        assert len(result) == 2
        assert all(result.altloc_id == "A")
        assert list(cast(np.ndarray, result.res_name)) == ["ALA", "VAL"]

    def test_select_different_altloc(self, basic_atom_array_altloc):
        """Test selecting different altloc_id values."""
        result_b = select_altloc(basic_atom_array_altloc, "B", return_full_array=False)
        result_c = select_altloc(basic_atom_array_altloc, "C", return_full_array=False)

        assert len(result_b) == 2
        assert all(result_b.altloc_id == "B")
        assert len(result_c) == 1
        assert all(result_c.altloc_id == "C")

    def test_select_nonexistent_altloc(self, basic_atom_array_altloc):
        """Test selecting altloc that doesn't exist returns empty array."""
        result = select_altloc(basic_atom_array_altloc, "Z", return_full_array=False)

        assert len(result) == 0

    def test_preserves_coordinates(self, basic_atom_array_altloc):
        """Test that selected atoms preserve their original coordinates."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)
        result_coord = cast(np.ndarray, result.coord)
        basic_coord = cast(np.ndarray, basic_atom_array_altloc.coord)

        np.testing.assert_array_equal(result_coord[0], basic_coord[0])
        np.testing.assert_array_equal(result_coord[1], basic_coord[2])

    def test_preserves_annotations(self, basic_atom_array_altloc):
        """Test that other annotations are preserved."""
        result = select_altloc(basic_atom_array_altloc, "A", return_full_array=False)

        assert list(cast(np.ndarray, result.chain_id)) == ["A", "A"]
        assert list(cast(np.ndarray, result.res_id)) == [1, 2]
        assert list(cast(np.ndarray, result.atom_name)) == ["CA", "CA"]


class TestSelectAltlocWithStack:
    """Tests for AtomArrayStack inputs."""

    def test_select_from_stack(self, atom_array_stack_altloc):
        """Test selecting altloc from AtomArrayStack."""
        result = select_altloc(atom_array_stack_altloc, "A", return_full_array=False)

        assert isinstance(result, AtomArrayStack)
        assert result.stack_depth() == 3
        assert result.array_length() == 2
        assert all(result[0].altloc_id == "A")
        assert all(result[1].altloc_id == "A")
        assert all(result[2].altloc_id == "A")

    def test_select_from_stack_with_full_array(self, atom_array_stack_altloc):
        """Test selecting from stack with return_full_array=True."""
        result = select_altloc(atom_array_stack_altloc, "A", return_full_array=True)
        result_stack = cast(AtomArrayStack, result)

        # Should include: 2 "A" atoms
        assert result_stack.stack_depth() == 3
        assert result_stack.array_length() == 2

    def test_stack_preserves_all_models(self, atom_array_stack_altloc):
        """Test that all models in stack are preserved after filtering."""
        result = select_altloc(atom_array_stack_altloc, "B", return_full_array=False)
        result_stack = cast(AtomArrayStack, result)

        assert result_stack.stack_depth() == atom_array_stack_altloc.stack_depth()
        # Each model should have 2 "B" atoms
        for i in range(result_stack.stack_depth()):
            model = cast(AtomArray, result_stack[i])
            assert len(model) == 2
            assert all(model.altloc_id == "B")


class TestSelectAltlocErrors:
    """Tests for error handling."""

    def test_missing_altloc_id_raises_error(self, atom_array_missing_altloc_id):
        """Test that missing altloc_id annotation raises AttributeError."""
        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array_missing_altloc_id, "A")

    def test_missing_occupancy_raises_error(self, atom_array_missing_occupancy):
        """Test that missing occupancy annotation raises AttributeError."""
        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array_missing_occupancy, "A")

    def test_missing_both_attributes_raises_error(self):
        """Test that missing both altloc_id and occupancy raises AttributeError."""
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3)

        with pytest.raises(
            AttributeError, match="must have `altloc_id` and `occupancy` annotations"
        ):
            select_altloc(atom_array, "A")

    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises TypeError."""
        invalid_input = "not an atom array"

        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(invalid_input, "A")  # ty: ignore[invalid-argument-type]

    def test_none_input_raises_error(self):
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            select_altloc(None, "A")  # ty: ignore[invalid-argument-type]


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


class TestFilterToCommonAtoms:
    """Tests for filter_to_common_atoms function."""

    def test_filters_to_common_atoms(self, atom_array_partial_overlap):
        """Test that only common atoms are returned."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Should have 3 common atoms (residues 3, 4, 5)
        assert filtered1.array_length() == 3
        assert filtered2.array_length() == 3
        assert list(cast(np.ndarray, filtered1.res_id)) == [3, 4, 5]
        assert list(cast(np.ndarray, filtered2.res_id)) == [3, 4, 5]

    def test_atoms_in_matching_order(self, atom_array_partial_overlap):
        """Test that returned atoms are in matching order."""
        array1, array2 = atom_array_partial_overlap

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        # Check that atom identifiers match
        assert np.array_equal(
            cast(np.ndarray, filtered1.chain_id), cast(np.ndarray, filtered2.chain_id)
        )
        assert np.array_equal(
            cast(np.ndarray, filtered1.res_id), cast(np.ndarray, filtered2.res_id)
        )
        assert np.array_equal(
            cast(np.ndarray, filtered1.atom_name), cast(np.ndarray, filtered2.atom_name)
        )

    def test_with_identical_arrays(self, basic_atom_array_altloc):
        """Test with two identical arrays."""
        filtered1, filtered2 = filter_to_common_atoms(
            basic_atom_array_altloc, basic_atom_array_altloc
        )

        # Should return all atoms
        assert filtered1.array_length() == len(basic_atom_array_altloc)
        assert filtered2.array_length() == len(basic_atom_array_altloc)

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

        with pytest.raises(RuntimeError, match="No common atoms found across all 2 structures"):
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

    def test_mixed_array_and_stack(self, basic_atom_array_altloc, atom_array_stack_altloc):
        """Test with one AtomArray and one AtomArrayStack."""
        # Both have same res_ids 1-5
        filtered1, filtered2 = filter_to_common_atoms(
            basic_atom_array_altloc, atom_array_stack_altloc
        )

        assert isinstance(filtered1, AtomArrayStack)
        assert isinstance(filtered2, AtomArrayStack)
        assert filtered1.array_length() == 5
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
            filter_to_common_atoms("not an array", array)  # ty: ignore[invalid-argument-type]

    def test_invalid_type_second_arg(self):
        """Test that invalid second argument raises TypeError."""
        array = AtomArray(3)
        array.coord = np.random.rand(3, 3)

        with pytest.raises(TypeError, match="must be AtomArray or AtomArrayStack"):
            filter_to_common_atoms(array, None)  # ty: ignore[invalid-argument-type]

    def test_preserves_coordinates(self, atom_array_partial_overlap):
        """Test that coordinates are preserved for common atoms."""
        # TODO filter_to_common_atoms now returns a tuple of AtomArrayStack
        array1, array2 = atom_array_partial_overlap
        original_coords1 = array1.coord.copy()
        original_coords2 = array2.coord.copy()

        filtered1, filtered2 = filter_to_common_atoms(array1, array2)

        assert filtered1 is not None and filtered1.coord is not None
        assert filtered2 is not None and filtered2.coord is not None

        # Filtered arrays should have coordinates from residues 3, 4, 5
        # which are indices 2, 3, 4 in array1 and 0, 1, 2 in array2
        np.testing.assert_array_equal(filtered1.coord[0, 0], original_coords1[2])
        np.testing.assert_array_equal(filtered2.coord[0, 0], original_coords2[0])


def _get_atom_array(structure: dict) -> AtomArray:
    arr = structure["asym_unit"]
    if isinstance(arr, AtomArrayStack):
        return cast(AtomArray, arr[0])
    return cast(AtomArray, arr)


def _get_atom_array_stack(structure: dict) -> AtomArrayStack:
    arr = structure["asym_unit"]
    if isinstance(arr, AtomArray):
        return stack([arr, arr])
    return cast(AtomArrayStack, arr)


class TestRemoveHydrogens:
    """Tests for remove_hydrogens function."""

    def test_no_hydrogen_or_deuterium_in_output(self, test_structure):
        """Output must contain zero H or D and preserve all heavy atoms."""
        arr = _get_atom_array(test_structure)
        result = remove_hydrogens(arr)
        elements = cast(np.ndarray, result.element)
        assert "H" not in elements
        assert "D" not in elements
        input_elements = cast(np.ndarray, arr.element)
        expected_count = int(np.sum((input_elements != "H") & (input_elements != "D")))
        assert len(result) == expected_count

    def test_atom_array_and_stack_give_same_atom_count(self, test_structure):
        """AtomArray and AtomArrayStack paths must select the same atoms."""
        arr_result = remove_hydrogens(_get_atom_array(test_structure))
        stack_result = remove_hydrogens(_get_atom_array_stack(test_structure))

        assert isinstance(stack_result, AtomArrayStack)
        stack_result = cast(AtomArrayStack, stack_result)
        assert len(arr_result) == stack_result.array_length()
        assert "H" not in cast(np.ndarray, stack_result.element)

    def test_is_idempotent(self, test_structure):
        """Property: applying remove_hydrogens twice == applying once."""
        once = remove_hydrogens(_get_atom_array(test_structure))
        twice = remove_hydrogens(once)
        assert len(once) == len(twice)

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            remove_hydrogens("bad input")  # pyright: ignore[reportArgumentType]


class TestKeepPolymer:
    """Tests for keep_polymer function."""

    def test_output_contains_only_polymer_atoms(self, test_structure):
        """Every atom in the result must pass filter_polymer."""
        arr = _get_atom_array(test_structure)
        result = keep_polymer(arr)
        assert len(result) > 0
        polymer_mask = filter_polymer(result)
        assert np.all(polymer_mask)

    def test_atom_array_and_stack_give_same_atom_count(self, test_structure):
        """Both AtomArray and AtomArrayStack should work."""
        arr_result = keep_polymer(_get_atom_array(test_structure))
        stack_result = keep_polymer(_get_atom_array_stack(test_structure))

        assert isinstance(stack_result, AtomArrayStack)
        stack_result = cast(AtomArrayStack, stack_result)
        assert len(arr_result) == stack_result.array_length()

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            keep_polymer(42)  # pyright: ignore[reportArgumentType]


STANDARD_AA_NAMES = frozenset(amino_acid_names())


class TestKeepAminoAcids:
    """Tests for keep_amino_acids function."""

    def test_output_contains_only_standard_amino_acid_residues(self, test_structure):
        """Purity (only AA residues) and completeness (all AA atoms preserved)."""
        arr = _get_atom_array(test_structure)
        result = keep_amino_acids(arr)
        result_names = set(cast(np.ndarray, result.res_name))
        assert len(result) > 0
        assert result_names <= STANDARD_AA_NAMES
        input_names = cast(np.ndarray, arr.res_name)
        expected_count = int(np.sum(np.isin(input_names, list(STANDARD_AA_NAMES))))
        assert len(result) == expected_count

    def test_atom_array_and_stack_give_same_atom_count(self, test_structure):
        """AtomArray and AtomArrayStack paths must select the same atoms."""
        arr_result = keep_amino_acids(_get_atom_array(test_structure))
        stack_result = keep_amino_acids(_get_atom_array_stack(test_structure))

        assert isinstance(stack_result, AtomArrayStack)
        stack_result = cast(AtomArrayStack, stack_result)
        assert len(arr_result) == stack_result.array_length()

    def test_invalid_type_raises_error(self):
        with pytest.raises(TypeError, match="can only accept AtomArray or AtomArrayStack"):
            keep_amino_acids(None)  # pyright: ignore[reportArgumentType]


class TestFilterFunctionsIntegration:
    """Property and composition tests across multiple filter functions."""

    def test_remove_hydrogens_and_keep_amino_acids_commute(self, test_structure):
        """These filters act on independent atom properties so order shouldn't matter."""
        arr = _get_atom_array(test_structure)
        path_a = keep_amino_acids(remove_hydrogens(arr))
        path_b = remove_hydrogens(keep_amino_acids(arr))
        assert len(path_a) == len(path_b)
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.res_id), cast(np.ndarray, path_b.res_id)
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.atom_name), cast(np.ndarray, path_b.atom_name)
        )
        np.testing.assert_array_equal(
            cast(np.ndarray, path_a.element), cast(np.ndarray, path_b.element)
        )

    def test_polymer_then_remove_hydrogens_pipeline(self, test_structure):
        """Chained pipeline produces valid output with no H/D and only polymer atoms."""
        arr = _get_atom_array(test_structure)
        result = remove_hydrogens(keep_polymer(arr))
        assert len(result) > 0
        elements = cast(np.ndarray, result.element)
        assert "H" not in elements
        assert "D" not in elements
