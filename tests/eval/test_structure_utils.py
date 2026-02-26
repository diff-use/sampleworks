"""Tests for structure_utils module."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.eval.structure_utils import (
    apply_selection,
    extract_selection_coordinates,
    get_asym_unit_from_structure,
    get_reference_atomarraystack,
    get_reference_structure_coords,
    parse_selection_string,
)


@pytest.fixture
def mock_protein_config(tmp_path: Path) -> ProteinConfig:
    """Mock ProteinConfig pointing to tmp_path for testing file operations."""
    return ProteinConfig(
        protein="test",
        base_map_dir=tmp_path,
        selection="chain A and resi 1-10",
        resolution=2.0,
        map_pattern="{occ_str}.ccp4",
        structure_pattern="{occ_str}.cif",
    )


class TestParseSelectionString:
    """Tests for parse_selection_string function."""

    def test_chain_only(self):
        """Test parsing chain-only selection."""
        result = parse_selection_string("chain A")
        assert result == ("A", None, None)

    def test_single_residue(self):
        """Test parsing single residue selection."""
        result = parse_selection_string("resi 10")
        assert result == (None, 10, 10)

    def test_residue_range(self):
        """Test parsing residue range selection."""
        result = parse_selection_string("resi 10-50")
        assert result == (None, 10, 50)

    def test_chain_and_residue_range(self):
        """Test parsing combined chain and residue range."""
        result = parse_selection_string("chain A and resi 10-50")
        assert result == ("A", 10, 50)

    def test_chain_and_single_residue(self):
        """Test parsing chain and single residue."""
        result = parse_selection_string("chain A and resi 10")
        assert result == ("A", 10, 10)

    def test_case_insensitive(self):
        """Test that parsing is case insensitive."""
        result = parse_selection_string("CHAIN a AND RESI 10")
        assert result == ("A", 10, 10)

    def test_empty_string(self, caplog):
        """Test parsing empty string returns warning and all None."""
        result = parse_selection_string("")
        assert result == (None, None, None)

        assert (
            "Selection string did not match any known patterns (e.g. 'chain A', 'resi 10-50')"
            in caplog.text
        )
        assert caplog.records[0].levelname == "WARNING"

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        result = parse_selection_string("chain  A  and  resi  10-20")
        assert result == ("A", 10, 20)


class TestApplySelection:
    """Tests for apply_selection function."""

    def test_none_selection_returns_unchanged(self, basic_atom_array_multichain):
        """Test that None selection returns original array."""
        result = apply_selection(basic_atom_array_multichain, None)
        assert result is basic_atom_array_multichain
        assert len(result) == len(basic_atom_array_multichain)

    def test_chain_selection(self, basic_atom_array_multichain):
        """Test filtering by chain."""
        result = apply_selection(basic_atom_array_multichain, "chain A")
        assert len(result) == 5
        assert all(result.chain_id == "A")

    def test_residue_selection(self, basic_atom_array_multichain):
        """Test filtering by residue range."""
        result = apply_selection(basic_atom_array_multichain, "resi 2-4")
        assert len(result) == 6
        res_ids = cast(np.ndarray, result.res_id)
        assert all((res_ids >= 2) & (res_ids <= 4))

    def test_single_residue_selection(self, basic_atom_array_multichain):
        """Test filtering by single residue."""
        result = apply_selection(basic_atom_array_multichain, "resi 3")
        assert len(result) == 2
        assert all(result.res_id == 3)

    def test_combined_selection(self, basic_atom_array_multichain):
        """Test filtering by chain and residue."""
        result = apply_selection(basic_atom_array_multichain, "chain A and resi 2-4")
        assert len(result) == 3
        assert all(result.chain_id == "A")
        res_ids = cast(np.ndarray, result.res_id)
        assert all((res_ids >= 2) & (res_ids <= 4))

    def test_no_matching_atoms_raises_valueerror(self, basic_atom_array_multichain):
        """Test that selection matching no atoms raises ValueError."""
        with pytest.raises(ValueError, match="matched no atoms"):
            apply_selection(basic_atom_array_multichain, "chain Z")

    def test_preserves_annotations(self, basic_atom_array_multichain):
        """Test that annotations are preserved after filtering."""
        result = apply_selection(basic_atom_array_multichain, "chain A")
        assert hasattr(result, "chain_id")
        assert hasattr(result, "res_id")
        assert hasattr(result, "atom_name")
        assert list(cast(np.ndarray, result.atom_name)) == ["CA"] * 5

    def test_preserves_coordinates(self, basic_atom_array_multichain):
        """Test that coordinates are preserved correctly."""
        result = apply_selection(basic_atom_array_multichain, "chain A and resi 1")
        result_coord = cast(np.ndarray, result.coord)
        basic_coord = cast(np.ndarray, basic_atom_array_multichain.coord)
        np.testing.assert_array_equal(result_coord[0], basic_coord[0])


class TestExtractSelectionCoordinates:
    """Tests for extract_selection_coordinates function."""

    def test_extracts_coordinates(self, basic_atom_array_multichain):
        """Test that coordinates are extracted correctly."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A")
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (5, 3)

    def test_with_atomarray_stack(self, atom_array_stack_simple):
        """Test extraction from AtomArrayStack uses first model."""
        coords = extract_selection_coordinates(atom_array_stack_simple, "resi 1-3")
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3, 3)
        first_model = cast(AtomArray, atom_array_stack_simple[0])
        np.testing.assert_array_equal(coords, first_model.coord[:3])

    def test_no_matching_atoms_raises_runtime_error(self, basic_atom_array_multichain):
        """Test that no matching atoms raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No atoms matched selection"):
            extract_selection_coordinates(basic_atom_array_multichain, "chain Z")

    def test_filters_nan_coordinates(self, caplog, atom_array_with_nan_coords):
        """Test that NaN coordinates are filtered out with warning."""
        coords = extract_selection_coordinates(atom_array_with_nan_coords, "chain A")
        assert len(coords) == 3
        assert np.isfinite(coords).all()

        assert "Filtered" in caplog.text
        assert "valid atoms remaining" in caplog.text
        assert "atoms with NaN/Inf coordinates" in caplog.text
        assert caplog.records[0].levelname == "WARNING"

    def test_all_nan_raises_runtime_error(self):
        """Test that all invalid coordinates raises RuntimeError."""
        atom_array = AtomArray(3)
        atom_array.coord = np.array([[np.nan, np.nan, np.nan]] * 3)
        atom_array.set_annotation("chain_id", np.array(["A", "A", "A"]))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA"]))

        with pytest.raises(RuntimeError, match="No valid.*finite.*coordinates"):
            extract_selection_coordinates(atom_array, "chain A")

    def test_returns_numpy_array(self, basic_atom_array_multichain):
        """Test that output is numpy array."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A")
        assert isinstance(coords, np.ndarray)
        assert coords.dtype in (np.float32, np.float64)

    def test_coordinate_values(self, basic_atom_array_multichain):
        """Test that coordinate values are correct."""
        coords = extract_selection_coordinates(basic_atom_array_multichain, "chain A and resi 1")
        expected = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(coords, expected)


class TestGetAsymUnitFromStructure:
    """Tests for get_asym_unit_from_structure function."""

    def test_returns_atomarray(self, basic_atom_array_multichain):
        """Test extraction of AtomArray from structure."""
        structure = {"asym_unit": basic_atom_array_multichain}
        result = get_asym_unit_from_structure(structure)
        assert isinstance(result, AtomArray)
        assert result is basic_atom_array_multichain

    def test_returns_atomarraystack(self, atom_array_stack_simple):
        """Test extraction of AtomArrayStack from structure."""
        structure = {"asym_unit": atom_array_stack_simple}
        result = get_asym_unit_from_structure(structure)
        assert isinstance(result, AtomArrayStack)
        assert result is atom_array_stack_simple

    def test_with_atom_array_index(self, atom_array_stack_simple):
        """Test extraction of specific model from stack."""
        structure = {"asym_unit": atom_array_stack_simple}
        result = get_asym_unit_from_structure(structure, atom_array_index=1)
        assert isinstance(result, AtomArray)
        assert len(result) == atom_array_stack_simple.array_length()

    def test_index_on_atomarray_ignored(self, basic_atom_array_multichain):
        """Test that index is ignored for AtomArray."""
        structure = {"asym_unit": basic_atom_array_multichain}
        result = get_asym_unit_from_structure(structure, atom_array_index=0)
        assert isinstance(result, AtomArray)
        assert result is basic_atom_array_multichain

    def test_invalid_type_raises_typeerror(self):
        """Test that invalid type raises TypeError."""
        structure = {"asym_unit": "not an atom array"}
        with pytest.raises(TypeError, match="Unexpected atom array type"):
            get_asym_unit_from_structure(structure)


class TestGetReferenceAtomArrayStack:
    """Tests for get_reference_atomarraystack function."""

    def test_returns_none_when_not_found(self, mock_protein_config):
        """Test that missing file returns (None, None)."""
        path, struct = get_reference_atomarraystack(mock_protein_config, occupancy_a=0.5)
        assert path is None
        assert struct is None

    def test_converts_atomarray_to_stack(self, tmp_path, basic_atom_array_multichain):
        """Test that single AtomArray is converted to stack."""
        from atomworks.io.utils.io_utils import to_cif_file

        structure_path = tmp_path / "0.5occA_0.5occB.cif"
        to_cif_file(basic_atom_array_multichain, structure_path)

        config = ProteinConfig(
            protein="test",
            base_map_dir=tmp_path,
            selection="chain A",
            resolution=2.0,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="{occ_str}.cif",
        )

        path, struct = get_reference_atomarraystack(config, occupancy_a=0.5)
        assert path is not None
        assert isinstance(struct, AtomArrayStack)
        assert struct.stack_depth() == 1

    def test_with_real_structure(self, resources_dir):
        """Test loading real structure with altlocs."""
        config = ProteinConfig(
            protein="6b8x",
            base_map_dir=resources_dir / "6b8x",
            selection="chain A",
            resolution=1.74,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="6b8x_final.pdb",
        )

        path, struct = get_reference_atomarraystack(config, occupancy_a=0.5)
        assert path is not None
        assert struct is not None
        assert isinstance(struct, AtomArrayStack)


class TestGetReferenceStructureCoords:
    """Tests for get_reference_structure_coords function."""

    def test_returns_empty_dict_when_no_valid(self, mock_protein_config):
        """Test that no valid structures returns None."""
        coords = get_reference_structure_coords(mock_protein_config, "test", occ_list=(0.0, 1.0))
        assert coords == {}

    def test_handles_exceptions_gracefully(self, tmp_path):
        """Test that exceptions are logged and function continues."""
        config = ProteinConfig(
            protein="test",
            base_map_dir=tmp_path,
            selection=["chain Z and resi 999",],
            resolution=2.0,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="{occ_str}.cif",
        )

        coords = get_reference_structure_coords(config, "test", occ_list=(0.5,))
        assert coords == {}

    def test_with_real_structure(self, resources_dir):
        """Test loading coords from real structure."""

        selections = ["chain A and resi 1-10", ]
        config = ProteinConfig(
            protein="6b8x",
            base_map_dir=resources_dir / "6b8x",
            selection=selections,
            resolution=1.74,
            map_pattern="{occ_str}.ccp4",
            structure_pattern="6b8x_final.pdb",
        )

        results = get_reference_structure_coords(config, "6b8x", occ_list=(0.5,))
        assert set(results.keys()) == set(selections)
        coords = results[selections[0]]
        if coords is not None:
            assert isinstance(coords, np.ndarray)
            assert coords.ndim == 2
            assert coords.shape[1] == 3
            assert np.isfinite(coords).all()
