"""Tests for atomarray_to_gemmi in generate_synthetic_sf, using real 6b8x structure."""

import logging
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from sampleworks.eval.generate_synthetic_sf import atomarray_to_gemmi
from sampleworks.eval.synthetic_utils import assign_occupancies
from sampleworks.utils.atom_array_utils import (
    detect_altlocs,
    find_all_altloc_ids,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    remove_hydrogens,
)
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser
from SFC_Torch.utils import assert_numpy


DMIN = 2.0


@pytest.fixture(scope="module")
def stripped_gemmi(resources_dir: Path) -> gemmi.Structure:
    """gemmi.Structure with hydrogens, ligands, and waters removed using gemmi methods."""
    s = gemmi.read_structure(str(resources_dir / "6b8x" / "6b8x_final.pdb"))
    s.remove_hydrogens()
    s.remove_ligands_and_waters()
    return s


@pytest.fixture(scope="module")
def stripped_atom_array(resources_dir: Path) -> AtomArray:
    """AtomArray with hydrogens, waters, and non-polymer/non-amino-acid atoms
    removed using existing utils."""
    arr = load_structure_with_altlocs(resources_dir / "6b8x" / "6b8x_final.pdb")
    arr = remove_hydrogens(arr)
    arr = remove_waters(arr)
    arr = keep_polymer(keep_amino_acids(arr))
    assert isinstance(arr, AtomArray)
    return arr


@pytest.fixture(scope="module")
def gemmi_structure_from_atomarray(
    stripped_atom_array, stripped_gemmi: gemmi.Structure
) -> gemmi.Structure:
    """gemmi.Structure converted from the stripped AtomArray."""
    return atomarray_to_gemmi(
        stripped_atom_array, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm
    )


def _compute_fprotein(gemmi_structure: gemmi.Structure, device: torch.device) -> np.ndarray:
    """Compute |Fprotein| amplitudes from a gemmi structure via SFcalculator at ``DMIN``
    resolution. The final assert_numpy converts any tensor or list to a numpy array."""
    sfc = SFcalculator(
        PDBParser(gemmi_structure),
        mtzdata=None,
        dmin=DMIN,
        mode="xray",
        anomalous=False,
        set_experiment=False,
        device=device,
    )
    sfc.calc_fprotein()
    return assert_numpy(sfc.Fprotein_asu)


class TestAtomArrayToGemmi:
    """Tests for atomarray_to_gemmi using the 6b8x structure."""

    def test_cell_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Unit cell parameters are preserved through the biotite→gemmi conversion."""
        result = gemmi_structure_from_atomarray.cell
        expected = stripped_gemmi.cell
        assert result.a == pytest.approx(expected.a)
        assert result.b == pytest.approx(expected.b)
        assert result.c == pytest.approx(expected.c)
        assert result.alpha == pytest.approx(expected.alpha)
        assert result.beta == pytest.approx(expected.beta)
        assert result.gamma == pytest.approx(expected.gamma)

    def test_space_group_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Space group is preserved through the biotite→gemmi conversion."""
        assert gemmi_structure_from_atomarray.spacegroup_hm == stripped_gemmi.spacegroup_hm

    def test_atoms_match_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        """Atom names and positions match the original PDB, in the same order."""
        # atom order is preserved: biotite keeps PDB file order, array2hier reconstructs it
        parser_from_atomarray = PDBParser(gemmi_structure_from_atomarray)
        parser_from_gemmi = PDBParser(stripped_gemmi)
        assert np.array_equal(parser_from_atomarray.atom_name, parser_from_gemmi.atom_name)
        np.testing.assert_allclose(
            parser_from_atomarray.atom_pos, parser_from_gemmi.atom_pos, atol=1e-3
        )

    def test_occupancy_change_is_applied(self, stripped_atom_array, stripped_gemmi):
        """Custom occupancy values are correctly written to each altloc group."""
        occ_values = [0.2, 0.8, 0.0]
        altloc_info = detect_altlocs(stripped_atom_array)
        arr = assign_occupancies(stripped_atom_array, altloc_info, "custom", occ_values)
        parser = PDBParser(
            atomarray_to_gemmi(arr, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm)
        )
        for altloc, expected in zip(altloc_info.altloc_ids, occ_values):
            assert np.allclose(parser.atom_occ[altloc_info.atom_masks[altloc]], expected)

    def test_fprotein_matches_direct_gemmi(
        self, gemmi_structure_from_atomarray, stripped_gemmi, device
    ):
        """Fprotein amplitudes from the converted structure match those from
        the original gemmi structure."""
        f_atomarray = _compute_fprotein(gemmi_structure_from_atomarray, device)
        f_direct = _compute_fprotein(stripped_gemmi, device)
        np.testing.assert_allclose(np.abs(f_atomarray), np.abs(f_direct), atol=1e-3)

    def test_saved_structure_loads_back_with_altlocs(
        self, gemmi_structure_from_atomarray, stripped_atom_array, tmp_path
    ):
        """Altloc labels must survive the round trip atomarray_to_gemmi --> cif
        --> atomarray. Regression guard."""
        out = tmp_path / "saved.cif"
        gemmi_structure_from_atomarray.make_mmcif_document().write_file(str(out))

        loaded = load_structure_with_altlocs(out)

        assert len(loaded) == len(stripped_atom_array)
        assert "altloc_id" in loaded.get_annotation_categories()
        # check the set of altloc ids is the same
        expected_altloc_ids = find_all_altloc_ids(stripped_atom_array)
        assert find_all_altloc_ids(loaded) == expected_altloc_ids
        # check on a per-atom basis for source altloc id survival
        for altloc_id in expected_altloc_ids:
            assert np.count_nonzero(loaded.altloc_id == altloc_id) == np.count_nonzero(
                stripped_atom_array.altloc_id == altloc_id
            )

    def test_occupancy_warns_on_extra_values(self, stripped_atom_array, caplog):
        """A warning is logged when more occupancy values are provided than there are altlocs."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with caplog.at_level(logging.WARNING):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.2, 0.8, 0.0, 0.0])
        assert "Extra values will be ignored" in caplog.text

    def test_occupancy_warns_on_missing_values(self, stripped_atom_array, caplog):
        """A warning is logged when fewer occupancy values are provided than there are altlocs."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with caplog.at_level(logging.WARNING):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.5, 0.5])
        assert "Missing values are automatically set to 0" in caplog.text

    def test_occupancy_raises_on_out_of_range(self, stripped_atom_array):
        """ValueError is raised when an occupancy value is outside [0.0, 1.0]."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with pytest.raises(ValueError, match="out of range"):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [1.5, 0.0, 0.0])

    def test_occupancy_raises_on_bad_sum(self, stripped_atom_array):
        """ValueError is raised when occupancy values do not sum to 1.0."""
        altloc_info = detect_altlocs(stripped_atom_array)
        with pytest.raises(ValueError, match="sum to 1.0"):
            assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.3, 0.3, 0.3])

    def test_fprotein_changes_with_occupancy(self, stripped_atom_array, stripped_gemmi, device):
        """Fprotein amplitudes differ when occupancies changes from uniform to custom values."""
        altloc_info = detect_altlocs(stripped_atom_array)

        arr_uniform = assign_occupancies(stripped_atom_array, altloc_info, "uniform")
        f_uniform = _compute_fprotein(
            atomarray_to_gemmi(arr_uniform, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        arr_custom = assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.2, 0.8, 0.0])
        f_custom = _compute_fprotein(
            atomarray_to_gemmi(arr_custom, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        assert not np.allclose(np.abs(f_uniform), np.abs(f_custom), atol=1e-3)
