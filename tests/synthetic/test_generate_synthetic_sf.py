"""Tests for atomarray_to_gemmi in synthetic_utils, using real 6b8x structure."""

import logging
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from sampleworks.synthetic.synthetic_utils import assign_occupancies, atomarray_to_gemmi
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
        # atom order is preserved: biotite keeps PDB file order, and atomarray_to_gemmi
        # emits atoms in that same order
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

    def test_saved_structure_round_trips_annotations(
        self, gemmi_structure_from_atomarray, stripped_atom_array, tmp_path
    ):
        """Every per-atom field must survive atomarray_to_gemmi --> cif --> atomarray.

        One reload guarding each column a write can silently drop/garble. Scattering-physics
        fields (coords, element, b_factor, occupancy) would corrupt structure factors without
        raising if lost; topology fields (chain_id, res_id, res_name, atom_name) drive residue/
        altloc matching and reconciliation. res_id and label_seq_id are the field that regressed:
        array2hier set only the author seqid, so label_seq_id wrote as "." and atomworks (label
        scheme) collapsed every residue to -1 on re-read.
        """
        out = tmp_path / "saved.cif"
        gemmi_structure_from_atomarray.make_mmcif_document().write_file(str(out))

        loaded = load_structure_with_altlocs(out)
        ref = stripped_atom_array

        assert len(loaded) == len(ref)

        # topology / matching labels (exact)
        assert np.array_equal(loaded.chain_id, ref.chain_id)
        assert np.array_equal(loaded.res_id, ref.res_id)
        assert set(np.unique(loaded.res_id)) != {-1}  # not collapsed to the degenerate -1
        assert np.array_equal(loaded.res_name, ref.res_name)
        assert np.array_equal(loaded.atom_name, ref.atom_name)

        # scattering-physics fields (element exact; floats within mmCIF write precision)
        assert np.array_equal(loaded.element, ref.element)
        assert len(np.unique(loaded.element)) > 1  # not collapsed to a single/blank symbol
        np.testing.assert_allclose(loaded.coord, ref.coord, atol=1e-3)
        np.testing.assert_allclose(loaded.b_factor, ref.b_factor, atol=1e-2)
        np.testing.assert_allclose(loaded.occupancy, ref.occupancy, atol=1e-2)

        # altloc: same id set, and same per-id atom counts
        assert "altloc_id" in loaded.get_annotation_categories()
        expected_altloc_ids = find_all_altloc_ids(ref)
        assert find_all_altloc_ids(loaded) == expected_altloc_ids
        for altloc_id in expected_altloc_ids:
            assert np.count_nonzero(loaded.altloc_id == altloc_id) == np.count_nonzero(
                ref.altloc_id == altloc_id
            )

    def test_multichain_shared_res_ids_survive_round_trip(self, tmp_path):
        """Two chains that share res_ids must not be merged across the chain boundary.

        array2hier keyed residue boundaries on res_id alone, so a chain boundary where both
        sides share a res_id (chains independently numbered from 1) merged the two residues.
        The direct builder keys on (chain_id, res_id); this guards that.
        """
        # Two chains A and B, each with residues 1 and 2 (overlapping numbering).
        n = 4
        arr = AtomArray(n)
        arr.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
        arr.chain_id = np.array(["A", "A", "B", "B"])
        arr.res_id = np.array([1, 2, 1, 2])
        arr.res_name = np.array(["ALA", "GLY", "ALA", "GLY"])
        arr.atom_name = np.array(["CA", "CA", "CA", "CA"])
        arr.element = np.array(["C", "C", "C", "C"])
        arr.set_annotation("b_factor", np.full(n, 20.0))
        arr.set_annotation("occupancy", np.ones(n))

        out = tmp_path / "multichain.cif"
        atomarray_to_gemmi(arr).make_mmcif_document().write_file(str(out))
        loaded = load_structure_with_altlocs(out)

        assert len(loaded) == n
        assert np.array_equal(loaded.chain_id, arr.chain_id)
        assert np.array_equal(loaded.res_id, arr.res_id)
        # distinct per-atom coords pin identity+order: a boundary merge would drop/reorder atoms
        np.testing.assert_allclose(loaded.coord, arr.coord, atol=1e-3)

    def test_empty_atom_array_raises(self):
        """An empty AtomArray fails fast rather than yielding a chain-less structure."""
        with pytest.raises(ValueError, match="empty AtomArray"):
            atomarray_to_gemmi(AtomArray(0))

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
