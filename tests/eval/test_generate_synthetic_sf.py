"""Tests for atomarray_to_gemmi in generate_synthetic_sf, using real 6b8x structure."""

from pathlib import Path

import numpy as np
import pytest
import torch


pytest.importorskip("SFC_Torch", reason="SFC_Torch not available; run in analysis-dev-sfc env")

import gemmi
from atomworks.io.transforms.atom_array import remove_waters
from sampleworks.eval.generate_synthetic_sf import atomarray_to_gemmi
from sampleworks.utils.atom_array_utils import (
    assign_occupancies,
    detect_altlocs,
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
    s = gemmi.read_structure(str(resources_dir / "6b8x" / "6b8x_final.pdb"))
    s.remove_hydrogens()
    s.remove_ligands_and_waters()
    return s


@pytest.fixture(scope="module")
def stripped_atom_array(resources_dir: Path):
    arr = load_structure_with_altlocs(resources_dir / "6b8x" / "6b8x_final.pdb")
    arr = remove_hydrogens(arr)
    arr = remove_waters(arr)
    return keep_polymer(keep_amino_acids(arr))


@pytest.fixture(scope="module")
def gemmi_structure_from_atomarray(
    stripped_atom_array, stripped_gemmi: gemmi.Structure
) -> gemmi.Structure:
    return atomarray_to_gemmi(
        stripped_atom_array, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm
    )


def _compute_fprotein(gemmi_structure: gemmi.Structure, device: torch.device) -> np.ndarray:
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
    def test_cell_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        result = gemmi_structure_from_atomarray.cell
        expected = stripped_gemmi.cell
        assert result.a == pytest.approx(expected.a)
        assert result.b == pytest.approx(expected.b)
        assert result.c == pytest.approx(expected.c)
        assert result.alpha == pytest.approx(expected.alpha)
        assert result.beta == pytest.approx(expected.beta)
        assert result.gamma == pytest.approx(expected.gamma)

    def test_space_group_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        assert gemmi_structure_from_atomarray.spacegroup_hm == stripped_gemmi.spacegroup_hm

    def test_atom_count_matches_pdb(self, gemmi_structure_from_atomarray, stripped_gemmi):
        assert (
            gemmi_structure_from_atomarray[0].count_atom_sites()
            == stripped_gemmi[0].count_atom_sites()
        )

    def test_occupancy_change_is_applied(self, stripped_atom_array, stripped_gemmi):
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
        f_atomarray = _compute_fprotein(gemmi_structure_from_atomarray, device)
        f_direct = _compute_fprotein(stripped_gemmi, device)
        np.testing.assert_allclose(np.abs(f_atomarray), np.abs(f_direct), atol=1e-3)

    def test_fprotein_changes_with_occupancy(self, stripped_atom_array, stripped_gemmi, device):
        altloc_info = detect_altlocs(stripped_atom_array)

        arr_default = assign_occupancies(stripped_atom_array, altloc_info, "default")
        f_default = _compute_fprotein(
            atomarray_to_gemmi(arr_default, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        arr_custom = assign_occupancies(stripped_atom_array, altloc_info, "custom", [0.2, 0.8, 0.0])
        f_custom = _compute_fprotein(
            atomarray_to_gemmi(arr_custom, stripped_gemmi.cell, stripped_gemmi.spacegroup_hm),
            device,
        )

        assert not np.allclose(np.abs(f_default), np.abs(f_custom), atol=1e-3)
