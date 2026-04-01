"""Tests for cif_utils module."""

import logging
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import load_any
from biotite.structure import array, Atom, AtomArray, AtomArrayStack
from sampleworks.utils.atom_array_utils import save_structure_to_cif
from sampleworks.utils.cif_utils import resolve_mixed_hetatm_atom_altlocs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atom(chain_id: str, res_id: int, res_name: str, hetero: bool, atom_name: str = "CA") -> Atom:
    return Atom(
        [0.0, 0.0, 0.0],
        chain_id=chain_id,
        res_id=res_id,
        res_name=res_name,
        hetero=hetero,
        atom_name=atom_name,
        element="C",
    )


def _write_cif(atoms: list[Atom], path: Path) -> Path:
    arr = array(atoms)
    arr.set_annotation("occupancy", np.ones(len(atoms), dtype=np.float32))
    arr.set_annotation("b_factor", np.zeros(len(atoms), dtype=np.float32))
    save_structure_to_cif(arr, path)
    return path


def _load(path: Path) -> AtomArray:
    result = load_any(path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(result, AtomArrayStack):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cif_clean(tmp_path) -> Path:
    """CIF with only ATOM records — no mixed ATOM/HETATM at the same position."""
    return _write_cif(
        [
            _atom("A", 99, "VAL", hetero=False),
            _atom("A", 100, "VAL", hetero=False),
            _atom("A", 101, "CYS", hetero=False),
            _atom("A", 102, "ALA", hetero=False),
        ],
        tmp_path / "clean.cif",
    )


@pytest.fixture
def cif_mixed(tmp_path) -> Path:
    """CIF mimicking the 6NI6 bug: CYS (ATOM) and CSO (HETATM) share residue 101."""
    return _write_cif(
        [
            _atom("A", 100, "VAL", hetero=False),
            _atom("A", 101, "CYS", hetero=False),  # canonical — keep
            _atom("A", 101, "CSO", hetero=True),  # modified altloc — remove
            _atom("A", 102, "ALA", hetero=False),
        ],
        tmp_path / "mixed.cif",
    )


@pytest.fixture
def cif_standalone_ligand(tmp_path) -> Path:
    """CIF where a HETATM ligand lives at its own res_id with no overlapping ATOM record."""
    return _write_cif(
        [
            _atom("A", 100, "VAL", hetero=False),
            _atom("A", 101, "CYS", hetero=False),
            _atom("A", 999, "ATP", hetero=True),  # ligand at unique position — untouched
        ],
        tmp_path / "ligand.cif",
    )


@pytest.fixture
def cif_same_resname_hetatm(tmp_path) -> Path:
    """CIF where ATOM and HETATM at the same position share the same residue name."""
    return _write_cif(
        [
            _atom("A", 101, "CYS", hetero=False, atom_name="N"),
            _atom("A", 101, "CYS", hetero=True, atom_name="OG"),  # same name — not our bug
        ],
        tmp_path / "same_resname.cif",
    )


@pytest.fixture
def cif_multiple_mixed(tmp_path) -> Path:
    """CIF with two separate mixed ATOM/HETATM positions on different chains."""
    return _write_cif(
        [
            _atom("A", 101, "CYS", hetero=False),  # chain A pos 101: canonical
            _atom("A", 101, "CSO", hetero=True),  # chain A pos 101: modified
            _atom("B", 50, "SER", hetero=False),  # chain B pos 50: canonical
            _atom("B", 50, "SEP", hetero=True),  # chain B pos 50: phosphoserine
        ],
        tmp_path / "multi_mixed.cif",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveMixedHetatmAtomAltlocs:
    # --- No-op cases ---

    def test_clean_cif_returns_original_path(self, cif_clean):
        assert resolve_mixed_hetatm_atom_altlocs(cif_clean) == cif_clean

    def test_standalone_hetatm_ligand_untouched(self, cif_standalone_ligand):
        """HETATM at a unique res_id (no overlapping ATOM) must not be removed."""
        assert resolve_mixed_hetatm_atom_altlocs(cif_standalone_ligand) == cif_standalone_ligand

    def test_same_resname_hetatm_not_removed(self, cif_same_resname_hetatm):
        """ATOM and HETATM sharing the same residue name at the same position are not our bug."""
        assert resolve_mixed_hetatm_atom_altlocs(cif_same_resname_hetatm) == cif_same_resname_hetatm

    # --- Fix applied ---

    def test_mixed_returns_new_path(self, cif_mixed):
        assert resolve_mixed_hetatm_atom_altlocs(cif_mixed) != cif_mixed

    def test_hetatm_records_removed_at_mixed_position(self, cif_mixed):
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_mixed)
        arr = _load(result_path)
        at_101 = arr[(arr.chain_id == "A") & (arr.res_id == 101)]
        assert not np.any(at_101.hetero)
        assert list(np.unique(at_101.res_name)) == ["CYS"]

    def test_canonical_atom_count_at_mixed_position(self, cif_mixed):
        """Exactly one atom (the CA of CYS) should remain at position 101."""
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_mixed)
        arr = _load(result_path)
        at_101 = arr[(arr.chain_id == "A") & (arr.res_id == 101)]
        assert len(at_101) == 1

    def test_other_residues_unaffected(self, cif_mixed):
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_mixed)
        arr = _load(result_path)
        for rid, expected_name in [(100, "VAL"), (102, "ALA")]:
            res = arr[(arr.chain_id == "A") & (arr.res_id == rid)]
            assert len(res) == 1
            assert res.res_name[0] == expected_name

    def test_multiple_mixed_positions_all_fixed(self, cif_multiple_mixed):
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_multiple_mixed)
        arr = _load(result_path)
        for chain, rid, expected_name in [("A", 101, "CYS"), ("B", 50, "SER")]:
            at_pos = arr[(arr.chain_id == chain) & (arr.res_id == rid)]
            assert not np.any(at_pos.hetero)
            assert list(np.unique(at_pos.res_name)) == [expected_name]

    # --- Warning ---

    def test_warning_logged_for_mixed_position(self, cif_mixed, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_mixed_hetatm_atom_altlocs(cif_mixed)
        assert "101" in caplog.text
        assert "CSO" in caplog.text

    def test_no_warning_for_clean_cif(self, cif_clean, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_mixed_hetatm_atom_altlocs(cif_clean)
        assert "HETATM" not in caplog.text

    def test_warning_per_position_for_multiple_mixed(self, cif_multiple_mixed, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_mixed_hetatm_atom_altlocs(cif_multiple_mixed)
        assert "CSO" in caplog.text
        assert "SEP" in caplog.text

    # --- Real CIF ---

    def test_real_6ni6_cif_is_fixed(self, resources_dir):
        """The 6NI6 density input CIF has CSO (altlocs B/C) and CYS (altloc A) at residue 101."""
        cif_path = resources_dir / "6NI6" / "6NI6_single_001_density_input.cif"
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_path)
        assert result_path != cif_path

    def test_real_6ni6_residue_101_only_cys(self, resources_dir):
        cif_path = resources_dir / "6NI6" / "6NI6_single_001_density_input.cif"
        result_path = resolve_mixed_hetatm_atom_altlocs(cif_path)
        arr = _load(result_path)
        at_101 = arr[(arr.chain_id == "A") & (arr.res_id == 101)]
        assert not np.any(at_101.hetero), "No HETATM records should remain at position 101"
        assert all(n == "CYS" for n in at_101.res_name), "Only CYS should remain at position 101"

    def test_real_6ni6_warning_mentions_residue_and_modified_name(self, resources_dir, caplog):
        cif_path = resources_dir / "6NI6" / "6NI6_single_001_density_input.cif"
        with caplog.at_level(logging.WARNING):
            resolve_mixed_hetatm_atom_altlocs(cif_path)
        assert "101" in caplog.text
        assert "CSO" in caplog.text
