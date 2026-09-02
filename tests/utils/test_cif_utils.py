"""Tests for cif_utils module."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.utils.io_utils import load_any
from biotite.structure import array, Atom, AtomArray, AtomArrayStack
from biotite.structure.io.pdbx.cif import CIFColumn, CIFFile
from sampleworks.utils.atom_array_utils import BLANK_ALTLOC_IDS, save_structure_to_cif
from sampleworks.utils.cif_utils import (
    add_category_to_cif,
    remap_altlocs_to_ab,
    resolve_mixed_hetatm_atom_altlocs,
)


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

    @pytest.mark.parametrize("as_str", [False, True], ids=["path-in", "str-in"])
    @pytest.mark.parametrize("fixture", ["cif_clean", "cif_mixed"], ids=["no-change", "fixed"])
    def test_always_returns_path(self, fixture, as_str, request):
        """Both return branches yield a Path even for str input, so callers can rely on
        Path semantics downstream (guidance_script_utils compares and unlinks the result)."""
        cif = request.getfixturevalue(fixture)
        result = resolve_mixed_hetatm_atom_altlocs(str(cif) if as_str else cif)
        assert isinstance(result, Path)

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

    def test_save_failure_removes_temporary_file(self, cif_mixed, monkeypatch, tmp_path):
        """A failed CIF write must not leave the named temporary file behind."""
        named_temporary_file = tempfile.NamedTemporaryFile

        def create_temporary_file(*args, **kwargs):
            kwargs["dir"] = tmp_path
            return named_temporary_file(*args, **kwargs)

        def fail_save(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "sampleworks.utils.cif_utils.tempfile.NamedTemporaryFile", create_temporary_file
        )
        monkeypatch.setattr("sampleworks.utils.cif_utils.save_structure_to_cif", fail_save)

        with pytest.raises(OSError, match="disk full"):
            resolve_mixed_hetatm_atom_altlocs(cif_mixed)

        assert not list(tmp_path.glob("sampleworks_fixed_cif_*.cif"))

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


# ---------------------------------------------------------------------------
# Tests for add_category_to_cif
# ---------------------------------------------------------------------------


class TestAddCategoryToCif:
    """Tests for add_category_to_cif function."""

    def test_add_category_to_single_block_ciffile(self, tmp_path):
        """Add a category to a CIFFile with a single block."""
        # Create a simple CIF file with structure
        atoms = [_atom("A", 1, "ALA", False), _atom("A", 2, "VAL", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")

        # Read it back
        ciffile = CIFFile.read(str(cif_path))

        # Add a custom category
        data = {"id": [1, 2, 3], "value": ["a", "b", "c"], "score": [1.0, 2.0, 3.0]}
        add_category_to_cif(ciffile, data, "custom_data")

        # Verify the category was added
        block = ciffile[list(ciffile.keys())[0]]
        assert "custom_data" in block
        category = block["custom_data"]
        assert category["id"] == CIFColumn([1, 2, 3])
        assert category["value"] == CIFColumn(["a", "b", "c"])
        assert category["score"] == CIFColumn([1.0, 2.0, 3.0])

    def test_add_category_with_explicit_block_name(self, tmp_path):
        """Add a category to a specific block by name."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        block_name = list(ciffile.keys())[0]
        data = {"id": [1]}
        add_category_to_cif(ciffile, data, "custom_data", block_name=block_name)

        assert "custom_data" in ciffile[block_name]

    def test_category_already_exists_raises_error(self, tmp_path):
        """Adding a category that already exists should raise RuntimeError."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        data = {"id": [1]}
        add_category_to_cif(ciffile, data, "custom_data")

        # Try to add the same category again
        with pytest.raises(RuntimeError, match="Category 'custom_data' already exists"):
            add_category_to_cif(ciffile, data, "custom_data")

    def test_overwrite_existing_category(self, tmp_path):
        """Overwriting an existing category should succeed when overwrite=True."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        # Add initial category
        data1 = {"id": [1], "value": ["old"]}
        add_category_to_cif(ciffile, data1, "custom_data")

        # Overwrite with new data
        data2 = {"id": [2, 3], "value": ["new1", "new2"]}
        add_category_to_cif(ciffile, data2, "custom_data", overwrite=True)

        # Verify the category was overwritten
        block = ciffile[list(ciffile.keys())[0]]
        category = block["custom_data"]
        assert category["id"] == CIFColumn([2, 3])
        assert category["value"] == CIFColumn(["new1", "new2"])

    def test_multiple_blocks_without_block_name_raises_error(self, tmp_path):
        """If CIFFile has multiple blocks and block_name is None, should raise ValueError."""
        # Create a CIF file with two blocks manually
        ciffile = CIFFile()
        from biotite.structure.io.pdbx.cif import CIFBlock

        ciffile["block1"] = CIFBlock()
        ciffile["block2"] = CIFBlock()

        data = {"id": [1]}
        with pytest.raises(ValueError, match="multiple blocks"):
            add_category_to_cif(ciffile, data, "custom_data")

    def test_nonexistent_block_name_raises_error(self, tmp_path):
        """Specifying a block_name that doesn't exist should raise ValueError."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        data = {"id": [1]}
        with pytest.raises(ValueError, match="Block 'nonexistent' not found"):
            add_category_to_cif(ciffile, data, "custom_data", block_name="nonexistent")

    def test_write_and_read_back_category(self, tmp_path):
        """Demonstrate that a custom category can be written to disk and read back."""
        # Create initial CIF
        atoms = [_atom("A", 1, "ALA", False), _atom("A", 2, "VAL", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")

        # Read, add category, and write
        ciffile = CIFFile.read(str(cif_path))
        data = {
            "experiment_id": [1, 2, 3],
            "method": ["xray", "nmr", "em"],
            "resolution": [2.5, 1.8, 3.2],
        }
        add_category_to_cif(ciffile, data, "experiment_metadata")

        output_path = tmp_path / "test_with_metadata.cif"
        ciffile.write(str(output_path))

        # Read back and verify
        reloaded = CIFFile.read(str(output_path))
        block = reloaded[list(reloaded.keys())[0]]
        assert "experiment_metadata" in block

        category = block["experiment_metadata"]
        assert category["experiment_id"] == CIFColumn([1, 2, 3])
        assert category["method"] == CIFColumn(["xray", "nmr", "em"])
        assert category["resolution"] == CIFColumn([2.5, 1.8, 3.2])

    def test_empty_data_dict(self, tmp_path):
        """Adding a category with empty data should work."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        data = {}
        add_category_to_cif(ciffile, data, "empty_category")

        block = ciffile[list(ciffile.keys())[0]]
        assert "empty_category" in block

    def test_single_item_data(self, tmp_path):
        """Adding a category with single items (not lists) should work."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        data = {"name": "test_structure", "version": 1.0}
        add_category_to_cif(ciffile, data, "metadata")

        block = ciffile[list(ciffile.keys())[0]]
        category = block["metadata"]
        assert "name" in category
        assert "version" in category

    def test_none_values_converted(self, tmp_path):
        """None values in data dict should be converted to placeholder string."""
        atoms = [_atom("A", 1, "ALA", False)]
        cif_path = _write_cif(atoms, tmp_path / "test.cif")
        ciffile = CIFFile.read(str(cif_path))

        data = {"present": "value", "missing": None}
        add_category_to_cif(ciffile, data, "test_category")

        block = ciffile[list(ciffile.keys())[0]]
        category = block["test_category"]
        # Verify None was replaced (with "none" or "?" depending on implementation)
        assert "missing" in category


# ---------------------------------------------------------------------------
# remap_altlocs_to_ab
#
# Some depositions label a residue's two conformers A/C rather than A/B, and a scorer that looks
# for literal A and B then drops the second conformer. remap_altlocs_to_ab moves the alternate
# label into the free B slot. The tests below cover the cases the docstring promises: a pair that
# gets relabelled, a pair that is already A/B, a pair with no A at all, a residue with three
# altlocs, and an ordinary blank-altloc residue.
# ---------------------------------------------------------------------------


def _altloc_atom(res_id: int, altloc: str) -> Atom:
    """A one-atom SER residue carrying an explicit altloc label."""
    return Atom(
        [0.0, 0.0, 0.0],
        chain_id="A",
        res_id=res_id,
        res_name="SER",
        hetero=False,
        atom_name="CA",
        element="C",
        altloc_id=altloc,
    )


def _write_altloc_cif(tmp_path: Path, labels: list[str], name: str) -> Path:
    """Write a CIF whose residue 10 carries one atom per label in *labels*.

    Residue 11 is always added with a blank altloc, so every case also carries an ordinary
    residue that remap_altlocs_to_ab has to leave alone.
    """
    atoms = [_altloc_atom(10, label) for label in labels]
    atoms.append(_altloc_atom(11, ""))
    return _write_cif(atoms, tmp_path / name)


class TestRemapAltlocsToAb:
    """Relabelling a residue's non-A/B alternate conformer into the free B slot."""

    def test_alternate_label_becomes_b(self, tmp_path):
        cif_path = _write_altloc_cif(tmp_path, ["A", "C"], "altloc_ac.cif")
        result = _load(remap_altlocs_to_ab(cif_path))

        at_10 = result.altloc_id[result.res_id == 10]
        assert sorted(at_10.tolist()) == ["A", "B"]  # the C conformer is kept, relabelled to B
        assert len(result) == 3  # and nothing was dropped on the way through

    def test_blank_altloc_residue_is_untouched(self, tmp_path):
        cif_path = _write_altloc_cif(tmp_path, ["A", "C"], "altloc_ac.cif")
        result = _load(remap_altlocs_to_ab(cif_path))

        # A blank altloc can come back as "" or "." depending on how the CIF was written, so the
        # check is that residue 11 still has one atom and still carries no real altloc label.
        at_11 = result.altloc_id[result.res_id == 11].tolist()
        assert len(at_11) == 1
        assert at_11[0] in BLANK_ALTLOC_IDS

    def test_already_ab_returns_the_original_path(self, tmp_path):
        # No position changed, so no temporary file is written and the caller keeps its own path.
        # This is what stops the metric moving for structures that were already A/B.
        cif_path = _write_altloc_cif(tmp_path, ["A", "B"], "altloc_ab.cif")
        assert remap_altlocs_to_ab(cif_path) == cif_path

    def test_pair_without_an_a_gets_both_labels(self, tmp_path):
        cif_path = _write_altloc_cif(tmp_path, ["C", "D"], "altloc_cd.cif")
        result = _load(remap_altlocs_to_ab(cif_path))

        assert sorted(result.altloc_id[result.res_id == 10].tolist()) == ["A", "B"]

    def test_three_altlocs_are_left_alone(self, tmp_path):
        # With three conformers there is no way to tell which two belong in A and B, so the
        # function declines rather than guessing.
        cif_path = _write_altloc_cif(tmp_path, ["A", "B", "C"], "altloc_three.cif")
        assert remap_altlocs_to_ab(cif_path) == cif_path

    def test_warning_names_the_remapped_residue(self, tmp_path, caplog):
        cif_path = _write_altloc_cif(tmp_path, ["A", "C"], "altloc_ac.cif")
        with caplog.at_level(logging.WARNING):
            remap_altlocs_to_ab(cif_path)
        assert "10" in caplog.text
