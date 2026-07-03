"""Tests for the reproducible input-preparation module."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from sampleworks.utils.cif_utils import add_category_to_cif, read_category_from_cif
from sampleworks.utils.input_prep import (
    carve,
    CarveSpec,
    load_registry,
    prepare_from_registry,
    prepare_input,
)


# Resolved without the tests/conftest.py fixtures so these tests run under the torch-free
# ``--confcutdir=tests/utils`` invocation as well as on the full environment.
RESOURCES = Path(__file__).resolve().parent.parent / "resources"


# ---------------------------------------------------------------------------
# Synthetic-deposit builders
# ---------------------------------------------------------------------------

_ATOM_COLS = [
    "group_PDB",
    "type_symbol",
    "label_atom_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "auth_asym_id",
    "auth_seq_id",
    "label_alt_id",
    "pdbx_PDB_model_num",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
]


def _row(
    chain: str,
    entity: str,
    auth: int,
    comp: str,
    *,
    label: int | None = None,
    atom: str = "CA",
    elem: str = "C",
    group: str = "ATOM",
    alt: str = ".",
    model: str = "1",
) -> dict:
    return {
        "group_PDB": group,
        "type_symbol": elem,
        "label_atom_id": atom,
        "label_comp_id": comp,
        "label_asym_id": chain,
        "label_entity_id": entity,
        "label_seq_id": str(label if label is not None else auth),
        "auth_asym_id": chain,
        "auth_seq_id": str(auth),
        "label_alt_id": alt,
        "pdbx_PDB_model_num": model,
        "Cartn_x": "0.000",
        "Cartn_y": "0.000",
        "Cartn_z": "0.000",
        "occupancy": "1.00",
        "B_iso_or_equiv": "0.00",
    }


def _atom_site_cols(rows: list[dict]) -> dict[str, list[str]]:
    return {col: [r[col] for r in rows] for col in _ATOM_COLS}


def _deposit(
    rows: list[dict],
    entity: dict,
    entity_poly: dict,
    entity_poly_seq: dict,
    *,
    struct_conn: dict | None = None,
    cell: dict | None = None,
) -> CIFFile:
    """Build a minimal but realistic RCSB-style deposit CIFFile."""
    cif = CIFFile()
    cif["dep"] = CIFBlock()
    cif["dep"]["atom_site"] = CIFCategory(columns=_atom_site_cols(rows), name="atom_site")
    add_category_to_cif(cif, entity, "entity", block_name="dep")
    add_category_to_cif(cif, entity_poly, "entity_poly", block_name="dep")
    add_category_to_cif(cif, entity_poly_seq, "entity_poly_seq", block_name="dep")
    if struct_conn is not None:
        add_category_to_cif(cif, struct_conn, "struct_conn", block_name="dep")
    if cell is not None:
        add_category_to_cif(cif, cell, "cell", block_name="dep")
    return cif


# A single-chain deposit: polymer chain A (author numbering 9..16, with a selenomethionine and a
# disulfide), plus a water and a ligand on the same author chain (separate entities). The polymer
# block is the whole entity, so reconcile aligns uniquely.
_CHAIN_A_RESIDUES = [
    (9, "SER"),
    (10, "THR"),
    (11, "MSE"),  # modified polymer residue (HETATM) -- must be kept
    (12, "TYR"),
    (13, "CYS"),
    (14, "GLY"),
    (15, "ALA"),
    (16, "CYS"),
]


def _single_chain_deposit() -> CIFFile:
    rows = []
    for canonical, (auth, comp) in enumerate(_CHAIN_A_RESIDUES, start=1):
        group = "HETATM" if comp == "MSE" else "ATOM"
        rows.append(_row("A", "1", auth, comp, label=canonical, group=group))
    # water + ligand on the same author chain, dropped by the entity-aware filter
    rows.append(_row("A", "2", 201, "HOH", atom="O", elem="O", group="HETATM"))
    rows.append(_row("A", "3", 301, "ATP", atom="PA", elem="P", group="HETATM"))

    entity = {
        "id": ["1", "2", "3"],
        "type": ["polymer", "water", "non-polymer"],
    }
    entity_poly = {
        "entity_id": ["1"],
        "type": ["polypeptide(L)"],
        "pdbx_strand_id": ["A"],
    }
    entity_poly_seq = {
        "entity_id": ["1"] * 8,
        "num": [str(i) for i in range(1, 9)],
        "mon_id": [comp for _, comp in _CHAIN_A_RESIDUES],
        "hetero": ["n"] * 8,
    }
    struct_conn = {
        "id": ["disulf1", "covale1"],
        "conn_type_id": ["disulf", "covale"],
        "ptnr1_auth_asym_id": ["A", "A"],
        "ptnr1_auth_seq_id": ["13", "13"],
        "ptnr1_label_comp_id": ["CYS", "CYS"],
        "ptnr2_auth_asym_id": ["A", "A"],
        "ptnr2_auth_seq_id": ["16", "301"],  # second row points at the dropped ligand
        "ptnr2_label_comp_id": ["CYS", "ATP"],
    }
    cell = {
        "length_a": ["40.000"],
        "length_b": ["50.000"],
        "length_c": ["60.000"],
        "angle_alpha": ["90.00"],
        "angle_beta": ["90.00"],
        "angle_gamma": ["90.00"],
    }
    return _deposit(rows, entity, entity_poly, entity_poly_seq, struct_conn=struct_conn, cell=cell)


def _two_chain_deposit() -> CIFFile:
    """Two distinct polymer entities on chains A and B."""
    rows = []
    for canonical, (auth, comp) in enumerate([(1, "SER"), (2, "THR"), (3, "TYR")], start=1):
        rows.append(_row("A", "1", auth, comp, label=canonical))
    for canonical, (auth, comp) in enumerate([(1, "ALA"), (2, "GLY"), (3, "LEU")], start=1):
        rows.append(_row("B", "2", auth, comp, label=canonical))
    entity = {"id": ["1", "2"], "type": ["polymer", "polymer"]}
    entity_poly = {
        "entity_id": ["1", "2"],
        "type": ["polypeptide(L)", "polypeptide(L)"],
        "pdbx_strand_id": ["A", "B"],
    }
    entity_poly_seq = {
        "entity_id": ["1", "1", "1", "2", "2", "2"],
        "num": ["1", "2", "3", "1", "2", "3"],
        "mon_id": ["SER", "THR", "TYR", "ALA", "GLY", "LEU"],
        "hetero": ["n"] * 6,
    }
    return _deposit(rows, entity, entity_poly, entity_poly_seq)


def _block(cif: CIFFile) -> CIFBlock:
    return cif[list(cif.keys())[0]]


def _column(cif: CIFFile, category: str, key: str) -> list[str]:
    return [str(v) for v in _block(cif)[category][key].as_array(str)]


# ---------------------------------------------------------------------------
# CarveSpec
# ---------------------------------------------------------------------------


class TestCarveSpec:
    def test_spec_hash_is_chain_order_independent(self):
        a = CarveSpec(chains=["A", "B"])
        b = CarveSpec(chains=["B", "A"])
        assert a.spec_hash() == b.spec_hash()

    def test_spec_hash_changes_with_chains_drop_numbering(self):
        base = CarveSpec(chains=["A"]).spec_hash()
        assert CarveSpec(chains=["A", "B"]).spec_hash() != base
        assert CarveSpec(chains=["A"], drop=frozenset({"water"})).spec_hash() != base
        assert CarveSpec(chains=["A"], numbering="from_one").spec_hash() != base

    def test_roundtrip_dict(self):
        spec = CarveSpec(chains=["B"], drop=frozenset({"water"}), numbering="from_one")
        restored = CarveSpec.from_dict(spec.to_dict())
        assert restored.spec_hash() == spec.spec_hash()

    def test_from_dict_defaults(self):
        spec = CarveSpec.from_dict({"chains": ["A"]})
        assert spec.drop == frozenset({"water", "ligand"})
        assert spec.numbering == "preserve_auth"

    def test_invalid_numbering_rejected(self):
        with pytest.raises(ValueError, match="numbering"):
            CarveSpec(chains=["A"], numbering="bogus")

    def test_invalid_drop_token_rejected(self):
        with pytest.raises(ValueError, match="drop"):
            CarveSpec(chains=["A"], drop=frozenset({"hydrogens"}))

    def test_empty_chains_rejected(self):
        with pytest.raises(ValueError, match="chain"):
            CarveSpec(chains=[])

    def test_selection_not_implemented(self):
        with pytest.raises(NotImplementedError):
            carve(_single_chain_deposit(), CarveSpec(chains=["A"], selection="chain A"))


# ---------------------------------------------------------------------------
# carve()
# ---------------------------------------------------------------------------


class TestCarveFilter:
    def test_drops_water_and_ligand_keeps_modified_residue(self):
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        comps = _column(out, "atom_site", "label_comp_id")
        assert comps == [comp for _, comp in _CHAIN_A_RESIDUES]
        assert "MSE" in comps  # selenomethionine kept
        assert "HOH" not in comps and "ATP" not in comps

    def test_chain_selection(self):
        out = carve(_two_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        assert set(_column(out, "atom_site", "auth_asym_id")) == {"A"}
        assert _column(out, "atom_site", "label_comp_id") == ["SER", "THR", "TYR"]

    def test_multi_chain_gets_distinct_entities(self):
        out = carve(_two_chain_deposit(), CarveSpec(chains=["A", "B"]), source="t")
        assert set(_column(out, "atom_site", "auth_asym_id")) == {"A", "B"}
        # each kept chain carried under its own entity id
        assert set(_column(out, "atom_site", "label_entity_id")) == {"1", "2"}

    def test_preserve_auth_numbering(self):
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        assert _column(out, "atom_site", "auth_seq_id") == [str(a) for a, _ in _CHAIN_A_RESIDUES]
        assert _column(out, "atom_site", "label_seq_id") == _column(out, "atom_site", "auth_seq_id")

    def test_from_one_numbering(self):
        out = carve(
            _single_chain_deposit(), CarveSpec(chains=["A"], numbering="from_one"), source="t"
        )
        assert _column(out, "atom_site", "auth_seq_id") == [str(i) for i in range(1, 9)]
        assert _column(out, "atom_site", "label_seq_id") == [str(i) for i in range(1, 9)]

    def test_entity_stub_present_no_sequence_carry(self):
        # The input carries only the validator's stub _entity (parent of label_entity_id); the real
        # entity sequence (_entity_poly_seq) belongs to the output CIF, not the input.
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        block = _block(out)
        assert "entity" in block
        assert set(_column(out, "entity", "id")) == {"1"}
        assert "entity_poly_seq" not in block
        assert "entity_poly" not in block

    def test_cell_carried(self):
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        assert "cell" in _block(out)
        assert _column(out, "cell", "length_a") == ["40.000"]

    def test_struct_conn_subset_keeps_disulfide_drops_ligand_conn(self):
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        assert _column(out, "struct_conn", "conn_type_id") == ["disulf"]
        assert _column(out, "struct_conn", "ptnr2_auth_seq_id") == ["16"]

    def test_struct_conn_remapped_under_from_one(self):
        out = carve(
            _single_chain_deposit(), CarveSpec(chains=["A"], numbering="from_one"), source="t"
        )
        # CYS 13 -> 5, CYS 16 -> 8 under 1..N renumbering
        assert _column(out, "struct_conn", "ptnr1_auth_seq_id") == ["5"]
        assert _column(out, "struct_conn", "ptnr2_auth_seq_id") == ["8"]

    def test_completeness_parents_present(self):
        out = carve(_single_chain_deposit(), CarveSpec(chains=["A"]), source="t")
        block = _block(out)
        assert "atom_type" in block and "chem_comp" in block
        # _chem_comp lists every residue the carried sequence references
        assert set(_column(out, "chem_comp", "id")) >= {comp for _, comp in _CHAIN_A_RESIDUES}

    def test_provenance_written(self):
        spec = CarveSpec(chains=["A"])
        out = carve(_single_chain_deposit(), spec, source="1abc")
        prov = read_category_from_cif(out, "sampleworks")
        assert prov["sampleworks_carve_source"] == "1abc"
        assert prov["sampleworks_carve_chains"] == "A"
        assert prov["sampleworks_carve_drop"] == "ligand,water"
        assert prov["sampleworks_carve_numbering"] == "preserve_auth"
        assert prov["sampleworks_carve_spec_hash"] == spec.spec_hash()

    def test_nondefault_drop_not_supported(self):
        # Retaining ligands/waters is future work; the carve must refuse rather than silently drop.
        spec = CarveSpec(chains=["A"], drop=frozenset({"water"}))
        with pytest.raises(NotImplementedError):
            carve(_single_chain_deposit(), spec, source="t")

    def test_unknown_chain_raises(self):
        with pytest.raises(ValueError, match="chain"):
            carve(_single_chain_deposit(), CarveSpec(chains=["Z"]), source="t")


# ---------------------------------------------------------------------------
# prepare_input() + registry
# ---------------------------------------------------------------------------


class TestPrepareInput:
    def test_cache_hit_skips_fetch(self, tmp_path, monkeypatch):
        deposit_path = tmp_path / "deposit.cif"
        _single_chain_deposit().write(str(deposit_path))

        calls = {"n": 0}

        def _fake_fetch(pdb_id, *args, **kwargs):
            calls["n"] += 1
            return deposit_path

        monkeypatch.setattr("sampleworks.utils.input_prep.fetch_rcsb_cif", _fake_fetch)

        spec = CarveSpec(chains=["A"])
        cache = tmp_path / "inputs"
        first = prepare_input("1abc", spec, cache_dir=cache)
        assert first.exists()
        assert calls["n"] == 1

        second = prepare_input("1abc", spec, cache_dir=cache)
        assert second == first
        assert calls["n"] == 1  # served from cache, no re-fetch

    def test_cache_key_uses_spec_hash(self, tmp_path, monkeypatch):
        deposit_path = tmp_path / "deposit.cif"
        _single_chain_deposit().write(str(deposit_path))
        monkeypatch.setattr(
            "sampleworks.utils.input_prep.fetch_rcsb_cif", lambda *a, **k: deposit_path
        )
        cache = tmp_path / "inputs"
        a = prepare_input("1abc", CarveSpec(chains=["A"]), cache_dir=cache)
        b = prepare_input("1abc", CarveSpec(chains=["A"], numbering="from_one"), cache_dir=cache)
        assert a != b
        assert a.exists() and b.exists()


class TestLoadRegistry:
    def test_load_registry(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "4OLE": {"chains": ["B"], "drop": ["water", "ligand"]},
                    "1abc": {"chains": ["A"], "numbering": "from_one"},
                }
            )
        )
        registry = load_registry(path)
        assert set(registry) == {"4ole", "1abc"}  # keys lower-cased
        assert registry["4ole"].chains == ["B"]
        assert registry["1abc"].numbering == "from_one"


# ---------------------------------------------------------------------------
# Reproduction of in-repo known inputs (carve from the deposit, compare invariants)
# ---------------------------------------------------------------------------

_HYDROGEN = {"H", "D"}

# (label, deposit source, spec, the in-repo carved input to reproduce). 2YL0/9BN8 use the deposit
# author numbering; 1vme used the deposit canonical (label) numbering -- a real, previously
# undeclared inconsistency the registry now records.
_KNOWNS = [
    (
        "2yl0",
        "2YL0/2YL0_rcsb.cif",
        CarveSpec(chains=["A"], numbering="preserve_auth"),
        "2YL0/2YL0_single_001_density_input.cif",
    ),
    (
        "9bn8",
        "9BN8/9BN8_rcsb.cif",
        CarveSpec(chains=["A"], numbering="preserve_auth"),
        "9BN8/9BN8_single_001_density_input.cif",
    ),
    (
        "1vme",
        "1vme/1vme_final.cif",
        CarveSpec(chains=["A"], numbering="preserve_label"),
        "1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif",
    ),
]


def _residue_view(cif, chain: str = "A") -> dict[str, tuple[str, frozenset[str]]]:
    """Map ``auth_seq_id -> (resname, frozenset of heavy-atom names)`` for a chain."""
    import numpy as np

    src = cif if isinstance(cif, CIFFile) else CIFFile.read(str(cif))
    block = _block(src)
    a = block["atom_site"]
    cols = {k: np.asarray(a[k].as_array(str)) for k in a}
    chain_col = "auth_asym_id" if "auth_asym_id" in cols else "label_asym_id"
    mask = cols[chain_col] == chain
    resname: dict[str, str] = {}
    atoms: dict[str, set[str]] = {}
    for i in np.nonzero(mask)[0]:
        if cols["type_symbol"][i] in _HYDROGEN:
            continue
        seq = str(cols["auth_seq_id"][i])
        resname[seq] = str(cols["label_comp_id"][i])
        atoms.setdefault(seq, set()).add(str(cols["label_atom_id"][i]))
    return {seq: (resname[seq], frozenset(atoms[seq])) for seq in resname}


class TestReproduceKnownInputs:
    @pytest.mark.parametrize("label, source, spec, target", _KNOWNS, ids=[k[0] for k in _KNOWNS])
    def test_chain_numbering_resname_match(self, label, source, spec, target):
        out = carve(RESOURCES / source, spec, source=label)
        got = _residue_view(out)
        want = _residue_view(RESOURCES / target)
        # same residues, same numbering, same residue identity (incl. modified residues)
        assert set(got) == set(want)
        assert {s: rn for s, (rn, _) in got.items()} == {s: rn for s, (rn, _) in want.items()}

    @pytest.mark.parametrize(
        "label, source, spec, target",
        [k for k in _KNOWNS if k[0] in ("2yl0", "9bn8")],
        ids=["2yl0", "9bn8"],
    )
    def test_heavy_atom_content_matches_clean_inputs(self, label, source, spec, target):
        # The two density inputs are clean carves (no hand-editing), so heavy-atom content matches
        # exactly. (1vme's target was hand-edited at terminal/altloc atoms, so it is excluded here;
        # its residue identity is covered by the test above.)
        out = carve(RESOURCES / source, spec, source=label)
        got = _residue_view(out)
        want = _residue_view(RESOURCES / target)
        shared = set(got) & set(want)
        mismatches = {s: got[s][1] ^ want[s][1] for s in shared if got[s][1] != want[s][1]}
        assert mismatches == {}

    def test_modified_residue_preserved(self):
        # 2YL0 keeps its N-terminal pyroglutamate (PCA); 1vme keeps selenomethionine (MSE).
        pca = carve(RESOURCES / "2YL0/2YL0_rcsb.cif", _KNOWNS[0][2], source="2yl0")
        assert "PCA" in _column(pca, "atom_site", "label_comp_id")
        mse = carve(RESOURCES / "1vme/1vme_final.cif", _KNOWNS[2][2], source="1vme")
        assert "MSE" in _column(mse, "atom_site", "label_comp_id")


class TestParseAcceptance:
    """The model-agnostic gate: atomworks must parse the carve into the right polymer sequence."""

    @pytest.mark.parametrize("label, source, spec, target", _KNOWNS, ids=[k[0] for k in _KNOWNS])
    def test_atomworks_parses_polymer(self, label, source, spec, target, tmp_path):
        parse = pytest.importorskip("atomworks").parse
        out = carve(RESOURCES / source, spec, source=label)
        out_path = tmp_path / f"{label}.cif"
        out.write(str(out_path))
        structure = parse(str(out_path), hydrogen_policy="remove", add_missing_atoms=False)
        info = structure["chain_info"]["A"]
        seq = info["processed_entity_canonical_sequence"]
        assert seq and seq.isalpha()  # one canonical polymer sequence for the kept chain
        if label == "1vme":
            assert "X" not in seq  # selenomethionine canonicalized to M, not unknown


# ---------------------------------------------------------------------------
# prepare_from_registry() + the production registry
# ---------------------------------------------------------------------------


class TestPrepareFromRegistry:
    def _registry(self, tmp_path) -> Path:
        path = tmp_path / "reg.json"
        path.write_text(
            json.dumps(
                {
                    "1abc": {"chains": ["A"], "numbering": "preserve_auth"},
                    "2def": {"chains": ["A"], "numbering": "from_one"},
                }
            )
        )
        return path

    def _patch_fetch(self, tmp_path, monkeypatch) -> list[str]:
        deposit = tmp_path / "deposit.cif"
        _single_chain_deposit().write(str(deposit))
        calls: list[str] = []

        def _fake_fetch(pdb_id, *args, **kwargs):
            calls.append(pdb_id)
            return deposit

        monkeypatch.setattr("sampleworks.utils.input_prep.fetch_rcsb_cif", _fake_fetch)
        return calls

    def test_prepares_every_entry_by_default(self, tmp_path, monkeypatch):
        calls = self._patch_fetch(tmp_path, monkeypatch)
        result = prepare_from_registry(self._registry(tmp_path), cache_dir=tmp_path / "inputs")
        assert set(result) == {"1abc", "2def"}
        assert all(path.exists() for path in result.values())
        assert sorted(calls) == ["1abc", "2def"]  # one fetch per prepared input

    def test_subset_selection_is_case_insensitive(self, tmp_path, monkeypatch):
        self._patch_fetch(tmp_path, monkeypatch)
        result = prepare_from_registry(
            self._registry(tmp_path), ["1ABC"], cache_dir=tmp_path / "inputs"
        )
        assert set(result) == {"1abc"}

    def test_unknown_pdb_id_raises_before_any_fetch(self, tmp_path, monkeypatch):
        calls = self._patch_fetch(tmp_path, monkeypatch)
        with pytest.raises(KeyError, match="9zzz"):
            prepare_from_registry(
                self._registry(tmp_path), ["1abc", "9zzz"], cache_dir=tmp_path / "inputs"
            )
        assert calls == []  # fail-fast: nothing fetched when any id is bogus

    def test_second_call_is_a_cache_hit(self, tmp_path, monkeypatch):
        calls = self._patch_fetch(tmp_path, monkeypatch)
        registry = self._registry(tmp_path)
        cache = tmp_path / "inputs"
        prepare_from_registry(registry, ["1abc"], cache_dir=cache)
        prepare_from_registry(registry, ["1abc"], cache_dir=cache)
        assert calls == ["1abc"]  # served from cache the second time, no re-fetch


class TestProductionRegistry:
    """The registry the prep script defaults to must stay loadable and cover the knowns."""

    def test_known_specs_present(self):
        registry = load_registry(files("sampleworks.data") / "input_registry.json")
        assert {"2yl0", "9bn8", "1vme", "4ole"} <= set(registry)
        assert registry["1vme"].numbering == "preserve_label"
        assert registry["2yl0"].numbering == "preserve_auth"
        assert registry["9bn8"].numbering == "preserve_auth"
        assert registry["4ole"].chains == ["B"]
