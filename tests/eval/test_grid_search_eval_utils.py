"""Tests for reading trial metadata from refined.cif (issue #68, R3), with path fallback."""

from pathlib import Path

import numpy as np
from biotite.structure import array, Atom
from biotite.structure.io.pdbx import CIFFile, set_structure
from sampleworks.eval.grid_search_eval_utils import scan_grid_search_results
from sampleworks.utils.cif_utils import add_category_to_cif, add_completeness_categories


def _arr():
    atoms = [
        Atom(
            [0.0, 0.0, 0.0],
            chain_id="A",
            res_id=9,
            res_name="MET",
            atom_name="N",
            element="N",
            hetero=False,
        ),
        Atom(
            [1.0, 0.0, 0.0],
            chain_id="A",
            res_id=10,
            res_name="GLY",
            atom_name="CA",
            element="C",
            hetero=False,
        ),
    ]
    a = array(atoms)
    a.set_annotation("occupancy", np.ones(len(a), dtype=np.float32))
    a.set_annotation("b_factor", np.full(len(a), 20.0, dtype=np.float32))
    return a


def _write_refined(path: Path, meta: dict | None) -> Path:
    """Write a sampleworks-style refined.cif, optionally carrying a _sampleworks category."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cif = CIFFile()
    set_structure(cif, _arr())
    if meta is not None:
        add_category_to_cif(cif, dict(meta), category_name="sampleworks")
    add_completeness_categories(cif)
    cif.write(str(path))
    return path


_FK_META = {
    "protein": "1vme_0.5occA_0.5occB",
    "model": "boltz2",
    "method": "X-RAY DIFFRACTION",
    "guidance_type": "fk_steering",
    "ensemble_size": 4,
    "sampleworks_scaler": "fk_steering",
    "sampleworks_gradient_weight": 1.5,
    "sampleworks_gd_steps": 100,
}


def test_trial_built_from_metadata(tmp_path):
    root = tmp_path / "1vme_0.5occA_0.5occB"
    _write_refined(
        root / "boltz2_X-RAY_DIFFRACTION" / "fk_steering" / "ens4_gw1.5_gd100" / "refined.cif",
        _FK_META,
    )
    trials = scan_grid_search_results(root, target_depth=3)
    assert len(trials) == 1
    t = trials[0]
    assert t.protein == "1vme"
    assert t.altloc_occupancies == {"A": 0.5, "B": 0.5}
    assert t.model == "boltz2"
    assert t.method == "X-RAY"  # full config value normalized to the legacy short label
    assert t.scaler == "fk_steering"
    # CIF read-back is all strings; eval must coerce to int/float
    assert t.ensemble_size == 4 and isinstance(t.ensemble_size, int)
    assert t.guidance_weight == 1.5 and isinstance(t.guidance_weight, float)
    assert t.gd_steps == 100 and isinstance(t.gd_steps, int)


def test_pure_guidance_gd_steps_null(tmp_path):
    meta = {
        "protein": "1vme_0.5occA_0.5occB",
        "model": "boltz2",
        "method": "X-RAY DIFFRACTION",
        "guidance_type": "pure_guidance",
        "ensemble_size": 2,
        "sampleworks_scaler": "pure_guidance",
        "sampleworks_gradient_weight": 0.3,
        "sampleworks_gd_steps": None,  # pure runs have no gd_steps -> written as "?"
    }
    root = tmp_path / "1vme_0.5occA_0.5occB"
    _write_refined(
        root / "boltz2_X-RAY_DIFFRACTION" / "pure_guidance" / "ens2_gw0.3" / "refined.cif", meta
    )
    trials = scan_grid_search_results(root, target_depth=3)
    assert len(trials) == 1
    assert trials[0].gd_steps is None
    assert trials[0].guidance_weight == 0.3


def test_falls_back_to_path_when_no_metadata(tmp_path):
    root = tmp_path / "5sop_1.0occA"
    _write_refined(root / "boltz2_MD" / "fk_steering" / "ens8_gw2.0_gd50" / "refined.cif", None)
    trials = scan_grid_search_results(root, target_depth=3)
    assert len(trials) == 1
    t = trials[0]
    assert t.protein == "5sop"
    assert t.method == "MD"
    assert (t.ensemble_size, t.guidance_weight, t.gd_steps) == (8, 2.0, 50)


def test_incomplete_metadata_falls_back_to_path(tmp_path):
    # _sampleworks present but lacking the canonical eval keys (an older output) -> path parse.
    stale = {"protein": "5sop_1.0occA", "model": "boltz2", "guidance_type": "fk_steering"}
    root = tmp_path / "5sop_1.0occA"
    _write_refined(root / "boltz2_MD" / "fk_steering" / "ens8_gw2.0_gd50" / "refined.cif", stale)
    trials = scan_grid_search_results(root, target_depth=3)
    assert len(trials) == 1
    # path fallback recovers what the stale metadata could not provide
    assert trials[0].gd_steps == 50
    assert trials[0].guidance_weight == 2.0
    assert trials[0].method == "MD"
