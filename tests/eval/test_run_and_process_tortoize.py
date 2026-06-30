"""Per-model tests for ``scripts/eval/run_and_process_tortoize.py``.

tortoize is mocked (the binary is not on the test runner), so these exercise the per-model
extract/score/merge plumbing, not tortoize itself.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import array, Atom, stack
from biotite.structure.io.pdbx import set_structure
from biotite.structure.io.pdbx.cif import CIFFile

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "run_and_process_tortoize.py"
)


def _load_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("run_and_process_tortoize_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script():
    return _load_script()


def _atom(chain: str, res_id: int, res_name: str, atom_name: str = "CA") -> Atom:
    return Atom(
        [0.0, 0.0, 0.0],
        chain_id=chain,
        res_id=res_id,
        res_name=res_name,
        hetero=False,
        atom_name=atom_name,
        element="C",
    )


def _write_cif(path: Path, n_models: int) -> Path:
    arr = array([_atom("A", 1, "ALA"), _atom("A", 2, "GLY")])
    arr.set_annotation("occupancy", np.ones(len(arr), dtype=np.float32))
    arr.set_annotation("b_factor", np.zeros(len(arr), dtype=np.float32))
    structure = stack([arr.copy() for _ in range(n_models)]) if n_models > 1 else arr
    cif = CIFFile()
    set_structure(cif, structure)
    cif.write(str(path))
    return path


def _tortoize_json() -> bytes:
    """A canned single-model tortoize JSON (tortoize labels a lone model "1")."""
    return json.dumps(
        {
            "model": {
                "1": {
                    "ramachandran-z": 0.5,
                    "torsion-z": 1.2,
                    "ramachandran-jackknife-sd": 0.1,
                    "torsion-jackknife-sd": 0.2,
                    "residues": [
                        {
                            "asymID": "A",
                            "compID": "ALA",
                            "seqID": 1,
                            "pdb": {"strandID": "A", "insCode": ""},
                            "ramachandran": {"z-score": 0.3, "ss-type": "H"},
                            "torsion": {"z-score": 0.4, "ss-type": "H"},
                        }
                    ],
                }
            }
        }
    ).encode()


def _mock_tortoize(monkeypatch, script, *, calls: list | None = None, raises: bool = False):
    def fake(cmd, *args, **kwargs):
        if calls is not None:
            calls.append(cmd)
        if raises:
            raise subprocess.CalledProcessError(1, cmd)
        return _tortoize_json()

    monkeypatch.setattr(script.subprocess, "check_output", fake)


class TestGetStatsForSinglePath:
    def test_multi_model_yields_one_row_per_model(self, script, monkeypatch, tmp_path):
        """A stacked ensemble is scored per model and merged, tagged with the source model."""
        path = _write_cif(tmp_path / "ensemble.cif", n_models=3)
        calls: list = []
        _mock_tortoize(monkeypatch, script, calls=calls)

        residues, protein = script.get_stats_for_single_path(path)

        assert len(calls) == 3  # one tortoize invocation per model
        assert not residues.empty and not protein.empty
        assert set(residues["model"]) == {"1", "2", "3"}
        assert set(protein["model"]) == {"1", "2", "3"}
        assert len(protein) == 3
        assert set(residues["path"]) == {path}

    def test_single_model_still_works(self, script, monkeypatch, tmp_path):
        path = _write_cif(tmp_path / "single.cif", n_models=1)
        calls: list = []
        _mock_tortoize(monkeypatch, script, calls=calls)

        residues, protein = script.get_stats_for_single_path(path)

        assert len(calls) == 1
        assert set(protein["model"]) == {"1"}
        assert len(protein) == 1

    def test_tortoize_failure_returns_empty(self, script, monkeypatch, tmp_path):
        """If every model fails tortoize, both frames come back empty (the loop contract)."""
        path = _write_cif(tmp_path / "ensemble.cif", n_models=2)
        _mock_tortoize(monkeypatch, script, raises=True)

        residues, protein = script.get_stats_for_single_path(path)

        assert residues.empty and protein.empty
