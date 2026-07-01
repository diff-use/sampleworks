"""Unit tests for the RCSB id extraction in ``scripts/patch_output_cif_files.py``.

This is the "main checkpoint" of the patching pipeline: the id is read from the folder
segment of a path and drives both the RCSB ``fetch`` and the reference-input lookup, so a
wrong id here would silently patch coordinates into the wrong template. These tests pin the
exact capture for legacy and extended ids and assert that malformed folders are rejected.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "patch_output_cif_files.py"


def _load_script():
    """Import the script module by path so tests don't require it on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("patch_output_cif_files_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load_script()


# The script-side default pattern. The TOML presets use ``[A-Za-z0-9]{4}`` for the legacy
# branch instead of ``.{4}``; both are exercised (see ``test_matches_toml_legacy_class``).
_DEFAULT_REGEX = r"grid_search_results/(pdb_[A-Za-z0-9]{8}|.{4})"


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("4hhb", "4hhb"),  # legacy, lowercase
        ("1VME", "1VME"),  # legacy, uppercase
        ("9BN8", "9BN8"),  # legacy, mixed case + digits
        ("pdb_00004hhb", "pdb_00004hhb"),  # extended transitional -- full id, NOT 0000 / 4hhb
        ("pdb_1000axyz", "pdb_1000axyz"),  # extended, genuinely-new id with no legacy form
        ("TEST", None),  # letter-led folder -> not a valid legacy id -> rejected
        ("logs", None),  # scratch folder -> rejected
        ("data", None),  # scratch folder -> rejected
    ],
)
def test_extract_rcsb_id_from_folder(script, folder: str, expected: str | None) -> None:
    """The id is taken verbatim from the folder segment; garbage folders return None."""
    path = Path(f"/data/results/grid_search_results/{folder}/trial_1/refined.cif")
    assert script.extract_rcsb_id(path, _DEFAULT_REGEX) == expected


def test_extract_rcsb_id_no_match_returns_none(script) -> None:
    """A path that doesn't contain the anchor at all yields None (not an exception)."""
    path = Path("/data/results/some_other_dir/4hhb/refined.cif")
    assert script.extract_rcsb_id(path, _DEFAULT_REGEX) is None


def test_matches_toml_legacy_class(script) -> None:
    """The TOML presets' legacy branch ``[A-Za-z0-9]{4}`` captures identically."""
    regex = r"grid_search_results/(pdb_[A-Za-z0-9]{8}|[A-Za-z0-9]{4})"
    base = Path("/data/results/grid_search_results")
    assert script.extract_rcsb_id(base / "4hhb" / "refined.cif", regex) == "4hhb"
    assert script.extract_rcsb_id(base / "pdb_00004hhb" / "refined.cif", regex) == "pdb_00004hhb"
    assert script.extract_rcsb_id(base / "TEST" / "refined.cif", regex) is None


@pytest.mark.parametrize(
    "bad_regex",
    [
        r"grid_search_results/pdb_[A-Za-z0-9]{8}",  # zero capturing groups
        r"grid_search_results/(pdb_)([A-Za-z0-9]{8})",  # two capturing groups
    ],
)
def test_extract_rcsb_id_requires_single_group(script, bad_regex: str) -> None:
    """A misconfigured pattern (not exactly one group) fails loudly, not silently."""
    path = Path("/data/results/grid_search_results/pdb_00004hhb/refined.cif")
    with pytest.raises(ValueError, match="exactly one capturing group"):
        script.extract_rcsb_id(path, bad_regex)
