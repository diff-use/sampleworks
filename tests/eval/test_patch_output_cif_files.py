"""Unit tests for the RCSB id extraction in ``scripts/patch_output_cif_files.py``.

This is the "main checkpoint" of the patching pipeline: the id is read from the folder
segment of a path and drives both the RCSB ``fetch`` and the reference-input lookup, so a
wrong id here would silently patch coordinates into the wrong template. These tests pin the
exact capture for legacy and extended ids and assert that malformed folders are rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .script_loader import load_script


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "patch_output_cif_files.py"


@pytest.fixture(scope="module")
def script():
    return load_script(_SCRIPT_PATH)


# An independent copy of the script's DEFAULT_RCSB_PATTERN, kept separate on purpose: the
# test is a spec, not a mirror of the implementation. If the script's default ever changes,
# this literal does not move with it -- the resulting behavior difference should surface as
# a failure here for a human to review, rather than being silently tracked.
_DEFAULT_REGEX = r"grid_search_results/(pdb_[A-Za-z0-9]{8}|[0-9][A-Za-z0-9]{3})"


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        # --- WORKS: the folder is exactly a PDB id -> extracted verbatim ---
        ("4hhb", "4hhb"),  # legacy, lowercase
        ("1VME", "1VME"),  # legacy, uppercase
        ("9BN8", "9BN8"),  # legacy, mixed case + digits
        ("pdb_00004hhb", "pdb_00004hhb"),  # extended transitional -- full id, NOT 0000 / 4hhb
        ("pdb_1000axyz", "pdb_1000axyz"),  # extended, genuinely-new id with no legacy form
        # --- REJECTED (-> None, skipped): not a bare PDB id, so we never mis-pick an entry ---
        ("TEST", None),  # not a PDB id at all (letter-led)
        ("logs", None),  # scratch folder
        ("data", None),  # scratch folder
        ("4hhb_final", None),  # id is only a PREFIX of the folder (would else capture "4hhb")
        ("4hhbEXTRA", None),  # id prefix, no separator
        ("pdb_1000abcd_final", None),  # extended-id prefix + suffix
        ("protease_pdb_1000abcd", None),  # id embedded after a prefix (reviewer's example)
        ("1abc_pdb_1000abcd", None),  # two id-like substrings -> ambiguous (would else pick "1abc")
    ],
)
def test_extract_rcsb_id_from_folder(script, folder: str, expected: str | None) -> None:
    """Only a folder that is *exactly* a PDB id is extracted; everything else returns None.

    The rejected rows are the important part: a folder like ``4hhb_final`` or
    ``1abc_pdb_1000abcd`` embeds an id-shaped substring, and without the whole-component rule
    the extractor would silently capture ``4hhb`` / ``1abc`` and patch the wrong entry. These
    are skipped instead. Extracting an id embedded inside a larger folder name is deliberately
    NOT supported by the default pattern (it is ambiguous when several id-like parts appear).
    """
    path = Path(f"/data/results/grid_search_results/{folder}/trial_1/refined.cif")
    assert script.extract_rcsb_id(path, _DEFAULT_REGEX) == expected


def test_extract_rcsb_id_no_match_returns_none(script) -> None:
    """A path that doesn't contain the anchor at all yields None (not an exception)."""
    path = Path("/data/results/some_other_dir/4hhb/refined.cif")
    assert script.extract_rcsb_id(path, _DEFAULT_REGEX) is None


def test_loose_pattern_captures_non_id_raises(script) -> None:
    """A permissive custom --rcsb-pattern that captures a non-id raises InvalidRcsbIdError.

    This is distinct from a plain no-match (which returns None): the captured token matched
    the pattern but isn't a real PDB id, so the user is told their pattern is the culprit.
    A real id captured by the same loose pattern is still accepted.
    """
    loose = r"grid_search_results/(.{4})"
    base = Path("/data/results/grid_search_results")
    with pytest.raises(script.InvalidRcsbIdError, match="not a valid PDB id"):
        script.extract_rcsb_id(base / "TEST" / "refined.cif", loose)
    assert script.extract_rcsb_id(base / "4hhb" / "refined.cif", loose) == "4hhb"


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
