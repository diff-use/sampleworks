"""Tests for tilde-truncated CCD code expansion in Protenix structure processing."""

from unittest.mock import patch

import pytest


pytest.importorskip("protenix", reason="Protenix not installed")

from sampleworks.models.protenix.structure_processing import (
    _build_ccd_suffix_map,
    _expand_tilde_ccd_code,
    structure_to_protenix_json,
)
from sampleworks.models.protenix.wrapper import _make_ccd_parse_error


class TestExpandTildeCCDCode:
    """Test _expand_tilde_ccd_code helper function."""

    def test_unique_match_expands(self):
        """~QS should expand uniquely to A1AQS."""
        result = _expand_tilde_ccd_code("~QS")
        assert result == "A1AQS"

    def test_ambiguous_match_returns_original(self):
        """When multiple codes share a suffix, return the truncated code."""
        fake_codes = ["A1DXX", "A1HXX", "GLY", "ALA"]
        _build_ccd_suffix_map.cache_clear()
        with patch(
            "protenix.data.ccd.get_all_ccd_code",
            return_value=fake_codes,
        ):
            result = _expand_tilde_ccd_code("~XX")
        _build_ccd_suffix_map.cache_clear()
        assert result == "~XX"

    def test_no_match_returns_original(self):
        """When no code matches the suffix, return the truncated code."""
        result = _expand_tilde_ccd_code("~$$")
        assert result == "~$$"


class TestStructureToProtenixJsonCCDExpansion:
    """Test that structure_to_protenix_json expands tilde CCD codes."""

    def test_9bn8_ligand_expanded(self, structure_9bn8):
        """9BN8 structure with ~QS ligand should produce CCD_A1AQS in JSON."""
        json_dict = structure_to_protenix_json(structure_9bn8)

        ligand_entries = [
            entry["ligand"]["ligand"] for entry in json_dict["sequences"] if "ligand" in entry
        ]

        tilde_codes = [code for code in ligand_entries if "~" in code]
        assert tilde_codes == [], (
            f"Tilde-truncated codes should have been expanded, but found: {tilde_codes}"
        )

        a1aqs_entries = [code for code in ligand_entries if "A1AQS" in code]
        assert len(a1aqs_entries) > 0, (
            f"Expected CCD_A1AQS in ligand entries, got: {ligand_entries}"
        )


class TestMakeCCDParseError:
    """Test _make_ccd_parse_error helper produces actionable diagnostics."""

    def test_includes_ligand_codes(self):
        """Error message should list the ligand CCD codes from the input."""
        json_dict = {
            "sequences": [
                {"ligand": {"ligand": "CCD_~QS", "count": 1}},
                {"proteinChain": {"sequence": "ACGT", "count": 1}},
            ]
        }
        error = _make_ccd_parse_error(json_dict)
        assert isinstance(error, TypeError)
        assert "CCD_~QS" in str(error)
        assert "tilde-prefixed CCD codes" in str(error)

    def test_handles_empty_sequences(self):
        """Should not crash on empty or missing sequences."""
        error = _make_ccd_parse_error({})
        assert isinstance(error, TypeError)
        assert "tilde-prefixed CCD codes" in str(error)

    def test_handles_multiple_ligands(self):
        """Should list all ligand CCD codes when multiple are present."""
        json_dict = {
            "sequences": [
                {"ligand": {"ligand": "CCD_~QS", "count": 1}},
                {"ligand": {"ligand": "CCD_ATP", "count": 1}},
            ]
        }
        error = _make_ccd_parse_error(json_dict)
        msg = str(error)
        assert "CCD_~QS" in msg
        assert "CCD_ATP" in msg
