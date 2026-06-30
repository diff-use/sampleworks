"""
Tests for the RMSD metrics.
"""

import numpy as np
import pytest
from biotite.structure import rmsd, superimpose
from sampleworks.metrics.rmsd import AllAtomRMSD
from sampleworks.utils.atom_array_utils import filter_to_common_atoms


# TODO: make these a bit more rigorous, similarly to tests/metrics/test_lddt_metrics.py


def test_all_atom_rmsd_identity(altlocA_backbone):
    """A stack compared to itself must have zero RMSD on every residue."""
    metric = AllAtomRMSD()
    results = metric.compute(altlocA_backbone, altlocA_backbone)

    assert results["best_of_1_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert results["residue_rmsd_scores"]
    for residue, scores in results["residue_rmsd_scores"].items():
        assert scores == pytest.approx([0.0], abs=1e-6), f"nonzero identity RMSD at {residue}"


def test_all_atom_rmsd_end_to_end(altlocA_backbone, altlocB_backbone):
    """
    A residue selection on the 6b8x altloc A/B backbone yields the expected global best_of_1_rmsd
    and per-residue scores.
    """
    selection_string = "res_id > 179 and res_id < 190"
    metric = AllAtomRMSD()
    results = metric.compute(altlocA_backbone, altlocB_backbone, selection_string)

    expected_results = {
        "best_of_1_rmsd": 0.7332,
        "residue_rmsd_scores": {
            "A180": [2.9206],
            "A181": [4.2650],
            "A182": [5.6639],
            "A183": [2.9397],
            "A184": [2.0595],
            "A185": [2.2885],
            "A186": [3.6929],
            "A187": [1.8138],
            "A188": [0.9547],
            "A189": [0.7596],
        },
    }

    assert results["best_of_1_rmsd"] == pytest.approx(expected_results["best_of_1_rmsd"], abs=0.002)

    assert set(results["residue_rmsd_scores"].keys()) == set(
        expected_results["residue_rmsd_scores"].keys()
    )

    for residue_key, expected_scores in expected_results["residue_rmsd_scores"].items():
        result_scores = results["residue_rmsd_scores"][residue_key]
        assert len(result_scores) == len(expected_scores)
        for got, want in zip(result_scores, expected_scores):
            assert got == pytest.approx(want, abs=0.005), (
                f"RMSD mismatch at {residue_key}: got {got}, expected {want}"
            )


def test_all_atom_rmsd_selection_does_not_change_global(altlocA_backbone, altlocB_backbone):
    """Selection filters the residue-level dict but must not change the global RMSD."""
    metric = AllAtomRMSD()
    all_residues = metric.compute(altlocA_backbone, altlocB_backbone)
    selected = metric.compute(altlocA_backbone, altlocB_backbone, "res_id > 179 and res_id < 190")

    assert selected["best_of_1_rmsd"] == pytest.approx(all_residues["best_of_1_rmsd"], abs=1e-6)
    assert len(selected["residue_rmsd_scores"]) < len(all_residues["residue_rmsd_scores"])


def test_segment_rmsd_absent_without_selection(altlocA_backbone, altlocB_backbone):
    """No segment_rmsd keys are emitted when selection is None."""
    metric = AllAtomRMSD()
    result = metric.compute(altlocA_backbone, altlocB_backbone)

    assert "segment_rmsd" not in result
    assert not any(k.endswith("_segment_rmsd") for k in result)


def test_segment_rmsd_present_with_selection(altlocA_backbone, altlocB_backbone):
    """segment_rmsd is a list of length N_member and best_of_N_segment_rmsd is its min."""
    metric = AllAtomRMSD()
    result = metric.compute(altlocA_backbone, altlocB_backbone, "res_id > 179 and res_id < 190")

    n_member = altlocB_backbone.stack_depth()
    assert "segment_rmsd" in result
    assert len(result["segment_rmsd"]) == n_member

    key = f"best_of_{n_member}_segment_rmsd"
    assert key in result
    assert result[key] == pytest.approx(min(result["segment_rmsd"]))


def test_segment_rmsd_matches_hand_computed(altlocA_backbone, altlocB_backbone):
    """segment_rmsd equals biotite.rmsd on the same coordinates."""
    selection = "res_id > 179 and res_id < 190"
    metric = AllAtomRMSD(superimpose=True)
    result = metric.compute(altlocA_backbone, altlocB_backbone, selection)

    # Recompute by hand: filter to common atoms, superpose, mask, then biotite.rmsd.
    pred, gt = filter_to_common_atoms(altlocA_backbone, altlocB_backbone)
    gt_ref = gt[0]
    pred_aligned, _ = superimpose(gt_ref, pred)
    mask = pred_aligned.mask(selection)
    expected = np.asarray(rmsd(gt_ref[mask], pred_aligned[:, mask]))

    np.testing.assert_allclose(result["segment_rmsd"], expected.tolist(), rtol=1e-5, atol=1e-6)


def test_segment_rmsd_identity_is_zero(altlocA_backbone):
    """Comparing a stack to itself yields zero segment_rmsd."""
    metric = AllAtomRMSD()
    result = metric.compute(altlocA_backbone, altlocA_backbone, "res_id > 179 and res_id < 190")

    assert result["segment_rmsd"]
    for v in result["segment_rmsd"]:
        assert v == pytest.approx(0.0, abs=1e-6)
