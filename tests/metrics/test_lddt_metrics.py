from typing import cast

import pytest
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArrayStack
from sampleworks.metrics.lddt import AllAtomLDDT, SelectedLDDT
from sampleworks.utils.atom_array_utils import select_altloc, select_backbone


# These tests are currently too high level, but they will serve for now to demonstrate
# the expected behavior and make sure nothing gets broken.


@pytest.fixture(scope="module")
def altlocA_backbone(structure_6b8x_with_altlocs) -> AtomArrayStack:
    altlocA = select_altloc(structure_6b8x_with_altlocs, "A", return_full_array=True)
    altlocA_bb = select_backbone(altlocA)
    return ensure_atom_array_stack(altlocA_bb)


@pytest.fixture(scope="module")
def altlocB_backbone(structure_6b8x_with_altlocs) -> AtomArrayStack:
    altlocB = select_altloc(structure_6b8x_with_altlocs, "B", return_full_array=True)
    altlocB_bb = select_backbone(altlocB)
    return ensure_atom_array_stack(altlocB_bb)


def test_all_atom_lddt_end_to_end(altlocA_backbone, altlocB_backbone):
    selection_string = "res_id > 179 and res_id < 190"
    allatom = AllAtomLDDT()
    results = allatom.compute(altlocA_backbone, altlocB_backbone, selection_string)

    expected_results = {
        "best_of_1_lddt": 0.970,
        "residue_lddt_scores": {
            "A180": [0.6408],
            "A181": [0.5700],
            "A182": [0.4511],
            "A183": [0.6169],
            "A184": [0.7267],
            "A185": [0.6974],
            "A186": [0.5572],
            "A187": [0.7665],
            "A188": [0.8666],
            "A189": [0.8819],
        },
    }

    assert "best_of_1_lddt" in results
    assert "residue_lddt_scores" in results

    # Check best_of_1_lddt value
    assert results["best_of_1_lddt"] == pytest.approx(expected_results["best_of_1_lddt"], abs=0.002)

    # Check that all expected keys are present in residue_lddt_scores
    assert set(results["residue_lddt_scores"].keys()) == set(
        expected_results["residue_lddt_scores"].keys()
    )

    # Check each residue's LDDT scores
    for residue_key in expected_results["residue_lddt_scores"]:
        assert residue_key in results["residue_lddt_scores"], f"Missing residue: {residue_key}"

        result_scores = results["residue_lddt_scores"][residue_key]
        expected_scores = expected_results["residue_lddt_scores"][residue_key]

        # Check that the list lengths match
        assert len(result_scores) == len(expected_scores), (
            f"Length mismatch for {residue_key}: got {len(result_scores)}, expected "
            f"{len(expected_scores)}"
        )

        # Check each score value
        for i, (result_score, expected_score) in enumerate(zip(result_scores, expected_scores)):
            assert result_score == pytest.approx(expected_score, abs=0.005), (
                f"Score mismatch for {residue_key}[{i}]: got {result_score}, expected "
                f"{expected_score}\nAll resulting scores: {results['residue_lddt_scores']}\n"
                f"Expected scores: {expected_results['residue_lddt_scores']}"
            )


def test_selected_lddt_end_to_end(altlocA_backbone, altlocB_backbone):
    import numpy as np

    selection_string = "res_id > 179 and res_id < 185"
    lddt = SelectedLDDT()
    results = lddt.compute(altlocA_backbone, altlocB_backbone, (selection_string,))

    expected_results = {
        selection_string: {
            "overall_lddt": np.array([0.8281]),
            "residue_lddt_scores": {
                "A180": [0.796],
                "A181": [0.8594],
                "A182": [0.7969],
                "A183": [0.8359],
                "A184": [0.8516],
            },
        }
    }

    assert selection_string in results
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    result_dict = results[selection_string]
    expected_dict = expected_results[selection_string]
    # Check overall value - now returns numpy array
    assert isinstance(result_dict["overall_lddt"], np.ndarray), (
        "overall_lddt should be a numpy array"
    )
    np.testing.assert_allclose(
        result_dict["overall_lddt"], expected_dict["overall_lddt"], atol=0.001
    )

    # Check that all expected keys are present in residue_lddt_scores
    result_residue_scores = cast(dict[str, list[float]], result_dict["residue_lddt_scores"])
    expected_residue_scores = cast(dict[str, list[float]], expected_dict["residue_lddt_scores"])
    assert set(result_residue_scores.keys()) == set(expected_residue_scores.keys())

    # Check each residue's LDDT scores
    for residue_key in expected_residue_scores:
        result_scores = result_residue_scores[residue_key]
        expected_scores = expected_residue_scores[residue_key]

        # Check that the list lengths match
        assert len(result_scores) == len(expected_scores), (
            f"Length mismatch for {residue_key}: got {len(result_scores)}, expected "
            f"{len(expected_scores)}"
        )

        # Check each score value
        for i, (result_score, expected_score) in enumerate(zip(result_scores, expected_scores)):
            assert result_score == pytest.approx(expected_score, abs=0.001), (
                f"Score mismatch for {residue_key}[{i}]: got {result_score}, expected "
                f"{expected_score}"
            )
