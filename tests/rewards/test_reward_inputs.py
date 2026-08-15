"""Tests for RewardInputs validation and atom array round-tripping.

Verifies that RewardInputs.from_atom_array rejects atom arrays with NaN
B-factors, NaN coordinates, and invalid occupancies, and that
RewardInputs.to_atom_array rebuilds the reference array a reward's
``prepare()`` consumes.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from biotite.structure import array, Atom, AtomArray
from sampleworks.core.rewards.protocol import RewardInputs


class TestRewardInputsFromAtomArray:
    """Validate that RewardInputs.from_atom_array rejects invalid atom arrays."""

    def test_nan_b_factors_from_missing_atoms_rejected(self, atom_array_1vme_with_missing_atoms):
        """
        RewardInputs rejects the raw atom array from add_missing_atoms.
        """
        with pytest.raises(ValueError, match="NaN B-factors"):
            RewardInputs.from_atom_array(atom_array_1vme_with_missing_atoms, ensemble_size=1)

    def test_cleaned_missing_atoms_accepted(self, atom_array_1vme_with_missing_atoms):
        """After applying the same fixes as the RF3 wrapper, the atom array passes."""
        aa = atom_array_1vme_with_missing_atoms.copy()

        # Fix NaN coordinates
        nan_coord_mask = np.any(np.isnan(aa.coord), axis=-1)
        if nan_coord_mask.any():
            resolved_coords = aa.coord[~nan_coord_mask]
            if len(resolved_coords) > 0:
                centroid = resolved_coords.mean(axis=0)
            else:
                centroid = np.zeros(3)
            n_nan = int(nan_coord_mask.sum())
            noise = np.random.normal(loc=0.0, scale=1.0, size=(n_nan, 3)).astype(np.float32)
            new_coords = aa.coord.copy()
            new_coords[nan_coord_mask] = centroid + noise
            aa.coord = new_coords

        # Fix occupancy
        aa.set_annotation("occupancy", np.ones(len(aa), dtype=np.float32))

        # Fix NaN b_factors
        nan_b_mask = np.isnan(aa.b_factor)
        if nan_b_mask.any():
            b_factors = aa.b_factor.copy()
            b_factors[nan_b_mask] = 20.0
            aa.set_annotation("b_factor", b_factors)

        reward_inputs = RewardInputs.from_atom_array(aa, ensemble_size=1)
        assert reward_inputs.b_factors.shape[-1] == len(aa)

    def test_nan_coordinates_rejected(self, structure_1vme):
        """NaN coordinates must be caught before constructing reward tensors."""
        atom_array = structure_1vme["asym_unit"].copy()
        # Fix any pre-existing NaN b_factors so the coordinate check passes
        b_factors = np.nan_to_num(atom_array.b_factor, nan=20.0)
        atom_array.set_annotation("b_factor", b_factors)
        coords = atom_array.coord.copy()
        coords[..., 3, :] = np.nan
        atom_array.coord = coords

        with pytest.raises(ValueError, match="NaN coordinates"):
            RewardInputs.from_atom_array(atom_array, ensemble_size=1)


@pytest.mark.gpu
class TestRewardInputsToAtomArray:
    """Verify the reference atom array handed to a reward's ``prepare()``.

    GPU-marked at class level because RewardInputs can hold CUDA tensors but the atom array
    annotations must be numpy arrays.
    """

    @pytest.fixture
    def toy_atom_array(self) -> AtomArray:
        """Four atoms with per-atom-distinct coords / B-factors and fractional occupancies."""
        atoms = [
            Atom(
                [1.0, 2.0, 3.0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name="N",
                element="N",
                b_factor=11.0,
                occupancy=0.5,
            ),
            Atom(
                [2.0, 3.0, 4.0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name="CA",
                element="C",
                b_factor=12.0,
                occupancy=0.5,
            ),
            Atom(
                [3.0, 4.0, 5.0],
                chain_id="A",
                res_id=2,
                res_name="ALA",
                atom_name="N",
                element="N",
                b_factor=13.0,
                occupancy=1.0,
            ),
            Atom(
                [4.0, 5.0, 6.0],
                chain_id="A",
                res_id=2,
                res_name="ALA",
                atom_name="CB",
                element="C",
                b_factor=14.0,
                occupancy=0.25,
            ),
        ]
        return array(atoms)

    @pytest.mark.parametrize(
        "ensemble_size,num_particles",
        [
            (1, 1),
            (4, 1),
            (4, 2),
        ],
    )
    def test_round_trip_preserves_coords_b_factors_and_topology(
        self, ensemble_size, num_particles, toy_atom_array, device
    ):
        """from_atom_array -> to_atom_array returns the template's own values.

        Parametrized over ensemble_size and num_particles, since those decide the batch
        layout the annotations are collapsed out of.
        """
        template = toy_atom_array
        reward_inputs = RewardInputs.from_atom_array(
            template,
            ensemble_size=ensemble_size,
            num_particles=num_particles,
            device=device,
        )

        result = reward_inputs.to_atom_array(template)

        np.testing.assert_allclose(result.coord, template.coord)
        np.testing.assert_allclose(result.b_factor, template.b_factor)
        # array_equal, not assert_allclose: these are int / string annotations. A float
        # annotation added here would compare NaN != NaN and belong above instead.
        for category in ["atom_name", "res_id", "res_name", "chain_id", "element"]:
            assert np.array_equal(
                result.get_annotation(category), template.get_annotation(category)
            ), f"annotation {category!r} did not round-trip"

    def test_input_values_win_over_template(self, toy_atom_array, device):
        """Reward-input values overwrite the template's own coords / B-factors.

        ``SampleworksProcessedStructure.to_reward_inputs`` substitutes
        structure-derived B-factors to the reward inputs build from the model atom via the
        reconciler (``eval/structure_utils.py``), so the reward inputs can differ from the
        original template atom array.
        """
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=2, device=device)
        reconciled_reward_inputs = replace(
            reward_inputs,
            input_coords=reward_inputs.input_coords + 10.0,
            b_factors=torch.full_like(reward_inputs.b_factors, 42.0),
        )

        result = reconciled_reward_inputs.to_atom_array(toy_atom_array)

        np.testing.assert_allclose(result.coord, toy_atom_array.coord + 10.0)
        np.testing.assert_allclose(result.b_factor, np.full(len(toy_atom_array), 42.0))
        # occupancy is forced to 1.0 as topology representation
        np.testing.assert_allclose(result.occupancy, np.ones(len(toy_atom_array)))

    def test_template_is_not_mutated(self, toy_atom_array, device):
        """The template is copied, so the caller's model atom array survives the call."""
        original_coords = toy_atom_array.coord.copy()
        originals = {
            category: toy_atom_array.get_annotation(category).copy()
            for category in ("b_factor", "occupancy")
        }
        reward_inputs = RewardInputs.from_atom_array(toy_atom_array, ensemble_size=1, device=device)
        reconciled_reward_inputs = replace(
            reward_inputs,
            input_coords=reward_inputs.input_coords + 5.0,
            b_factors=torch.rand_like(reward_inputs.b_factors),
        )

        reconciled_reward_inputs.to_atom_array(toy_atom_array)

        np.testing.assert_allclose(toy_atom_array.coord, original_coords)
        for category, original in originals.items():
            np.testing.assert_allclose(toy_atom_array.get_annotation(category), original)

    def test_atom_count_mismatch_raises(self, toy_atom_array, device):
        """A template from a different atom space is rejected, not silently broadcast."""
        template = toy_atom_array
        reward_inputs = RewardInputs.from_atom_array(template, ensemble_size=1, device=device)
        n_subset_atoms = len(template) - 1
        with pytest.raises(ValueError, match=f"template_atom_array has {n_subset_atoms} atoms"):
            reward_inputs.to_atom_array(template[:n_subset_atoms])
