"""Tests for RewardInputs validation.

Verifies that RewardInputs.from_atom_array rejects atom arrays with NaN
B-factors, NaN coordinates, and invalid occupancies.
"""

import numpy as np
import pytest
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
            centroid = resolved_coords.mean(axis=0)
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

    def test_zero_occupancy_rejected(self, structure_1vme):
        """Zero occupancy (from unresolved atoms) must be caught."""
        atom_array = structure_1vme["asym_unit"].copy()
        # Fix any pre-existing NaN b_factors so the occupancy check passes
        b_factors = np.nan_to_num(atom_array.b_factor, nan=20.0)
        atom_array.set_annotation("b_factor", b_factors)
        # Fix any pre-existing NaN coords so the occupancy check passes
        coords = np.nan_to_num(atom_array.coord, nan=0.0)
        atom_array.coord = coords
        occupancies = atom_array.occupancy.copy()
        occupancies[0:3] = 0.0
        atom_array.set_annotation("occupancy", occupancies)

        with pytest.raises(ValueError, match="invalid occupancy"):
            RewardInputs.from_atom_array(atom_array, ensemble_size=1)
