"""Tests specific to the structure-factor (reciprocal-space) reward function.

Reward-agnostic contract tests (interface, relative correlation, coordinate gradients,
batch=1 occupancy gradients, batch/edge handling) live in
``test_reward_function_contract.py``, where they run against every reward. What remains
here is specific to ``StructureFactorRewardFunction``
(``sampleworks.core.rewards.structure_factor``) or to SFcalculator's forward model.

The headline SF-specific behavior is that SFcalculator has **no per-conformer (batch)
occupancy/B axis**: ``F_protein_batch`` batches only coordinates (its docstring still reads
``TODO: Support batched B factors``), so ``__call__`` can only honor a single shared
occupancy/B vector. Rather than silently using row 0 and dropping the rest, the reward
*rejects* non-broadcast occupancy/B — making the limitation a loud, caller-visible
invariant instead of a latent footgun. See ``test_per_conformer_occupancy_or_b_raises``.
"""

import pytest
import torch

from tests.rewards.reward_input_helpers import build_scattering_indices


@pytest.mark.gpu
@pytest.mark.slow
class TestStructureFactorOccupancy:
    """SFC has no per-conformer occupancy/B axis; the reward enforces broadcast-identical input."""

    @pytest.mark.parametrize("field", ["occupancies", "b_factors"])
    def test_per_conformer_occupancy_or_b_raises(
        self, reward_function_1vme_sf, test_coordinates_1vme_sf, device, field
    ):
        """Per-conformer (non-broadcast) occupancy/B is rejected, not silently dropped.

        SFcalculator batches only coordinates, so __call__ honors a single shared occupancy/B
        vector. Production always feeds broadcast-identical rows (the batch=1 and identical-row
        ensemble cases are covered by the shared contract tests); this pins that genuinely
        per-conformer values raise instead of being silently ignored. The guard runs before any
        SF compute, so this also documents that the [batch, n_atoms] signature does NOT mean
        per-conformer occupancy/B is supported.
        """
        coords, atom_array = test_coordinates_1vme_sf
        elements = build_scattering_indices(atom_array, device)
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        batch = 3
        kwargs = dict(
            coordinates=coords.unsqueeze(0).expand(batch, -1, -1),
            elements=elements.unsqueeze(0).expand(batch, -1),
            b_factors=b_factors.unsqueeze(0).expand(batch, -1),
            occupancies=occupancies.unsqueeze(0).expand(batch, -1),
        )
        # Make one conformer's values genuinely differ (additive shift works regardless of zeros).
        perturbed = kwargs[field].clone()
        perturbed[1] += 1.0
        kwargs[field] = perturbed

        with pytest.raises(ValueError, match="identical across the batch"):
            reward_function_1vme_sf(**kwargs)
