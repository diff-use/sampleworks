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

import logging

import pytest
import torch
from sampleworks.core.rewards.structure_factor import (
    _detect_mtz_metadata,
    StructureFactorRewardFunction,
)

from tests.rewards.reward_input_helpers import build_scattering_indices


class TestStructureFactorConstruction:
    """Construction-time behavior. These resolve columns in ``__init__`` before any SF compute,
    so they run on CPU (no GPU/``prepare()`` needed) and read only the MTZ header."""

    @pytest.fixture
    def mtz_columns(self, mtz_path_1vme):
        """``(amplitude_cols, sigma_cols)`` detected in the test MTZ, in MTZ order.

        The 1vme MTZ is multi-set, so ``amplitude_cols``/``sigma_cols`` each list more than one
        column (e.g. ``[Fprotein, Ftotal]`` / ``[SIGFprotein, SIGFtotal]``). Tests assert by
        position rather than literal names so they don't hard-code the layout.
        """
        _, _, amplitude_cols, sigma_cols = _detect_mtz_metadata(str(mtz_path_1vme))
        assert len(amplitude_cols) > 1 and len(sigma_cols) > 1  # precondition: a multi-set MTZ
        return amplitude_cols, sigma_cols

    def test_multi_set_mtz_autoselects_first_and_warns(self, mtz_path_1vme, mtz_columns, caplog):
        """A multi-set MTZ with no ``expcolumns`` auto-selects the first amplitude paired with its
        matching sigma, and warns that it picked among several sets.
        """
        amplitude_cols, sigma_cols = mtz_columns
        with caplog.at_level(logging.WARNING):
            reward = StructureFactorRewardFunction(mtz_path_1vme, device=torch.device("cpu"))
        assert reward.expcolumns == [amplitude_cols[0], sigma_cols[0]]
        assert "multiple amplitude columns" in caplog.text

    def test_explicit_expcolumns_override_selection(self, mtz_path_1vme, mtz_columns):
        """Explicit ``expcolumns`` are used verbatim, overriding the auto-selected first pair."""
        amplitude_cols, sigma_cols = mtz_columns
        explicit = [amplitude_cols[-1], sigma_cols[-1]]  # the last set, not the default first
        reward = StructureFactorRewardFunction(
            mtz_path_1vme, expcolumns=explicit, device=torch.device("cpu")
        )
        assert reward.expcolumns == explicit

    def test_unknown_expcolumns_raise(self, mtz_path_1vme, mtz_columns):
        """Explicit ``expcolumns`` naming a column absent from the MTZ fail fast at construction."""
        _, sigma_cols = mtz_columns
        with pytest.raises(ValueError, match="not found as MTZ amplitude/sigma columns"):
            StructureFactorRewardFunction(
                mtz_path_1vme, expcolumns=["Fnope", sigma_cols[0]], device=torch.device("cpu")
            )

    @pytest.mark.parametrize("bad_partition", [0, -5])
    def test_nonpositive_batch_partition_raises(self, mtz_path_1vme, bad_partition):
        """A non-positive ``batch_partition`` (an OOM knob) fails fast at construction.

        The check precedes column resolution, so no ``expcolumns`` are needed.
        """
        with pytest.raises(ValueError, match="batch_partition must be a positive integer"):
            StructureFactorRewardFunction(
                mtz_path_1vme, batch_partition=bad_partition, device=torch.device("cpu")
            )


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
