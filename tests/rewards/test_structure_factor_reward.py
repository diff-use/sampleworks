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
from biotite.structure import AtomArray
from sampleworks.core.rewards.structure_factor import (
    _detect_mtz_metadata,
    StructureFactorRewardFunction,
)
from sampleworks.utils.atom_array_utils import (
    build_pairwise_altloc_arrays,
    find_all_altloc_ids,
)

from tests.rewards.reward_input_helpers import build_reward_input_tensors_without_coords


def make_prepared_reward(mtz_path, atom_array, device: torch.device, **kwargs):
    """Construct and ``prepare()`` an SF reward with a per-test config.

    Touches the SFcalculator forward model (heavier than a header read), so callers are marked
    ``gpu``. ``kwargs`` pass straight to the constructor (e.g. ``bulk_solvent``,
    ``normalize_amplitude``, ``exclude_free_reflections``, ``expcolumns``). The single fixed
    config lives in the ``reward_function_1vme_sf`` fixture; these tests need varied configs.
    """
    rf = StructureFactorRewardFunction(mtz_path, device=device, **kwargs)
    rf.prepare(atom_array)
    return rf


@pytest.fixture(scope="module")
def sf_true_inputs(test_coordinates_1vme_sf, device: torch.device) -> dict:
    """``__call__`` kwargs (batch=1) for the true 1vme structure.

    The reward-agnostic contract tests already exercise the generic interface/gradient/batch
    behavior against the SF reward's default config, so the tests here only add the
    config-specific deltas and reuse these inputs.
    """
    coords, atom_array = test_coordinates_1vme_sf
    elements, b_factors, occupancies = build_reward_input_tensors_without_coords(atom_array, device)
    return dict(
        coordinates=coords.unsqueeze(0),
        elements=elements.unsqueeze(0),
        b_factors=b_factors.unsqueeze(0),
        occupancies=occupancies.unsqueeze(0),
    )


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
        elements, b_factors, occupancies = build_reward_input_tensors_without_coords(
            atom_array, device
        )

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


@pytest.mark.gpu
class TestStructureFactorBulkSolvent:
    """The bulk-solvent modes. ``off`` scores ``|Fprotein|`` (covered by the contract tests);
    ``combined``/``per_conformer`` score ``|Ftotal|`` with default-scaled solvent. The two
    ``|Ftotal|`` modes differ only for a real ensemble. Comparisons are relative, so they don't
    depend on the raw ``|F|`` scale.
    """

    @pytest.fixture(scope="class")
    def sf_ensemble_inputs(
        self, structure_1vme_sf, device: torch.device
    ) -> tuple[dict[str, torch.Tensor], AtomArray]:
        """A genuine 2-conformer ensemble (altloc A vs B) as ``(__call__ kwargs, state-A reference)``.

        ``build_pairwise_altloc_arrays`` pairs each altloc with the shared blank-altloc atoms and
        filters to their common atoms, so both frames share one topology and differ *only* in the
        alternate-conformation coordinates — the divergence that drives ``combined`` and
        ``per_conformer`` apart. Residues modeled in only one altloc have no counterpart and are
        dropped to keep that shared topology.

        SFcalculator has no per-conformer occ/B axis, so ``b_factors`` is shared from altloc-A
        across the batch, and occupancy shared at the uniform ``1/E = 0.5``.
        """
        aa = structure_1vme_sf
        altloc_ids = sorted(find_all_altloc_ids(aa))  # ["A", "B"]
        array_a, array_b = build_pairwise_altloc_arrays(aa, altloc_ids)[tuple(altloc_ids)]
        ref = array_a[0]  # single-conformer state-A reference; retains element/b_factor/occupancy

        coords = torch.stack(
            [
                torch.from_numpy(array_a.coord[0]),  # altloc-A conformer [N, 3]
                torch.from_numpy(array_b.coord[0]),  # altloc-B conformer [N, 3]
            ]
        ).to(device=device, dtype=torch.float32)  # [2, N, 3]

        elements, b_factors, _ = build_reward_input_tensors_without_coords(ref, device)
        n_atoms = coords.shape[1]
        occ = torch.full((n_atoms,), 0.5, device=device)  # uniform 1/E
        kwargs = dict(
            coordinates=coords,
            elements=elements.unsqueeze(0).expand(2, -1),
            b_factors=b_factors.unsqueeze(0).expand(2, -1),
            occupancies=occ.unsqueeze(0).expand(2, -1),
        )
        return kwargs, ref

    def test_combined_fits_ftotal_column(
        self, mtz_path_1vme, test_coordinates_1vme_sf, sf_true_inputs, device
    ):
        """Adding default-scaled bulk solvent (``combined``) fits the synthetic ``Ftotal``
        column far better than protein-only (``off``) — the synthetic ground truth was generated
        with the same default scales."""
        _, atom_array = test_coordinates_1vme_sf
        reward_with_solvent_off = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="off",
            normalize_amplitude=False,
        )
        reward_with_solvent_combined = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )
        loss_combined = reward_with_solvent_combined(**sf_true_inputs)
        loss_off = reward_with_solvent_off(**sf_true_inputs)
        assert loss_combined < loss_off

    def test_ftotal_modes_agree_for_single_conformer(
        self, mtz_path_1vme, test_coordinates_1vme_sf, sf_true_inputs, device
    ):
        """For a single conformer (E=1) ``mask(<rho>)`` and ``<mask(rho)>`` are the same mask,
        so ``combined`` and ``per_conformer`` give the same loss."""
        _, atom_array = test_coordinates_1vme_sf
        reward_with_solvent_combined = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )
        reward_with_solvent_per_conformer = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="per_conformer",
            normalize_amplitude=False,
        )
        torch.testing.assert_close(
            reward_with_solvent_combined(**sf_true_inputs),
            reward_with_solvent_per_conformer(**sf_true_inputs),
        )

    def test_per_conformer_averages_masks_over_ensemble(
        self, mtz_path_1vme, test_coordinates_1vme_sf, sf_true_inputs, device
    ):
        """E=2 *identical* conformers (occ 1/2 each) score the same as the single conformer.

        This guards the equal-population assumption in ``per_conformer``: the protein path bakes
        ``atom_occ`` into each conformer and *sums*, while the solvent path averages
        scale-invariant per-conformer masks (``Fmask_HKL_batch.mean(dim=0)``) — a hardcoded
        uniform ``1/E`` weight. Both are population *averages*, so occ = 1/E keeps them
        consistent and the ensemble matches the single conformer; this would break if
        ``per_conformer`` summed (rather than averaged) the masks. Non-uniform per-conformer
        occupancy is properly rejected by the reward now — see
        ``TestStructureFactorOccupancy.test_per_conformer_occupancy_or_b_raises``.
        """
        _, atom_array = test_coordinates_1vme_sf
        reward = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="per_conformer",
            normalize_amplitude=False,
        )
        ensemble = dict(
            coordinates=sf_true_inputs["coordinates"].expand(2, -1, -1),
            elements=sf_true_inputs["elements"].expand(2, -1),
            b_factors=sf_true_inputs["b_factors"].expand(2, -1),
            occupancies=(sf_true_inputs["occupancies"] / 2).expand(2, -1),
        )
        torch.testing.assert_close(reward(**sf_true_inputs), reward(**ensemble))

    def test_ftotal_modes_diverge_for_distinct_conformer_ensemble(
        self, mtz_path_1vme, sf_ensemble_inputs, device
    ):
        """For a genuine 2-conformer ensemble (altloc A vs B), ``mask(<rho>) != <mask(rho)>``, so
        ``combined`` and ``per_conformer`` give *different* losses.

        This is the behavior that justifies keeping both modes — the agree-cases above (E=1 and
        E=2 *identical* conformers) never exercise the nonlinearity. Here the two frames differ in
        the alternate-conformation coordinates, so the combined mask (built from the summed
        density) and the per-conformer mean of masks genuinely diverge.
        """
        ensemble, ref = sf_ensemble_inputs
        reward_combined = make_prepared_reward(
            mtz_path_1vme,
            ref,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )
        reward_per_conformer = make_prepared_reward(
            mtz_path_1vme,
            ref,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="per_conformer",
            normalize_amplitude=False,
        )
        loss_combined = reward_combined(**ensemble)
        loss_per_conformer = reward_per_conformer(**ensemble)
        assert torch.isfinite(loss_combined) and torch.isfinite(loss_per_conformer)
        assert not torch.allclose(loss_combined, loss_per_conformer)


@pytest.mark.gpu
class TestStructureFactorConfig:
    """Config knobs beyond the fixture default: the raw-amplitude path and reflection selection."""

    def test_perturbed_has_higher_loss_with_unnormalized_amplitude(
        self, mtz_path_1vme, test_coordinates_1vme_sf, sf_true_inputs, device
    ):
        """The ``normalize_amplitude=False`` branch (raw ``|F|`` vs ``sfc.Fo``) runs and ranks
        the true structure below a perturbed one. The normalized path is the fixture default,
        covered by the contract tests."""
        _, atom_array = test_coordinates_1vme_sf
        reward = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            normalize_amplitude=False,
        )
        torch.manual_seed(42)
        coords = sf_true_inputs["coordinates"]
        perturbed = {**sf_true_inputs, "coordinates": coords + torch.randn_like(coords) * 0.5}
        assert reward(**perturbed) > reward(**sf_true_inputs)

    def test_exclude_free_set_drops_reflections(
        self, mtz_path_1vme, test_coordinates_1vme_sf, device
    ):
        """``exclude_free_reflections=True`` drops the R-free test set from the scored mask.

        The synthetic MTZ's free column is ``R-free-flags`` with the test set flagged 1
        (rs/Phenix convention), so SFcalculator is pointed at it explicitly — its defaults
        (``FreeR_flag`` / testset value 0) don't match. Outliers are always excluded regardless.
        """
        _, atom_array = test_coordinates_1vme_sf
        free_flag_kwargs = {"freeflag": "R-free-flags", "testset_value": 1}
        reward_all = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=False,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        reward_work = make_prepared_reward(
            mtz_path_1vme,
            atom_array,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=True,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        assert reward_all.sfc.free_flag.any()  # safety: the free set is actually recognized
        assert int(reward_work._reflection_mask.sum()) < int(reward_all._reflection_mask.sum())
