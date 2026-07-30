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
from pathlib import Path
from unittest import mock

import gemmi
import numpy as np
import pytest
import reciprocalspaceship as rs
import torch
from biotite.structure import AtomArray
from reciprocalspaceship.utils import add_rfree
from sampleworks.core.rewards.structure_factor import (
    _MIN_RETAINED_REFLECTION_FRACTION,
    _MIN_RETAINED_REFLECTIONS,
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
    rf = StructureFactorRewardFunction(mtz_path, **kwargs)
    rf.prepare(atom_array, device=device)
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
    """Construction-time behavior, on CPU: no GPU and no ``prepare()``/SF compute.

    ``__init__`` reads exactly three things from the MTZ (``_resolve_mtz_metadata``): the unit
    cell, the space group, and the amplitude/sigma column layout. For efficiency, we build a
    ``toy_multi_set_mtz`` instead of taking the session-scoped ``mtz_path_1vme`` that on a CPU
    costs ~18 s.
    """

    @pytest.fixture
    def toy_unit_cell(self) -> gemmi.UnitCell:
        """The unit cell written into ``toy_multi_set_mtz``."""
        return gemmi.UnitCell(11.0, 22.0, 33.0, 90.0, 100.0, 120.0)

    @pytest.fixture
    def toy_space_group(self) -> str:
        """The space group (Hermann-Mauguin string) written into ``toy_multi_set_mtz``."""
        return "P 1 2 1"

    @pytest.fixture
    def protein_columns(self) -> list[str]:
        """The ``[amplitude, sigma]`` column names for protein structure factors."""
        return ["Fprotein", "SIGFprotein"]

    @pytest.fixture
    def total_columns(self) -> list[str]:
        """The ``[amplitude, sigma]`` column names for total structure factors."""
        return ["Ftotal", "SIGFtotal"]

    @pytest.fixture
    def toy_multi_set_mtz(
        self,
        tmp_path: Path,
        toy_unit_cell: gemmi.UnitCell,
        toy_space_group: str,
        protein_columns: list[str],
        total_columns: list[str],
    ) -> Path:
        """A minimal multi-set MTZ carrying header metadata on unit cell and space group."""
        hkl = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [2, 1, 0]], dtype=np.int32)
        amplitudes = np.array([100.0, 90.0, 80.0, 70.0, 60.0], dtype=np.float32)
        columns = {"H": hkl[:, 0], "K": hkl[:, 1], "L": hkl[:, 2]}
        for (amplitude_col, sigma_col), scale in ((protein_columns, 1.0), (total_columns, 1.1)):
            columns[amplitude_col] = amplitudes * scale
            columns[sigma_col] = amplitudes / 10.0
        dataset = rs.DataSet(
            columns, cell=toy_unit_cell, spacegroup=toy_space_group
        ).infer_mtz_dtypes()
        dataset.set_index(["H", "K", "L"], inplace=True)
        path = tmp_path / "toy_multi_set.mtz"
        dataset.write_mtz(str(path))
        return path

    def test_multi_set_mtz_requires_expcolumns(self, toy_multi_set_mtz):
        """A multi-set MTZ with no ``expcolumns`` is ambiguous and fails fast, rather than
        silently auto-selecting the first amplitude/sigma pair.
        """
        with pytest.raises(ValueError, match="Multiple SFAmplitude columns"):
            StructureFactorRewardFunction(toy_multi_set_mtz)

    def test_explicit_expcolumns_override_selection(self, toy_multi_set_mtz, total_columns):
        """Explicit ``expcolumns`` are used verbatim, overriding the auto-selected first pair."""
        reward = StructureFactorRewardFunction(toy_multi_set_mtz, expcolumns=total_columns)
        assert reward.expcolumns == total_columns

    def test_unknown_expcolumns_raise(self, toy_multi_set_mtz, protein_columns):
        """Explicit ``expcolumns`` naming a column absent from the MTZ fail fast at construction."""
        with pytest.raises(ValueError, match="is not among the dataset's SFAmplitude columns"):
            StructureFactorRewardFunction(
                toy_multi_set_mtz, expcolumns=["Fnonexistent", protein_columns[1]]
            )

    @pytest.mark.parametrize(
        "bad_expcolumns",
        [
            ["Fprotein"],  # missing the sigma
            ["Fprotein", "SIGFprotein", "Ftotal"],  # too many
            [],
            "FP",  # a bare string of length 2 would otherwise split into ["F", "P"]
        ],
    )
    def test_malformed_expcolumns_raise(self, toy_multi_set_mtz, bad_expcolumns):
        """``expcolumns`` that is not an ``[amplitude, sigma]`` pair fails fast, including a
        bare string, which would otherwise be indexed character-wise into a bogus column pair.
        """
        with pytest.raises(ValueError, match=r"expcolumns must be a \[amplitude, sigma\] pair"):
            StructureFactorRewardFunction(toy_multi_set_mtz, expcolumns=bad_expcolumns)

    def test_cell_and_space_group_read_from_mtz(
        self, toy_multi_set_mtz, toy_unit_cell, toy_space_group, total_columns
    ):
        """With no caller override, the cell and space group are parsed off the MTZ."""
        reward = StructureFactorRewardFunction(toy_multi_set_mtz, expcolumns=total_columns)
        assert reward.space_group == toy_space_group
        assert reward.unit_cell.parameters == pytest.approx(toy_unit_cell.parameters, abs=1e-3)

    def test_provided_cell_and_space_group_override_mtz_with_warning(
        self, toy_multi_set_mtz, total_columns, caplog
    ):
        """A caller-supplied cell / space group disagreeing with the MTZ is used and warned."""
        override_cell = gemmi.UnitCell(80.0, 90.0, 100.0, 90.0, 90.0, 90.0)
        override_space_group = "P 21 21 21"
        with caplog.at_level(logging.WARNING):
            reward = StructureFactorRewardFunction(
                toy_multi_set_mtz,
                expcolumns=total_columns,
                unit_cell=override_cell,
                space_group=override_space_group,
            )
        assert "Provided unit_cell" in caplog.text
        assert "Provided space_group" in caplog.text
        assert reward.space_group == override_space_group
        assert reward.unit_cell.parameters == pytest.approx(override_cell.parameters, abs=1e-3)

    @pytest.mark.parametrize("bad_partition", [0, -5, 10.0, 2.5, "10", None, True])
    def test_invalid_batch_partition_raises(self, toy_multi_set_mtz, bad_partition):
        """A non-integer or non-positive ``batch_partition`` (an OOM knob) fails fast.

        The check precedes column resolution, so no ``expcolumns`` are needed.
        """
        with pytest.raises(ValueError, match="batch_partition must be a positive integer"):
            StructureFactorRewardFunction(toy_multi_set_mtz, batch_partition=bad_partition)


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
        """A 2-conformer ensemble (altloc A vs B) as ``(__call__ kwargs, state-A reference)``.

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
        altloc_a, altloc_b = altloc_ids
        array_a, array_b = build_pairwise_altloc_arrays(aa, altloc_ids)[(altloc_a, altloc_b)]
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
        self, mtz_path_1vme, structure_1vme_sf, sf_true_inputs, device
    ):
        """Adding default-scaled bulk solvent (``combined``) fits the synthetic ``Ftotal``
        column far better than protein-only (``off``) — the synthetic ground truth was generated
        with the same default scales."""
        reward_with_solvent_off = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="off",
            normalize_amplitude=False,
        )
        reward_with_solvent_combined = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )
        loss_combined = reward_with_solvent_combined(**sf_true_inputs)
        loss_off = reward_with_solvent_off(**sf_true_inputs)
        assert loss_combined < loss_off

    def test_ftotal_modes_agree_for_single_conformer(
        self, mtz_path_1vme, structure_1vme_sf, sf_true_inputs, device
    ):
        """For a single conformer (E=1) ``mask(<rho>)`` and ``<mask(rho)>`` are the same mask,
        so ``combined`` and ``per_conformer`` give the same loss."""
        reward_with_solvent_combined = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Ftotal", "SIGFtotal"],
            bulk_solvent="combined",
            normalize_amplitude=False,
        )
        reward_with_solvent_per_conformer = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
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
        self, mtz_path_1vme, structure_1vme_sf, sf_true_inputs, device
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
        reward = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
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

        The mock.patch.object supplements the numerical difference test by checking the dispatch
        logic and ensure that the correct SFcalculator method for Fsolvent is called.
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
        # Specifcially checking the dispatch logic in _compute_ensemble_ftotal, where
        # ``bulk_solvent="per_conformer"`` should route to a calc_fsolvent_batch call.
        with mock.patch.object(
            reward_per_conformer.sfc,
            "calc_fsolvent_batch",
            wraps=reward_per_conformer.sfc.calc_fsolvent_batch,
        ) as sfc_calc_fsolvent_batch:
            loss_per_conformer = reward_per_conformer(**ensemble)
        assert sfc_calc_fsolvent_batch.call_count == 1
        assert torch.isfinite(loss_combined) and torch.isfinite(loss_per_conformer)
        assert not torch.allclose(loss_combined, loss_per_conformer), (
            f"combined and per_conformer agree: {loss_combined.item()} vs "
            f"{loss_per_conformer.item()}"
        )


@pytest.mark.gpu
class TestStructureFactorConfig:
    """Config knobs beyond the fixture default: the raw-amplitude path and reflection selection."""

    def test_perturbed_has_higher_loss_with_unnormalized_amplitude(
        self, mtz_path_1vme, structure_1vme_sf, sf_true_inputs, device
    ):
        """The ``normalize_amplitude=False`` branch (raw ``|F|`` vs ``sfc.Fo``) runs and ranks
        the true structure below a perturbed one. The normalized path is the fixture default,
        covered by the contract tests."""
        reward = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            normalize_amplitude=False,
        )
        torch.manual_seed(42)
        coords = sf_true_inputs["coordinates"]
        perturbed = {**sf_true_inputs, "coordinates": coords + torch.randn_like(coords) * 0.5}
        assert reward(**perturbed) > reward(**sf_true_inputs)

    def test_exclude_free_set_drops_reflections(self, mtz_path_1vme, structure_1vme_sf, device):
        """``exclude_free_reflections=True`` drops the R-free test set from the scored mask.

        The synthetic MTZ's free column is ``R-free-flags`` with the test set flagged 1
        (rs/Phenix convention), so SFcalculator is pointed at it explicitly — its defaults
        (``FreeR_flag`` / testset value 0) don't match. Outliers are always excluded regardless.
        """
        free_flag_kwargs = {"freeflag": "R-free-flags", "testset_value": 1}
        reward_all = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=False,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        reward_work = make_prepared_reward(
            mtz_path_1vme,
            structure_1vme_sf,
            device,
            expcolumns=["Fprotein", "SIGFprotein"],
            exclude_free_reflections=True,
            sfcalculator_kwargs=free_flag_kwargs,
        )
        assert reward_all.sfc.free_flag.any()  # safety: the free set is actually recognized
        assert int(reward_work._reflection_mask.sum()) < int(reward_all._reflection_mask.sum())

    @pytest.fixture
    def mtz_all_free_1vme(self, mtz_path_1vme, tmp_path) -> Path:
        """The test MTZ with *every* reflection flagged as the R-free test set.

        Built with rs's own ``add_rfree`` (``fraction=1.0``) so the flag column name and
        convention match the committed MTZ's (``R-free-flags``, test set = 1).
        """
        dataset = add_rfree(rs.read_mtz(str(mtz_path_1vme)), fraction=1.0, seed=0)
        path = tmp_path / "1vme_all_free.mtz"
        dataset.write_mtz(str(path))
        return path

    def test_empty_reflection_mask_raises(self, mtz_all_free_1vme, structure_1vme_sf, device):
        """A mask that scores no reflection fails in ``prepare()``."""
        with pytest.raises(ValueError, match="No reflections remain"):
            make_prepared_reward(
                mtz_all_free_1vme,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                exclude_free_reflections=True,
                sfcalculator_kwargs={"freeflag": "R-free-flags", "testset_value": 1},
            )

    def test_inverted_testset_value_warns(self, mtz_path_1vme, structure_1vme_sf, device, caplog):
        """An inverted ``testset_value`` keeps only the test set, and is warned about.

        Inverting the flag keeps the fraction of valid reflections small although the
        asbolute size can still be large.
        """
        with caplog.at_level(logging.WARNING):
            reward = make_prepared_reward(
                mtz_path_1vme,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                exclude_free_reflections=True,
                sfcalculator_kwargs={"freeflag": "R-free-flags", "testset_value": 0},
            )
        n_used, n_total = int(reward._reflection_mask.sum()), len(reward.sfc.Fo)
        # check that the fraction precondition held and the floor's precondition did not hold
        assert n_used < _MIN_RETAINED_REFLECTION_FRACTION * n_total
        assert n_used >= _MIN_RETAINED_REFLECTIONS
        assert f"Only {n_used}/{n_total} reflections remain" in caplog.text
        # check that only the warning message from the fraction threshold is logged
        assert "testset_value" in caplog.text
        assert "the MTZ reflection range" not in caplog.text

    def test_small_reflection_set_warns(self, mtz_path_1vme, structure_1vme_sf, device, caplog):
        """A reflection set too small of absolute size to guide coordinates should warn.

        Truncating to 10 A leaves only a few hundred reflections but the fraction of valid
        reflections can still be large.
        """
        with caplog.at_level(logging.WARNING):
            reward = make_prepared_reward(
                mtz_path_1vme,
                structure_1vme_sf,
                device,
                expcolumns=["Fprotein", "SIGFprotein"],
                resolution=10.0,
            )
        n_used, n_total = int(reward._reflection_mask.sum()), len(reward.sfc.Fo)
        # check that the floor precondition held and the fraction's precondition did not hold
        assert n_used < _MIN_RETAINED_REFLECTIONS
        assert n_used >= _MIN_RETAINED_REFLECTION_FRACTION * n_total
        # check that only the warning message from the absolute threshold is logged
        assert "the MTZ reflection range" in caplog.text
        assert f"{n_used}/{n_total}" not in caplog.text
