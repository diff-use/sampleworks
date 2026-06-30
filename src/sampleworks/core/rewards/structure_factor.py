"""Reciprocal-space reward function for structure-factor amplitudes.

Scores structures against experimental (or synthetic) structure-factor
amplitudes ``|Fobs|`` using ``SFcalculator`` from ``SFC_Torch``. This is the
reciprocal-space counterpart to :class:`RealSpaceRewardFunction` in
``real_space_density.py``.

Unlike the real-space reward, ``SFcalculator`` needs the full topology
(``PDBParser`` -> ``gemmi.Structure``: atom names/elements, unit cell, space
group, ``resolution`` -> the HKL set) at construction. That information is only
available once the model atom array is known, i.e. *after*
``process_structure_to_trajectory_input`` runs inside ``sample()``. We
therefore split construction in two:

* ``__init__`` stores only the up-front config (target MTZ, ``resolution``,
  ``scattering_factor_mode``, unit cell, space group, loss, device); it does
  *not* build ``SFcalculator``.
* :meth:`prepare` builds ``SFcalculator`` from the model atom array. The caller
  (a step scaler) is responsible for invoking it before the first ``__call__``.

v1 computes ``|Fprotein|`` (no bulk solvent, no scales). The roadmap to
``|Ftotal|`` reuses the same ``__call__`` shape via ``calc_fsolvent_batch`` /
``calc_ftotal_batch`` with frozen, periodically-refit scales.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import torch
from jaxtyping import Float, Int
from loguru import logger
from sampleworks.eval.synthetic_utils import atomarray_to_gemmi
from sampleworks.utils.torch_utils import try_gpu
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser


if TYPE_CHECKING:
    from biotite.structure import AtomArray


# Loss callable: maps (|Fcalc|, |Fobs|) over the masked reflections to a scalar.
AmplitudeLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Tolerances for warning when a caller-supplied unit cell disagrees with the MTZ's,
# passed to gemmi.UnitCell.is_similar: relative tolerance on the cell edges (a, b, c)
# and absolute degrees on the angles (alpha, beta, gamma).
_CELL_LENGTH_REL_TOL = 1e-2
_CELL_ANGLE_DEG_TOL = 0.5


def _detect_mtz_metadata(
    mtzfile: str,
) -> tuple[gemmi.UnitCell, str | None, list[str], list[str]]:
    """Read crystal metadata and experimental-column candidates from an MTZ.

    Parameters
    ----------
    mtzfile
        Path to the MTZ to read.

    Returns
    -------
    tuple
        ``(unit_cell, space_group_hm, amplitude_columns, sigma_columns)``, where the last
        two are *all* structure-factor-amplitude and standard-deviation columns found.
        Pairing/selecting among them is left to :func:`_resolve_expcolumns`.

    Raises
    ------
    ValueError
        If the MTZ has no amplitude column or no sigma column.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(mtzfile)
    amplitude_cols = list(ds.select_mtzdtype(rs.StructureFactorAmplitudeDtype()).columns)
    sigma_cols = list(ds.select_mtzdtype(rs.StandardDeviationDtype()).columns)
    if not amplitude_cols or not sigma_cols:
        raise ValueError(
            f"MTZ '{mtzfile}' needs both amplitude F and sigma SIGF columns for SFC; "
            f"found amplitudes:{amplitude_cols}, sigmas:{sigma_cols}."
        )
    spacegroup = ds.spacegroup.hm if ds.spacegroup is not None else None
    return ds.cell, spacegroup, amplitude_cols, sigma_cols


def _resolve_expcolumns(
    expcolumns: list[str] | None,
    amplitude_cols: list[str],
    sigma_cols: list[str],
) -> list[str]:
    """Resolve the ``[amplitude, sigma]`` columns to read, logging the choice.

    Parameters
    ----------
    expcolumns
        Caller-provided ``[amplitude, sigma]`` to use verbatim, or ``None`` to auto-detect.
    amplitude_cols, sigma_cols
        Candidate amplitude and standard-deviation columns detected in the MTZ.

    Returns
    -------
    list of str
        The resolved ``[amplitude, sigma]`` pair.

    Raises
    ------
    ValueError
        If a provided ``expcolumns`` name is not an amplitude/sigma column in the MTZ.

    Notes
    -----
    Auto-detection takes the first amplitude column and attempts to find a matching sigma
    (e.g. ``Fprotein`` -> ``SIGFprotein``; case-insensitive), defaulting to the first sigma
    column if no match is found.
    """
    if expcolumns is not None:
        valid = set(amplitude_cols) | set(sigma_cols)
        unknown = [c for c in expcolumns if c not in valid]
        if unknown:
            raise ValueError(
                f"Provided expcolumns {list(expcolumns)} include {unknown}, not found as MTZ "
                f"amplitude/sigma columns (amplitudes={amplitude_cols}, sigmas={sigma_cols})."
            )
        return list(expcolumns)

    amplitude = amplitude_cols[0]
    if len(amplitude_cols) > 1:
        logger.warning(
            f"MTZ has multiple amplitude columns {amplitude_cols}; auto-selected the first, "
            f"'{amplitude}'. Pass expcolumns=[amplitude, sigma] to choose another."
        )
    expected_sigma = f"sig{amplitude}".lower()
    sigma = next((s for s in sigma_cols if s.lower() == expected_sigma), None)
    if sigma is None:
        sigma = sigma_cols[0]
        logger.warning(
            f"No 'SIG{amplitude}' column to match the auto-selected amplitude; "
            f"falling back to the first sigma column '{sigma}'."
        )
    logger.info(
        f"No expcolumns provided; auto-detected SFC columns: "
        f"amplitude='{amplitude}', sigma='{sigma}'."
    )
    return [amplitude, sigma]


class StructureFactorRewardFunction:
    """Reward for fitting structure-factor amplitudes via SFcalculator.

    The reward compares the model amplitudes ``|Fcalc|`` against a target
    ``|Fobs|`` loaded from an MTZ. For an ensemble (batch dimension), the
    members are combined as a single multi-conformer model in reciprocal space:
    ``F_total(h) = (1/N) * sum_e F_e(h)`` (a *complex* sum, since
    ``|sum F| != sum |F|``), then ``|F_total|`` is compared to ``|Fobs|``. This
    mirrors the real-space reward's sum-over-batch density semantics.

    Construction is two-phase: see the module docstring. Call :meth:`prepare`
    with the model atom array before the first ``__call__``.

    Parameters
    ----------
    mtzfile
        Path to the MTZ holding the target amplitudes (and a sigma column).
        Loaded with ``set_experiment=True`` so ``sfc.Fo`` / HKL set / bins are
        populated and the calculation is aligned to the target reflections.
    expcolumns
        Column names ``[amplitude, sigma]`` in the MTZ. If ``None`` (default), the first
        amplitude column is auto-detected and paired with its ``SIG`` + amplitude sigma
        (e.g. ``Fprotein`` -> ``SIGFprotein``); if provided, a ``ValueError`` is raised when
        a name is absent from the MTZ.
    resolution
        High-resolution limit (dmin) in Angstrom, or ``None`` (default) to use
        the MTZ's own resolution. The HKL set comes from the ``mtzfile``; when
        given, ``resolution`` further truncates it.
    unit_cell
        Crystallographic unit cell, stamped onto the gemmi structure built in
        :meth:`prepare` (SFcalculator reads the cell from there, not the MTZ).
        If ``None`` (default), read from the ``mtzfile``; if provided, used as-is
        but a warning is logged when it disagrees with the MTZ.
    space_group
        Space group as a Hermann-Mauguin string, stamped onto the gemmi
        structure. Same MTZ-default / warn-on-mismatch behavior as ``unit_cell``.
    scattering_factor_mode
        SFcalculator scattering mode: ``"xray"`` or ``"cryoem"``.
    loss
        Callable ``(|Fcalc|, |Fobs|) -> scalar`` over masked reflections.
        Defaults to mean-squared error on amplitudes. Pass any callable
        (L1, R-factor-style, resolution-weighted, ...) to customize.
    normalize_amplitude
        If True, score normalized structure factors (E-values) instead of ``|F|``.
    exclude_free_reflections
        If True, drop the R-free test-set reflections from the loss (use only
        the working set). Default False: use all non-outlier reflections.
    batch_partition
        Ensemble chunk size forwarded to ``SFcalculator.calc_fprotein_batch`` as its
        ``PARTITION`` parameter. SFC's own default (20) can still lead to OOM.
    device
        Torch device. Auto-selects a GPU when omitted.
    sfcalculator_kwargs
        Extra keyword arguments forwarded verbatim to ``SFcalculator(...)`` in
        :meth:`prepare`, overriding the defaults set here (e.g. ``n_bins``,
        ``anomalous``, ``freeflag``). Minimal escape hatch until these are given
        a typed config.
    """

    def __init__(
        self,
        mtzfile: str | Path,
        *,
        expcolumns: list[str] | None = None,
        resolution: float | None = None,
        unit_cell: gemmi.UnitCell | None = None,
        space_group: str | None = None,
        scattering_factor_mode: str = "xray",
        loss: AmplitudeLoss | None = None,
        normalize_amplitude: bool = False,
        exclude_free_reflections: bool = False,
        batch_partition: int = 10,
        device: torch.device | None = None,
        sfcalculator_kwargs: dict | None = None,
    ):
        if device is None:
            device = try_gpu()
        self.device = device
        self.mtzfile = str(mtzfile)
        self.resolution = resolution
        self.scattering_factor_mode = scattering_factor_mode
        self.exclude_free_reflections = exclude_free_reflections
        if batch_partition <= 0:
            raise ValueError(f"batch_partition must be a positive integer, got {batch_partition}.")
        self.batch_partition = batch_partition
        self.loss: AmplitudeLoss = loss if loss is not None else torch.nn.MSELoss()
        self.normalize_amplitude = normalize_amplitude  # |F| vs resolution-bin normalized |E|
        self.sfcalculator_kwargs = dict(sfcalculator_kwargs) if sfcalculator_kwargs else {}

        # Resolve crystal metadata / column names against the MTZ.
        self._resolve_mtz_metadata(unit_cell, space_group, expcolumns)

        # All SFcalculator init kwargs are known except `pdbmodel` (needs the model
        # atom array), which is injected in prepare().
        self._sfc_kwargs: dict = dict(
            mtzdata=self.mtzfile,
            dmin=self.resolution,
            mode=self.scattering_factor_mode,
            anomalous=False,
            set_experiment=True,
            expcolumns=self.expcolumns,
            device=self.device,
        )
        self._sfc_kwargs.update(self.sfcalculator_kwargs)

        # Populated by prepare(); None until then.
        self.sfc: SFcalculator | None = None
        self._reflection_mask: torch.Tensor | None = None

    def _resolve_mtz_metadata(
        self,
        unit_cell: gemmi.UnitCell | None,
        space_group: str | None,
        expcolumns: list[str] | None,
    ) -> None:
        """Set ``self.unit_cell`` / ``space_group`` / ``expcolumns``, resolved against the MTZ.

        The MTZ is the source of truth (it must be consistent with its own
        reflections), so it is always read. For any argument the caller supplied
        (non-``None``), that value is used but a warning is logged on disagreement, so a
        stale or mismatched override is visible. (SFcalculator reads the cell/space group
        from the gemmi structure built in :meth:`prepare`, not the MTZ, so we must supply
        real values.)
        """
        cell, spacegroup, amplitude_cols, sigma_cols = _detect_mtz_metadata(self.mtzfile)

        if unit_cell is not None and not unit_cell.is_similar(
            cell, _CELL_LENGTH_REL_TOL, _CELL_ANGLE_DEG_TOL
        ):
            logger.warning(
                f"Provided unit_cell {unit_cell.parameters} differs from the MTZ's "
                f"{cell.parameters}; using the provided value."
            )
        self.unit_cell = unit_cell if unit_cell is not None else cell

        if space_group is not None and space_group != spacegroup:
            logger.warning(
                f"Provided space_group {space_group!r} differs from the MTZ's "
                f"{spacegroup!r}; using the provided value."
            )
        self.space_group = space_group if space_group is not None else spacegroup

        self.expcolumns = _resolve_expcolumns(expcolumns, amplitude_cols, sigma_cols)

    def prepare(self, atom_array: AtomArray) -> None:
        """Build the SFcalculator from the model atom array.

        Must be called once before the first ``__call__``, with the same atom
        array that the sampled coordinates correspond to (model atom space:
        ``model_atom_array or atom_array``). The atom ordering of ``atom_array``
        defines the column order of the coordinate tensor passed to ``__call__``.

        Per-atom B-factors / occupancy are set per ``__call__`` (not here), leaving
        the door open to refining them during sampling.

        Parameters
        ----------
        atom_array
            Biotite AtomArray for the atoms the model operates on. Needs
            ``chain_id``, ``res_id``, ``res_name``, ``atom_name``, ``element``
            annotations (its ``b_factor``/``occupancy`` are baked as defaults but
            overridden each ``__call__``). A missing ``altloc_id`` is defaulted to
            blank inside ``atomarray_to_gemmi``.
        """
        gemmi_structure = atomarray_to_gemmi(
            atom_array,
            unit_cell=self.unit_cell,
            space_group=self.space_group,
        )
        self.sfc = SFcalculator(pdbmodel=PDBParser(gemmi_structure), **self._sfc_kwargs)
        # inspect_data estimates solvent percentage and grid size from atom positions
        # and vdW radii, independent of occupancy / B-factor.
        self.sfc.inspect_data()

        # |Eo| are computed in SFC's experiment init (inside a try/except).
        if self.normalize_amplitude and getattr(self.sfc, "Eo", None) is None:
            raise RuntimeError(
                "normalize_amplitude=True requires sfc.Eo, but "
                "SFcalculator did not populate them from this MTZ."
            )

        # Reflection mask: drop outliers, and optionally the free (test) set, when
        # computing the loss. Outlier / free_flag are numpy bool arrays from SFC.
        mask_np = ~self.sfc.Outlier
        if self.exclude_free_reflections:
            mask_np = mask_np & ~self.sfc.free_flag
        self._reflection_mask = torch.from_numpy(mask_np).to(self.device)

        logger.info(
            f"Prepared StructureFactorRewardFunction: n_atoms={len(self.sfc.atom_pos_orth)}, "
            f"n_reflections={len(self.sfc.Fo)}, n_used={int(mask_np.sum())}, "
            f"cell={self.sfc.unit_cell}, space_group={self.sfc.space_group.hm}, "
            f"solventpct={self.sfc.solventpct}, gridsize={self.sfc.gridsize}"
        )

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute the amplitude loss for the (ensemble of) coordinates.

        Call ``.backward()`` on the result to get gradients w.r.t.
        ``coordinates``.

        ``elements`` is ignored (topology is fixed in the SFcalculator built by
        :meth:`prepare`). ``b_factors`` and ``occupancies`` are reset onto the
        SFcalculator each call (mirroring ``RealSpaceRewardFunction``), and must be
        broadcast-identical across the batch dim (enforced; SFcalculator has no
        per-conformer occupancy/B axis, so non-broadcast input raises ``ValueError``).

        Parameters
        ----------
        coordinates
            Atomic coordinates ``[batch, n_atoms, 3]`` in model atom space,
            matching the atom ordering passed to :meth:`prepare`.
        b_factors
            Per-atom isotropic B-factors ``[batch, n_atoms]``, written to
            ``sfc.atom_b_iso`` (reconciled: real deposited where shared with the
            structure, 20.0 for model-only / NaN atoms).
        occupancies
            Per-atom occupancies ``[batch, n_atoms]`` (uniform ``1/E`` from the
            pipeline), written to ``sfc.atom_occ``; the ``1/E`` weighting makes the
            complex ensemble sum the multi-conformer total.

        Returns
        -------
        torch.Tensor
            Scalar reward (loss).
        """
        if self.sfc is None or self._reflection_mask is None:
            raise RuntimeError(
                "StructureFactorRewardFunction.prepare() must be called with the model "
                "atom array before the reward is evaluated."
            )

        # SFcalculator has no per-conformer (batch) occupancy/B axis, so these must be shared
        # across the ensemble; row 0 is used. Reject non-broadcast input as a guard.
        for name, tensor in (("occupancy", occupancies), ("B-factor", b_factors)):
            if not torch.equal(tensor, tensor[:1].expand_as(tensor)):
                raise ValueError(
                    f"StructureFactorRewardFunction requires {name} identical across the "
                    "batch dim (SFcalculator has no per-conformer occupancy/B axis); got "
                    "per-conformer values."
                )
        self.sfc.atom_b_iso = b_factors[0]
        self.sfc.atom_occ = occupancies[0]

        # Multi-conformer combination: complex sum over the ensemble [batch, n_hkl].
        # occ = 1/E (set per call) makes the summed |F| the multi-conformer total.
        Fprotein_batch = self.sfc.calc_fprotein_batch(
            coordinates, Return=True, PARTITION=self.batch_partition
        )
        Fprotein = Fprotein_batch.sum(dim=0)

        mask = self._reflection_mask
        if self.normalize_amplitude:
            calc = self.sfc.calc_Ec(Fprotein).abs()
            obs = self.sfc.Eo
        else:
            # Raw amplitudes (arbitrary scale; relative comparisons only).
            calc = torch.abs(Fprotein)
            obs = self.sfc.Fo
        return self.loss(calc[mask], obs[mask])
