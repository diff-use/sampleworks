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

``bulk_solvent`` selects the scored amplitude and, when solvent is included, how
it is combined across an ensemble (see :meth:`_compute_ensemble_ftotal`):

* ``"off"`` (default): ``|Fprotein|`` — bare protein, no solvent, no scales.
* ``"combined"``: ``|Ftotal|`` with one bulk-solvent mask from the *combined*
  protein density, ``mask(<rho>)`` — matches the altloc single-structure
  ``Ftotal`` that ``generate_synthetic_sf`` writes to the MTZ.
* ``"per_conformer"``: ``|Ftotal|`` with the mean of the per-conformer masks,
   ``<mask(rho)>`` — the ensemble-averaged bulk solvent. Each of the conformers
  contributes a solvent mask at 1/E weight.

``"combined"`` and ``"per_conformer"`` differ only for a real ensemble
(batch > 1); the mask operator is nonlinear, so ``mask(<rho>) != <mask(rho)>``.
Both use the default, *unrefined* scales ``kiso=1``, ``kmask=0.35``, small
``uaniso``; refining them during sampling is left for a later revision.
``normalize_amplitude`` (``|F|`` vs ``|E|``) is orthogonal and composes with any
``bulk_solvent`` choice.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import reciprocalspaceship as rs
import torch
from jaxtyping import Complex, Float, Int
from loguru import logger
from sampleworks.synthetic.synthetic_utils import atomarray_to_gemmi, resolve_mtz_column
from sampleworks.utils.torch_utils import try_gpu
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser


if TYPE_CHECKING:
    from biotite.structure import AtomArray


# Loss callable: maps (|Fcalc|, |Fobs|) over the masked reflections to a scalar.
AmplitudeLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Bulk-solvent treatment for the scored amplitude. "off": |Fprotein|. "combined":
# |Ftotal| with one mask from the combined density. "per_conformer": |Ftotal| with
# the mean of per-conformer masks (see the class docstring / _compute_ensemble_ftotal).
_BULK_SOLVENT_MODES = ("off", "combined", "per_conformer")

# Tolerances for warning when a caller-supplied unit cell disagrees with the MTZ's,
# passed to gemmi.UnitCell.is_similar: relative tolerance on the cell edges (a, b, c)
# and absolute degrees on the angles (alpha, beta, gamma).
_CELL_LENGTH_REL_TOL = 1e-2
_CELL_ANGLE_DEG_TOL = 0.5


def _resolve_expcolumns(expcolumns: list[str] | None, ds: rs.DataSet) -> list[str]:
    """Resolve the ``[amplitude, sigma]`` columns to read from ``ds``, logging the choice.

    Parameters
    ----------
    expcolumns
        Caller-provided ``[amplitude, sigma]`` to use verbatim, or ``None`` to auto-detect.
    ds
        The parsed MTZ whose amplitude and sigma columns are selected.

    Returns
    -------
    list of str
        The resolved ``[amplitude, sigma]`` pair.

    Raises
    ------
    ValueError
        If a provided ``expcolumns`` is not a length-2 ``[amplitude, sigma]`` pair, or names
        a column absent from the MTZ's amplitude/sigma columns; or, when auto-detecting, if
        the MTZ has zero or more than one amplitude (or sigma) column.

    Notes
    -----
    Auto-detection requires the MTZ to hold exactly one amplitude column and one sigma
    column. A multi-set MTZ (e.g. ``Fprotein`` + ``Ftotal``) is ambiguous and forces the
    caller to pass ``expcolumns`` explicitly.
    """
    amplitude_dtype, sigma_dtype = rs.StructureFactorAmplitudeDtype(), rs.StandardDeviationDtype()
    if expcolumns is not None:
        if len(expcolumns) != 2:
            raise ValueError(
                f"expcolumns must be a [amplitude, sigma] pair; got {list(expcolumns)}."
            )
        amplitude = resolve_mtz_column(ds, amplitude_dtype, column=expcolumns[0])
        sigma = resolve_mtz_column(ds, sigma_dtype, column=expcolumns[1])
        return [amplitude, sigma]

    amplitude = resolve_mtz_column(ds, amplitude_dtype)
    sigma = resolve_mtz_column(ds, sigma_dtype)
    logger.info(
        f"No expcolumns provided; auto-detected SFC columns: "
        f"amplitude='{amplitude}', sigma='{sigma}'."
    )
    return [amplitude, sigma]


class StructureFactorRewardFunction:
    def __init__(
        self,
        mtzfile: str | Path,
        *,
        expcolumns: list[str] | None = None,
        resolution: float | None = None,
        unit_cell: gemmi.UnitCell | None = None,
        space_group: str | None = None,
        scattering_factor_mode: str = "xray",
        bulk_solvent: str = "off",
        loss: AmplitudeLoss | None = None,
        normalize_amplitude: bool = False,
        exclude_free_reflections: bool = False,
        batch_partition: int = 10,
        device: torch.device | None = None,
        sfcalculator_kwargs: dict | None = None,
    ):
        """Reward for fitting structure-factor amplitudes via SFcalculator.

        The reward compares the model amplitudes ``|Fcalc|`` against a target
        ``|Fobs|`` loaded from an MTZ. For an ensemble (batch dimension) of
        ``E`` conformers indexed by ``e``, the per-conformer *protein* structure
        factors are combined by a *complex* sum in the reciprocal space
        (``|sum F| != sum |F|``): ``Fprotein(h) = sum_e F_e(h)``. The atomic
        occupancy is accounted for in the calculation of the per-conformer
        *protein* structure factor F_e(h). ``|Fcalc|`` is then ``|Fprotein|``
        (``bulk_solvent="off"``) or ``|Ftotal|`` once bulk solvent is folded in
        (see ``bulk_solvent``).

        Construction is two-phase: see the module docstring. Call :meth:`prepare`
        with the model atom array before the first ``__call__``.

        Parameters
        ----------
        mtzfile
            Path to the MTZ holding the target amplitudes (and a sigma column).
            Loaded with ``set_experiment=True`` so ``sfc.Fo`` / HKL set / bins are
            populated and the calculation is aligned to the target reflections.
        expcolumns
            Column names ``[amplitude, sigma]`` in the MTZ. If ``None`` (default), the
            amplitude and sigma columns are auto-detected, which requires the MTZ to hold
            exactly one of each; a multi-set MTZ (e.g. ``Fprotein`` + ``Ftotal``) is
            ambiguous and raises, forcing an explicit ``[amplitude, sigma]``. If provided,
            a ``ValueError`` is raised when it is not a length-2 pair or names a column
            absent from the MTZ.
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
        bulk_solvent
            Bulk-solvent treatment, one of ``"off"`` (default; score ``|Fprotein|``),
            ``"combined"`` (``|Ftotal|`` with one mask from the combined density), or
            ``"per_conformer"`` (``|Ftotal|`` with the plain, unweighted mean of the
            per-conformer masks). The per-conformer mean is *not* occupancy-weighted,
            meaning each conformer contributes a solvent mask at 1/E weight. For a
            single conformer, ``"combined"`` and ``"per_conformer"`` coincide. With
            multiple conformers, ``"per_conformer"`` should be a more faithful model.
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
        if device is None:
            device = try_gpu()
        self.device = device
        self.mtzfile = str(mtzfile)
        self.resolution = resolution
        self.scattering_factor_mode = scattering_factor_mode
        if bulk_solvent not in _BULK_SOLVENT_MODES:
            raise ValueError(
                f"bulk_solvent must be one of {_BULK_SOLVENT_MODES}, got {bulk_solvent!r}."
            )
        self.bulk_solvent = bulk_solvent
        self.exclude_free_reflections = exclude_free_reflections
        if batch_partition <= 0:
            raise ValueError(f"batch_partition must be a positive integer, got {batch_partition}.")
        self.batch_partition = batch_partition
        self.loss: AmplitudeLoss = loss if loss is not None else torch.nn.MSELoss()
        self.normalize_amplitude = normalize_amplitude  # |F| vs resolution-bin normalized |E|

        # Resolve crystal metadata / column names against the MTZ.
        self._resolve_mtz_metadata(unit_cell, space_group, expcolumns)

        # All SFcalculator init kwargs are known except `pdbmodel` (needs the model
        # atom array) and `mtzdata` (a copy of the parsed dataset), both injected in
        # prepare(). Caller-supplied sfcalculator_kwargs override these defaults.
        self._sfc_kwargs = dict(
            dmin=self.resolution,
            mode=self.scattering_factor_mode,
            anomalous=False,
            set_experiment=True,
            expcolumns=self.expcolumns,
            device=self.device,
        )
        if sfcalculator_kwargs:
            self._sfc_kwargs.update(sfcalculator_kwargs)

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
        reflections), so it is always read. It is parsed once here and the dataset is
        retained on ``self._mtz_dataset`` so :meth:`prepare` can build ``SFcalculator``
        from it directly instead of re-reading the file. A caller-supplied
        ``unit_cell`` / ``space_group`` (non-``None``) is used but a warning is logged on
        disagreement, so a stale or mismatched override is visible. (SFcalculator reads
        the cell/space group from the gemmi structure built in :meth:`prepare`, not the
        MTZ, so we must supply real values.) ``expcolumns`` is instead validated against
        the MTZ's columns and raises on an unknown name; see :func:`_resolve_expcolumns`.
        """
        self._mtz_dataset = rs.read_mtz(self.mtzfile)
        cell_mtz = self._mtz_dataset.cell
        spacegroup_mtz = (
            self._mtz_dataset.spacegroup.hm if self._mtz_dataset.spacegroup is not None else None
        )

        if unit_cell is not None and not unit_cell.is_similar(
            cell_mtz, _CELL_LENGTH_REL_TOL, _CELL_ANGLE_DEG_TOL
        ):
            logger.warning(
                f"Provided unit_cell {unit_cell.parameters} differs from the MTZ's "
                f"{cell_mtz.parameters}; using the provided value."
            )
        self.unit_cell = unit_cell if unit_cell is not None else cell_mtz

        if space_group is not None and space_group != spacegroup_mtz:
            logger.warning(
                f"Provided space_group {space_group!r} differs from the MTZ's "
                f"{spacegroup_mtz!r}; using the provided value."
            )
        self.space_group = space_group if space_group is not None else spacegroup_mtz

        self.expcolumns = _resolve_expcolumns(expcolumns, self._mtz_dataset)

    def prepare(self, atom_array: AtomArray) -> None:
        """Build the SFcalculator from the model atom array. Here, constructing the
        ``SFcalculator`` consumes the MTZ dataset parsed at ``__init__`` (no second
        file read) and populates the observed structure factor amplitudes ``sfc.Fo``,
        the observed HKL set in the ASU, the resolution bins, the outlier mask
        ``sfc.Outlier``, the R-free flags ``sfc.free_flag``, and the normalized
        ``|Eo|`` in ``sfc.Eo``.

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
        # SFcalculator mutates its mtzdata in place (dropna / hkl_to_asu on the reference),
        # so hand it a copy of the once-parsed dataset; self._mtz_dataset stays pristine and
        # prepare() remains safely re-runnable.
        sfc_kwargs = {"mtzdata": self._mtz_dataset.copy(), **self._sfc_kwargs}
        self.sfc = SFcalculator(pdbmodel=PDBParser(gemmi_structure), **sfc_kwargs)
        # inspect_data estimates solvent percentage and grid size from atom positions
        # and vdW radii, independent of occupancy / B-factor.
        self.sfc.inspect_data()

        # Ftotal uses the default (unrefined) scales kiso=1, kmask=0.35, small uaniso,
        # matching generate_synthetic_sf's Ftotal. They don't depend on coordinates, so
        # set them once here. _set_scales only needs atom_pos_frac (set at construction)
        # and n_bins (set by the experiment init) for dtype/device and bin count.
        if self.bulk_solvent != "off":
            self.sfc._set_scales(requires_grad=False)

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
            mask_np &= ~self.sfc.free_flag
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

        fcalc = Fprotein if self.bulk_solvent == "off" else self._compute_ensemble_ftotal(Fprotein)

        mask = self._reflection_mask
        if self.normalize_amplitude:
            calc = self.sfc.calc_Ec(fcalc).abs()
            obs = self.sfc.Eo
        else:
            calc = torch.abs(fcalc)
            obs = self.sfc.Fo
        return self.loss(calc[mask], obs[mask])

    def _compute_ensemble_ftotal(
        self,
        Fprotein_HKL: Complex[torch.Tensor, "n_hkl"],  # noqa: F821, UP037
    ) -> Complex[torch.Tensor, "n_hkl"]:  # noqa: F821, UP037
        """Add default-scaled bulk solvent to the ensemble Fprotein to form Ftotal.

        ``Ftotal(h) = kiso * aniso(h) * (Fprotein(h) + kmask * Fmask(h))`` with the
        default (unrefined) scales set in :meth:`prepare`, evaluated on the
        experimental HKL set. ``Fprotein`` is the ensemble complex sum; the
        bulk-solvent ``Fmask`` is combined per :attr:`bulk_solvent`. The two
        ``|Ftotal|`` modes are identical for a single conformer.

        ``bulk_solvent="combined"`` (``mask(<rho>)``)
            One mask built from the combined protein density. ``calc_fsolvent``
            builds the mask by FFT over the full ASU set, so the combined
            ``Fprotein_asu`` (not just the HKL subset) is fed to it. Matches the
            altloc single-structure Ftotal in the synthetic MTZ.

        ``bulk_solvent="per_conformer"`` (``<mask(rho)>``)
            Average of the per-conformer bulk solvent masks. ``rsgrid2realmask``
            normalizes the protein density and cuts at a quantile, so each conformer
            contributes ``Fmask`` in a scale-invariant way. All conformers are assumed
            to contribute the solvent mask equally (1/E weight).

        Parameters
        ----------
        Fprotein_HKL
            Ensemble complex-sum Fprotein on the experimental HKL set ``[n_hkl]``.

        Returns
        -------
        torch.Tensor
            Complex Ftotal on the experimental HKL set ``[n_hkl]``.
        """
        assert self.sfc is not None  # prepare() built it; __call__ guards before dispatching here
        self.sfc.Fprotein_HKL = Fprotein_HKL  # drives calc_ftotal on the HKL set
        if self.bulk_solvent == "per_conformer":
            # calc_fsolvent_batch masks each conformer (from Fprotein_asu_batch, set by
            # calc_fprotein_batch); the mean applies the 1/E weight -> <mask(rho)>.
            Fmask_HKL_batch = self.sfc.calc_fsolvent_batch(
                Return=True, PARTITION=self.batch_partition
            )
            self.sfc.Fmask_HKL = Fmask_HKL_batch.mean(dim=0)
        else:
            # One mask from the combined density -> mask(<rho>). SFC calc_solvent() requires
            # having the Fprotein_asu set, but batch mode only sets Fprotein_asu_batch.
            self.sfc.Fprotein_asu = self.sfc.Fprotein_asu_batch.sum(dim=0)
            self.sfc.calc_fsolvent()  # sets Fmask_HKL
        return self.sfc.calc_ftotal()  # default scales set in prepare()
