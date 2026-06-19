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


def _detect_mtz_metadata(mtzfile: str) -> tuple[gemmi.UnitCell, str | None, list[str]]:
    """Read crystal metadata and the experimental column names from an MTZ.

    Returns ``(unit_cell, space_group_hm, [amplitude_column, sigma_column])``. The
    columns are the first structure-factor-amplitude and standard-deviation columns
    (mirroring the amplitude auto-detection in
    ``generate_synthetic_sf.process_amplitudes_to_dataset``, but detecting the sigma
    column too rather than assuming a ``SIG`` prefix). Used to fill any crystal /
    column config the caller left as ``None``.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(mtzfile)
    amplitude_cols = ds.select_mtzdtype(rs.StructureFactorAmplitudeDtype()).columns
    sigma_cols = ds.select_mtzdtype(rs.StandardDeviationDtype()).columns
    if len(amplitude_cols) == 0 or len(sigma_cols) == 0:
        raise ValueError(
            f"MTZ '{mtzfile}' needs a structure-factor-amplitude column and a "
            f"standard-deviation column; found amplitudes={list(amplitude_cols)}, "
            f"sigmas={list(sigma_cols)}."
        )
    spacegroup = ds.spacegroup.hm if ds.spacegroup is not None else None
    return ds.cell, spacegroup, [amplitude_cols[0], sigma_cols[0]]


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
    expcolumns
        Column names ``[amplitude, sigma]`` in the MTZ. If ``None`` (default), the
        first structure-factor-amplitude and standard-deviation columns are
        auto-detected from the ``mtzfile``; if provided, used as-is but a warning is
        logged when they differ from the detected columns.
    exclude_free_reflections
        If True, drop the R-free test-set reflections from the loss (use only
        the working set). Default False: use all non-outlier reflections.
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
        exclude_free_reflections: bool = False,
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
        self.loss: AmplitudeLoss = loss if loss is not None else torch.nn.MSELoss()
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
        cell, spacegroup, columns = _detect_mtz_metadata(self.mtzfile)

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

        if expcolumns is not None and list(expcolumns) != list(columns):
            logger.warning(
                f"Provided expcolumns {expcolumns} differ from the MTZ's detected "
                f"{columns}; using the provided value."
            )
        self.expcolumns = expcolumns if expcolumns is not None else columns

    def prepare(
        self,
        atom_array: AtomArray,
        b_factors: Float[torch.Tensor, " n_atoms"],
        occupancies: Float[torch.Tensor, " n_atoms"],
    ) -> None:
        """Build the SFcalculator from the model atom array and per-atom B/occ.

        Must be called once before the first ``__call__``, with the same atom
        array that the sampled coordinates correspond to (model atom space:
        ``model_atom_array or atom_array``). The atom ordering of ``atom_array``
        defines the column order of the coordinate tensor passed to ``__call__``.

        B-factors and occupancy are fixed throughout sampling, so they are set here
        once (overriding the defaults ``atomarray_to_gemmi`` bakes from
        ``atom_array``) rather than on every ``__call__``.

        Parameters
        ----------
        atom_array
            Biotite AtomArray for the atoms the model operates on. Needs
            ``chain_id``, ``res_id``, ``res_name``, ``atom_name``, ``element``
            annotations. A missing ``altloc_id`` is defaulted to blank inside
            ``atomarray_to_gemmi``.
        b_factors
            Per-atom isotropic B-factors ``[n_atoms]`` (e.g. the pipeline's
            reconciled values: real deposited where shared with the structure,
            20.0 for model-only / NaN atoms). Set on ``sfc.atom_b_iso``.
        occupancies
            Per-atom occupancies ``[n_atoms]``. For the multi-conformer ensemble
            this is ``1/E`` (E = ensemble size), so the per-member structure
            factors summed in ``__call__`` form the multi-conformer total. Set on
            ``sfc.atom_occ``. Required: omitting it would leave occ at 1.0 and
            silently scale the amplitudes by E.
        """
        gemmi_structure = atomarray_to_gemmi(
            atom_array,
            unit_cell=self.unit_cell,
            space_group=self.space_group,
        )
        self.sfc = SFcalculator(pdbmodel=PDBParser(gemmi_structure), **self._sfc_kwargs)
        # inspect_data estimates solvent % / grid size from atom positions + vdW radii
        # (independent of occupancy / B-factor); run it before overriding occ to 1/E.
        self.sfc.inspect_data()
        self.sfc.atom_b_iso = b_factors
        self.sfc.atom_occ = occupancies

        # Reflection mask: drop outliers, and optionally the free (test) set.
        # Outlier / free_flag are numpy bool arrays from SFcalculator.
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

        ``elements``, ``b_factors`` and ``occupancies`` are accepted for protocol
        compatibility but ignored: topology and per-atom B-factors / occupancy are
        fixed in the SFcalculator built by :meth:`prepare` and do not change during
        sampling.

        Parameters
        ----------
        coordinates
            Atomic coordinates ``[batch, n_atoms, 3]`` in model atom space,
            matching the atom ordering passed to :meth:`prepare`.

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

        # Multi-conformer combination: complex sum over the ensemble [batch, n_hkl].
        # occ = 1/E (baked in prepare) makes the summed |F| the multi-conformer total.
        Fprotein_batch = self.sfc.calc_fprotein_batch(coordinates, Return=True)
        Fprotein = Fprotein_batch.sum(dim=0)

        Fcalc = torch.abs(Fprotein)
        mask = self._reflection_mask
        return self.loss(Fcalc[mask], self.sfc.Fo[mask])
