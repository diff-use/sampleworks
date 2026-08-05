"""Score IT-opt arms with the paper's RSCC protocol (791 max-RMSD subsegments).

The published scorer is ``scripts/eval/rscc_grid_search_script.py``. This driver reuses its
exact primitives -- the differentiable density forward model, the global uniform-weight Kabsch
alignment, and ``extract_tight`` at 2.0 A -- so the numbers are directly comparable to the
occupancy-sweep results. What it does *not* reuse is the Trial/ProteinConfig directory scanner,
because the IT-opt output tree is ``{PROTEIN}_native_occ/{arm}/refined.cif`` (depth 3) rather
than the ``{PROTEIN}_{occ}/{model}/{scaler}/ens{N}_gw{W}/`` depth-4 grid-search layout, and
``native_occ`` does not parse as an occupancy key.

Selections come from the segmentation CSV (one row per protein, semicolon-joined), which encodes
the paper's rule: the contiguous 3-residue window maximising altloc A-B RMSD, or the whole
segment when it is 3 residues or shorter.

The RSCC mask is built from reference coordinates including *all* altlocs, matching
``get_reference_structure_coords``, which unions the altloc-A-only and altloc-B-only coordinate
sets for exactly this purpose.

Usage
-----
    pixi run -e analysis python it_opt_scratch/score_paper_rscc.py \
        --runs-dir it_opt_scratch/targets_out_40 \
        --inputs-dir /home/dev/test_data \
        --selections-csv it_opt_scratch/paper_maxrmsd_selections.csv \
        --arms baseline s_only s_plus_z z_only \
        --out it_opt_scratch/patch_tree/itopt_paper_rscc.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import extract_selection_coordinates
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    remove_atoms_with_any_nan_coords,
)
from sampleworks.utils.cif_utils import resolve_mixed_hetatm_atom_altlocs
from sampleworks.utils.density_utils import build_density_transformer, run_density_transformer
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap

RESOLUTION = 1.0


def read_selections(csv_path: Path) -> dict[str, list[str]]:
    """Read the per-protein segmentation CSV into {PROTEIN: [selection, ...]}."""
    out: dict[str, list[str]] = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            sels = [s.strip() for s in row["selection"].split(";") if s.strip()]
            if sels:
                out[row["protein"].strip().upper()] = sels
    return out


def align_prediction_to_reference(ref_atom_array, pred_atom_array):
    """Global uniform-weight Kabsch of prediction onto reference, applied to every predicted atom.

    Mirrors rscc_grid_search_script.py: the transform is fitted on the atoms common to both
    structures, then applied to the whole predicted array. Alignment is deliberately global --
    fitting on the 3-residue window itself would let a wrong local conformation be rotated into
    apparent agreement.
    """
    try:
        ref_common, pred_common = filter_to_common_atoms(ref_atom_array, pred_atom_array)
    except RuntimeError:
        # The prediction relabels every chain to 'A' and renumbers residues from 1, while the
        # reference keeps the deposited chain id and numbering (e.g. chain 'P', res 5-234). Strict
        # (chain,res,name) matching then finds nothing. Fall back to sequential per-chain matching,
        # which realigns these (otherwise identical) structures. This only runs when strict matching
        # raises, so the proteins that already align keep their exact matched-atom set unchanged.
        ref_common, pred_common = filter_to_common_atoms(
            ref_atom_array, pred_atom_array, normalize_ids=True
        )
    ref_t = torch.from_numpy(ref_common.coord).float()
    pred_t = torch.from_numpy(pred_common.coord).float()
    ref_t = match_batch(ref_t, pred_t.shape[0])
    if ref_t.ndim != 3 or ref_t.shape[1] != pred_t.shape[1]:
        raise ValueError(f"shape mismatch: ref {tuple(ref_t.shape)} vs pred {tuple(pred_t.shape)}")

    n_atoms = ref_t.shape[1]
    _, transform = weighted_rigid_align_differentiable(
        true_coords=pred_t,
        pred_coords=ref_t,
        weights=torch.ones(1, n_atoms),
        mask=torch.ones(1, n_atoms),
        return_transforms=True,
        allow_gradients=False,
    )
    moved = apply_forward_transform(
        torch.from_numpy(pred_atom_array.coord), transform, rotation_only=False
    )
    pred_atom_array.coord = moved.numpy()
    return pred_atom_array


def ensure_density_annotations(atom_array):
    """Add the occupancy / b_factor annotations the density forward model requires."""
    n = atom_array.coord.shape[-2]
    for name, value in (("occupancy", 1.0), ("b_factor", 20.0)):
        if name not in atom_array.get_annotation_categories():
            atom_array.set_annotation(name, np.full(n, value))
    return atom_array


def score_protein(
    protein: str,
    selections: list[str],
    runs_dir: Path,
    inputs_dir: Path,
    arms: list[str],
    device: torch.device,
    target_filename: str = "refined.cif",
    dir_template: str = "{protein}_native_occ",
    maps_dir: Path | None = None,
    map_template: str = "{protein}_uniform_1.00A.ccp4",
) -> list[dict]:
    """Compute per-(arm, selection) RSCC for one protein."""
    rows: list[dict] = []
    maps_base = maps_dir if maps_dir is not None else inputs_dir / "density_maps"
    map_path = maps_base / map_template.format(protein=protein)
    ref_path = inputs_dir / "processed" / protein / f"{protein}_single_001_density_input.cif"
    # Match generation: collapse modified-residue positions (mixed ATOM/HETATM, different resname,
    # e.g. CYS+CSO) that atomworks would otherwise duplicate into an extra residue, so the reference
    # carries the same atoms as a prediction generated from the cleaned CIF. No-op otherwise.
    ref_path = resolve_mixed_hetatm_atom_altlocs(ref_path)

    def fail(arm: str, err: str) -> None:
        for sel in selections:
            rows.append(
                {
                    "protein": protein,
                    "arm": arm,
                    "selection": sel,
                    "rscc": np.nan,
                    "error": err,
                    "base_map_path": str(map_path),
                }
            )

    try:
        base_xmap = XMap.fromfile(str(map_path), resolution=RESOLUTION).canonical_unit_cell()
        transformer, _ = build_density_transformer(base_xmap, em_mode=False, device=device)
        # Mask coords keep every altloc: load_any preserves them, whereas parse() drops all but
        # the first. The mask must cover both conformations, as the published scorer does by
        # unioning the A-only and B-only reference structures.
        ref_all_altlocs = load_any(str(ref_path))
        sel_coords = {}
        for sel in selections:
            try:
                coords = extract_selection_coordinates(ref_all_altlocs, sel)
            except Exception as e:  # noqa: BLE001 - selection syntax varies per protein
                logger.warning(f"{protein}: selection {sel!r} failed: {e}")
                continue
            if len(coords) and np.isfinite(coords).all():
                sel_coords[sel] = coords
            else:
                logger.warning(f"{protein}: selection {sel!r} empty or non-finite")
        # Alignment target. load_any, not parse: parse() reconciles against the
        # pdbx_poly_seq_scheme that the patcher inherits from the deposited RCSB entry, and with
        # atomworks 2.1.1 and no CCD mirror that turns ~93% of atoms into NaN placeholders.
        # load_any reads atom_site as written, and is the reader the patcher itself uses.
        ref_atom_array = remove_atoms_with_any_nan_coords(load_any(str(ref_path)))
    except Exception as e:  # noqa: BLE001 - per-protein setup failure should not kill the sweep
        logger.error(f"{protein}: setup failed: {e}\n{traceback.format_exc()}")
        for arm in arms:
            fail(arm, f"setup: {e}")
        return rows

    base_cache: dict[str, np.ndarray] = {}
    for arm in arms:
        cif = runs_dir / dir_template.format(protein=protein) / arm / target_filename
        if not cif.exists():
            fail(arm, f"{target_filename} missing")
            continue
        try:
            aa = remove_atoms_with_any_nan_coords(load_any(str(cif)))
            # load_any does not populate these annotations even when the CIF carries the columns;
            # the forward model requires both. The written values are occupancy 1.0 and B 20.0,
            # which is also what parse() supplied to the published scorer.
            aa = ensure_density_annotations(aa)
            aa = align_prediction_to_reference(ref_atom_array, aa)
            # One density from the whole ensemble: RSCC scores the multi-conformer model, not
            # individual members.
            computed = run_density_transformer(transformer, aa)
            computed_xmap = copy.copy(base_xmap)
            computed_xmap.array = computed.cpu().numpy()
            if computed_xmap.array.shape != base_xmap.array.shape:
                raise ValueError(
                    f"density shape {computed_xmap.array.shape} != base {base_xmap.array.shape}"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"{protein}/{arm}: {e}\n{traceback.format_exc()}")
            fail(arm, str(e))
            continue

        for sel, coords in sel_coords.items():
            row = {
                "protein": protein,
                "arm": arm,
                "selection": sel,
                "base_map_path": str(map_path),
                "error": None,
            }
            try:
                extracted_base = base_cache.get(sel)
                if extracted_base is None:
                    _, extracted_base = base_xmap.extract_tight(
                        coords, padding=DEFAULT_SELECTION_PADDING
                    )
                    if extracted_base is None or extracted_base.shape[0] == 0:
                        raise ValueError("extracted base map empty")
                    base_cache[sel] = extracted_base
                _, extracted_computed = computed_xmap.extract_tight(
                    coords, padding=DEFAULT_SELECTION_PADDING
                )
                if extracted_computed is None or extracted_computed.shape[0] == 0:
                    raise ValueError("extracted computed map empty")
                row["rscc"] = rscc(extracted_base, extracted_computed)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
                row["rscc"] = np.nan
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--selections-csv", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["baseline", "s_only", "s_plus_z", "z_only"])
    p.add_argument("--proteins", nargs="+", default=None, help="Subset; default all in the CSV.")
    p.add_argument("--dir-template", default="{protein}_native_occ",
                   help="Per-protein dir name under --runs-dir; '{protein}' is substituted.")
    p.add_argument("--target-filename", default="refined.cif",
                   help="CIF to score in each arm dir; use refined-patched.cif after patching.")
    p.add_argument("--maps-dir", type=Path, default=None,
                   help="Dir holding the density maps; default <inputs-dir>/density_maps.")
    p.add_argument("--map-template", default="{protein}_uniform_1.00A.ccp4",
                   help="Map filename template; '{protein}' is substituted. Use "
                        "'{protein}_0.5occA_0.5occB_1.00A.ccp4' for the 0.5/0.5 occupancy maps.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    selections = read_selections(args.selections_csv)
    if args.proteins:
        wanted = {x.upper() for x in args.proteins}
        selections = {k: v for k, v in selections.items() if k in wanted}
    total = sum(len(v) for v in selections.values())
    logger.info(f"{len(selections)} proteins, {total} selections, arms={args.arms}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    rows: list[dict] = []
    for i, (protein, sels) in enumerate(sorted(selections.items()), 1):
        logger.info(f"[{i}/{len(selections)}] {protein} ({len(sels)} selections)")
        rows.extend(
            score_protein(
                protein, sels, args.runs_dir, args.inputs_dir, args.arms, device,
                target_filename=args.target_filename,
                dir_template=args.dir_template,
                maps_dir=args.maps_dir,
                map_template=args.map_template,
            )
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint after each protein

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    logger.info(f"wrote {args.out}: {len(df)} rows")

    ok = df[df["rscc"].notna()]
    logger.info(f"scored {len(ok)}/{len(df)} rows")
    if not ok.empty:
        summary = ok.groupby("arm")["rscc"].agg(
            n="size",
            median="median",
            frac_ge_08=lambda s: (s >= 0.8).mean(),
            frac_ge_09=lambda s: (s >= 0.9).mean(),
        )
        logger.info(f"\n{summary.to_string()}")


if __name__ == "__main__":
    main()
