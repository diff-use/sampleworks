"""Score IT-opt arms with the paper's min-altloc-RMSD metric (Figure 3E).

The paper computes, per altloc selection, the *minimum* RMSD over the predicted ensemble to
altloc A and to altloc B of the deposited reference, separately. Its purpose is diagnostic: an
ensemble can raise RSCC by fitting one altloc well, and only the pair (min-to-A, min-to-B)
reveals whether both conformations were reached.

Scope matches the paper's RSCC protocol -- the same 3-residue max-RMSD subsegments -- and the
prediction is placed by the same global uniform-weight Kabsch used in ``score_paper_rscc.py``.
That is deliberate: aligning on the subsegment itself would let a wrong local conformation be
rotated into agreement.

Per-altloc conformers are read with gemmi, which exposes the altloc character directly.
``load_any`` drops the altloc annotation entirely and ``parse`` keeps only the first altloc,
so neither can separate the two conformers. Atoms with a blank altloc are shared and belong to
both conformers.

Comparator: ``occ_sweep_results/min_altloc_rmsd_results.csv`` (protenix, 0.5/0.5), already
computed by the published pipeline over the same selections.

Usage
-----
    python it_opt_scratch/score_paper_rmsd.py \
        --runs-dir it_opt_scratch/patch_tree --dir-template "{protein}" \
        --target-filename refined-patched.cif \
        --inputs-dir /home/dev/test_data \
        --selections-csv it_opt_scratch/paper_maxrmsd_selections.csv \
        --arms baseline coord_guidance s_only s_plus_z z_only \
        --out it_opt_scratch/patch_tree/itopt_paper_rmsd.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import traceback
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from sampleworks.utils.atom_array_utils import remove_atoms_with_any_nan_coords

# Same directory as this script, which is sys.path[0] when run as `python it_opt_scratch/...`.
from score_paper_rscc import align_prediction_to_reference, read_selections

SELECTION_RE = re.compile(r"^chain\s+(\S+)\s+and\s+resi\s+(-?\d+)\s*-\s*(-?\d+)$")


def parse_selection(selection: str) -> tuple[str, range]:
    """Split ``chain A and resi 12-14`` into ``("A", range(12, 15))``."""
    m = SELECTION_RE.match(selection.strip())
    if m is None:
        raise ValueError(f"unparseable selection: {selection!r}")
    chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
    return chain, range(lo, hi + 1)


def reference_conformers(
    ref_path: Path, selections: list[str]
) -> dict[str, dict[str, dict[tuple[str, int, str], np.ndarray]]]:
    """Per selection, the altloc-A and altloc-B conformers keyed by (chain, res_id, atom_name).

    Atoms with a blank altloc are shared between conformers, so they appear in both.
    """
    st = gemmi.read_structure(str(ref_path))
    by_res: dict[tuple[str, int], list] = {}
    for chain in st[0]:
        for res in chain:
            by_res.setdefault((chain.name, res.seqid.num), []).extend(
                (a.name, a.altloc, np.array([a.pos.x, a.pos.y, a.pos.z])) for a in res
            )

    out: dict[str, dict[str, dict[tuple[str, int, str], np.ndarray]]] = {}
    for sel in selections:
        chain, residues = parse_selection(sel)
        conformers: dict[str, dict[tuple[str, int, str], np.ndarray]] = {"A": {}, "B": {}}
        for res_id in residues:
            for name, altloc, xyz in by_res.get((chain, res_id), []):
                alt = altloc.strip()
                targets = ("A", "B") if alt == "" else (alt,)
                for t in targets:
                    if t in conformers:
                        conformers[t][(chain, res_id, name)] = xyz
        out[sel] = conformers
    return out


def prediction_lookup(atom_array) -> dict[tuple[str, int, str], np.ndarray]:
    """Map (chain, res_id, atom_name) -> per-model coordinates, shape [n_models, 3]."""
    coords = atom_array.coord
    if coords.ndim == 2:  # single model -> add the model axis
        coords = coords[None]
    return {
        (str(c), int(r), str(n)): coords[:, i]
        for i, (c, r, n) in enumerate(
            zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name, strict=True)
        )
    }


def min_rmsd_over_ensemble(
    conformer: dict[tuple[str, int, str], np.ndarray],
    pred: dict[tuple[str, int, str], np.ndarray],
) -> tuple[float, int]:
    """Minimum over ensemble members of RMSD to one reference conformer, plus atoms matched."""
    keys = [k for k in conformer if k in pred]
    if not keys:
        return float("nan"), 0
    ref = np.stack([conformer[k] for k in keys])  # [n_atoms, 3]
    prd = np.stack([pred[k] for k in keys], axis=1)  # [n_models, n_atoms, 3]
    per_model = np.sqrt(((prd - ref[None]) ** 2).sum(-1).mean(-1))  # [n_models]
    return float(per_model.min()), len(keys)


def score_protein(
    protein: str,
    selections: list[str],
    runs_dir: Path,
    inputs_dir: Path,
    arms: list[str],
    target_filename: str,
    dir_template: str,
) -> list[dict]:
    rows: list[dict] = []
    ref_path = inputs_dir / "processed" / protein / f"{protein}_single_001_density_input.cif"

    def fail(arm: str, err: str) -> None:
        for sel in selections:
            rows.append(
                {
                    "protein": protein,
                    "arm": arm,
                    "selection": sel,
                    "min_rmsd_to_A": np.nan,
                    "min_rmsd_to_B": np.nan,
                    "n_atoms_A": 0,
                    "n_atoms_B": 0,
                    "error": err,
                }
            )

    try:
        conformers = reference_conformers(ref_path, selections)
        # Alignment target: the same array score_paper_rscc.py aligns against, so both metrics
        # place the prediction identically.
        ref_atom_array = remove_atoms_with_any_nan_coords(load_any(str(ref_path)))
    except Exception as e:  # noqa: BLE001
        logger.error(f"{protein}: setup failed: {e}\n{traceback.format_exc()}")
        for arm in arms:
            fail(arm, f"setup: {e}")
        return rows

    for arm in arms:
        cif = runs_dir / dir_template.format(protein=protein) / arm / target_filename
        if not cif.exists():
            fail(arm, f"{target_filename} missing")
            continue
        try:
            aa = remove_atoms_with_any_nan_coords(load_any(str(cif)))
            aa = align_prediction_to_reference(ref_atom_array, aa)
            pred = prediction_lookup(aa)
        except Exception as e:  # noqa: BLE001
            logger.error(f"{protein}/{arm}: {e}\n{traceback.format_exc()}")
            fail(arm, str(e))
            continue

        for sel in selections:
            rmsd_a, n_a = min_rmsd_over_ensemble(conformers[sel]["A"], pred)
            rmsd_b, n_b = min_rmsd_over_ensemble(conformers[sel]["B"], pred)
            rows.append(
                {
                    "protein": protein,
                    "arm": arm,
                    "selection": sel,
                    "min_rmsd_to_A": rmsd_a,
                    "min_rmsd_to_B": rmsd_b,
                    "n_atoms_A": n_a,
                    "n_atoms_B": n_b,
                    "error": None if n_a and n_b else "no matching atoms",
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--selections-csv", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["baseline", "s_only", "s_plus_z", "z_only"])
    p.add_argument("--proteins", nargs="+", default=None)
    p.add_argument("--dir-template", default="{protein}_native_occ")
    p.add_argument("--target-filename", default="refined.cif")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    selections = read_selections(args.selections_csv)
    if args.proteins:
        wanted = {x.upper() for x in args.proteins}
        selections = {k: v for k, v in selections.items() if k in wanted}
    logger.info(
        f"{len(selections)} proteins, {sum(len(v) for v in selections.values())} selections, "
        f"arms={args.arms}"
    )

    rows: list[dict] = []
    for i, (protein, sels) in enumerate(sorted(selections.items()), 1):
        logger.info(f"[{i}/{len(selections)}] {protein} ({len(sels)} selections)")
        rows.extend(
            score_protein(
                protein, sels, args.runs_dir, args.inputs_dir, args.arms,
                args.target_filename, args.dir_template,
            )
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    ok = df[df.min_rmsd_to_A.notna() & df.min_rmsd_to_B.notna()]
    logger.info(f"wrote {args.out}: {len(df)} rows, {len(ok)} scored")
    if not ok.empty:
        nearer = ok[["min_rmsd_to_A", "min_rmsd_to_B"]].min(axis=1)
        worse = ok[["min_rmsd_to_A", "min_rmsd_to_B"]].max(axis=1)
        summary = pd.DataFrame(
            {
                "n": ok.groupby("arm").size(),
                "med_nearer": nearer.groupby(ok.arm).median(),
                "med_max": worse.groupby(ok.arm).median(),
                "max_le_2A": worse.le(2.0).groupby(ok.arm).mean(),
                "max_le_1A": worse.le(1.0).groupby(ok.arm).mean(),
            }
        )
        logger.info(f"\n{summary.to_string()}")


if __name__ == "__main__":
    main()
