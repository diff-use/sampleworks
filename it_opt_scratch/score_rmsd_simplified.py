"""Min-altloc-RMSD for one prediction against a two-conformer reference. Self-contained.

Answers: did the ensemble actually reach BOTH conformations the crystal shows, or just one?

For each of the paper's 3-residue windows, this reports two numbers -- the closest any ensemble
member gets to altloc A, and the closest any member gets to altloc B, measured separately. The
pair is the point. RSCC can look good when an ensemble fits one conformer well and ignores the
other; only the pair reveals whether both were found.

  IN    --prediction       refined.cif from a run (multi-model = the ensemble)
        --reference         the deposited .cif, which must contain altloc A and B
        --selections-csv    the paper's per-protein windows
        --protein           which row of that CSV to use, e.g. 2YL0

  OUT   --out              one row per window:
                            min_rmsd_to_A   closest member to conformer A, Angstrom
                            min_rmsd_to_B   closest member to conformer B, Angstrom
                            n_atoms_A/_B    atoms actually compared (0 = nothing matched)

  EXIT  0 if windows scored, 1 if none did.

How to read it: both numbers small means the ensemble captured both conformers. One small and
one large means it collapsed onto a single conformation -- so max(A, B) is the honest per-window
summary, and the median of that across windows is the honest per-arm summary.

Two details that decide the numbers:

* Altlocs are read with gemmi, which exposes the altloc character directly. `load_any` drops the
  annotation and `parse` keeps only the first altloc, so neither can separate the conformers.
  Atoms with a blank altloc are shared and belong to both.
* The prediction is placed by a GLOBAL uniform-weight Kabsch onto the reference, never fitted on
  the window itself -- fitting locally would let a wrong conformation be rotated into agreement.

Run it:
  pixi run -e analysis python it_opt_scratch/score_rmsd_simplified.py \
      --prediction out/2YL0/refined-patched.cif \
      --reference  /home/dev/test_data/processed/2YL0/2YL0_single_001_density_input.cif \
      --selections-csv it_opt_scratch/paper_maxrmsd_selections.csv \
      --protein 2YL0 --out 2YL0_rmsd.csv

Use the patched CIF (scripts/patch_output_cif_files.py) when you have one. Atoms are matched by
(chain, residue number, atom name), and for some proteins the raw prediction relabels chains and
renumbers from 1 while the reference keeps its deposited numbering -- then nothing matches and
every window comes back with n_atoms 0. This script stops with that message rather than writing
a file full of blanks.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from sampleworks.utils.atom_array_utils import (
    filter_to_common_atoms,
    remove_atoms_with_any_nan_coords,
)
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch

SELECTION_RE = re.compile(r"^chain\s+(\S+)\s+and\s+resi\s+(-?\d+)\s*-\s*(-?\d+)$")


def main() -> int:
    args = parse_args()

    windows = read_windows(args.selections_csv, args.protein)
    if not windows:
        sys.exit(f"no selections for {args.protein} in {args.selections_csv}")

    # The two reference conformers per window, straight from the altloc characters.
    conformers = reference_conformers(args.reference, windows)

    prediction = remove_atoms_with_any_nan_coords(load_any(str(args.prediction)))
    alignment_target = remove_atoms_with_any_nan_coords(load_any(str(args.reference)))
    prediction = align_to_reference(alignment_target, prediction)
    predicted_atoms = prediction_lookup(prediction)

    rows = []
    for window in windows:
        rmsd_a, n_a = min_rmsd_over_ensemble(conformers[window]["A"], predicted_atoms)
        rmsd_b, n_b = min_rmsd_over_ensemble(conformers[window]["B"], predicted_atoms)
        rows.append(
            {
                "protein": args.protein,
                "selection": window,
                "min_rmsd_to_A": rmsd_a,
                "min_rmsd_to_B": rmsd_b,
                "n_atoms_A": n_a,
                "n_atoms_B": n_b,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)

    scored = table[table[["n_atoms_A", "n_atoms_B"]].gt(0).all(axis=1)]
    if scored.empty:
        logger.error(
            "no atoms matched between prediction and reference. Atoms are matched by "
            "(chain, residue number, atom name) -- the prediction is probably renumbered. "
            "Patch it first: scripts/patch_output_cif_files.py"
        )
        return 1

    worse = scored[["min_rmsd_to_A", "min_rmsd_to_B"]].max(axis=1)
    logger.info(f"wrote {args.out}: {len(scored)}/{len(table)} windows scored")
    logger.info(
        f"median max(A,B) {worse.median():.3f} A   "
        f"windows with both conformers within 1 A: {worse.le(1.0).mean():.0%}"
    )
    return 0


def read_windows(csv_path: Path, protein: str) -> list[str]:
    """The paper's windows for one protein: the semicolon-joined `selection` column of its row."""
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["protein"].strip().upper() == protein.upper():
                return [s.strip() for s in row["selection"].split(";") if s.strip()]
    return []


def parse_selection(selection: str) -> tuple[str, range]:
    """Split ``chain A and resi 12-14`` into ``("A", range(12, 15))``."""
    match = SELECTION_RE.match(selection.strip())
    if match is None:
        raise ValueError(f"unparseable selection: {selection!r}")
    chain, low, high = match.group(1), int(match.group(2)), int(match.group(3))
    return chain, range(low, high + 1)


def reference_conformers(reference_path: Path, windows: list[str]) -> dict:
    """Per window, the altloc-A and altloc-B conformers keyed by (chain, residue, atom name).

    Read with gemmi because it exposes the altloc character per atom. An atom with a blank
    altloc is shared between the two conformers, so it is placed in both.
    """
    structure = gemmi.read_structure(str(reference_path))
    atoms_by_residue: dict[tuple[str, int], list] = {}
    for chain in structure[0]:
        for residue in chain:
            atoms_by_residue.setdefault((chain.name, residue.seqid.num), []).extend(
                (a.name, a.altloc, np.array([a.pos.x, a.pos.y, a.pos.z])) for a in residue
            )

    out = {}
    for window in windows:
        chain_name, residues = parse_selection(window)
        conformers = {"A": {}, "B": {}}
        for residue_id in residues:
            for atom_name, altloc, xyz in atoms_by_residue.get((chain_name, residue_id), []):
                shared = altloc.strip() == ""
                for label in ("A", "B") if shared else (altloc.strip(),):
                    if label in conformers:
                        conformers[label][(chain_name, residue_id, atom_name)] = xyz
        out[window] = conformers
    return out


def prediction_lookup(atom_array) -> dict:
    """Map (chain, residue, atom name) -> that atom's coordinates in every ensemble member.

    Values have shape [n_models, 3]; a single-model file is given a leading axis of 1.
    """
    coords = atom_array.coord
    if coords.ndim == 2:
        coords = coords[None]
    return {
        (str(chain), int(residue), str(name)): coords[:, i]
        for i, (chain, residue, name) in enumerate(
            zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name, strict=True)
        )
    }


def min_rmsd_over_ensemble(conformer: dict, predicted_atoms: dict) -> tuple[float, int]:
    """Closest any single ensemble member gets to one reference conformer.

    The minimum is over members, not an average: the question is whether ANY member found this
    conformation, not whether the ensemble is centred on it.
    """
    shared_keys = [key for key in conformer if key in predicted_atoms]
    if not shared_keys:
        return float("nan"), 0
    reference = np.stack([conformer[key] for key in shared_keys])  # [n_atoms, 3]
    predicted = np.stack([predicted_atoms[key] for key in shared_keys], axis=1)  # [n_models, n, 3]
    per_member = np.sqrt(((predicted - reference[None]) ** 2).sum(-1).mean(-1))  # [n_models]
    return float(per_member.min()), len(shared_keys)


def align_to_reference(reference, prediction):
    """Global uniform-weight Kabsch of prediction onto reference, applied to every atom.

    Alignment is deliberately GLOBAL: fitting on the 3-residue window itself would let a wrong
    local conformation be rotated into apparent agreement.
    """
    try:
        reference_common, prediction_common = filter_to_common_atoms(reference, prediction)
    except RuntimeError:
        # Some predictions relabel every chain to 'A' and renumber residues from 1 while the
        # reference keeps its deposited chain id and numbering, so strict (chain, res, name)
        # matching finds nothing. Sequential per-chain matching realigns these otherwise
        # identical structures. Only runs when strict matching raises.
        reference_common, prediction_common = filter_to_common_atoms(
            reference, prediction, normalize_ids=True
        )

    reference_coords = torch.from_numpy(reference_common.coord).float()
    prediction_coords = torch.from_numpy(prediction_common.coord).float()
    reference_coords = match_batch(reference_coords, prediction_coords.shape[0])
    if reference_coords.ndim != 3 or reference_coords.shape[1] != prediction_coords.shape[1]:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference_coords.shape)} "
            f"vs prediction {tuple(prediction_coords.shape)}"
        )

    n_atoms = reference_coords.shape[1]
    _, transform = weighted_rigid_align_differentiable(
        true_coords=prediction_coords,
        pred_coords=reference_coords,
        weights=torch.ones(1, n_atoms),
        mask=torch.ones(1, n_atoms),
        return_transforms=True,
        allow_gradients=False,
    )
    moved = apply_forward_transform(
        torch.from_numpy(prediction.coord), transform, rotation_only=False
    )
    prediction.coord = moved.numpy()
    return prediction


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prediction", type=Path, required=True, help="refined.cif to score")
    ap.add_argument("--reference", type=Path, required=True,
                    help="deposited .cif containing altlocs A and B")
    ap.add_argument("--selections-csv", type=Path, required=True, help="the paper's windows")
    ap.add_argument("--protein", required=True, help="which protein's windows, e.g. 2YL0")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(main())
