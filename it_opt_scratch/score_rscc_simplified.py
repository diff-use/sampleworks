"""RSCC for one prediction against one density map. Self-contained: just run it.

Scores how well a predicted ensemble explains observed density, over the paper's 3-residue
windows. One prediction per invocation, so a sweep is a plain loop in the caller.

  IN    --prediction       refined.cif from a run (an ensemble is expected; see below)
        --reference         the deposited .cif -- the alignment target, and the source of the
                            coordinates that define each window
        --map               the observed .ccp4
        --resolution        that map's resolution, in Angstrom
        --selections-csv    the paper's per-protein windows (semicolon-joined column)
        --protein           which row of that CSV to use, e.g. 2YL0

  OUT   --out              one row per window: protein, selection, rscc, error

  EXIT  0 if every window scored, 1 if any errored.

The whole ensemble becomes a SINGLE calculated density, compared to the map once. That is the
point of the metric: RSCC scores the multi-conformer model, not its members individually, which
is how an ensemble can explain density that no single conformer can.

Run it:
  pixi run -e analysis python it_opt_scratch/score_rscc_simplified.py \
      --prediction  out/2YL0/refined.cif \
      --reference   /home/dev/test_data/processed/2YL0/2YL0_single_001_density_input.cif \
      --map         density_maps/2YL0_0.5occA_0.5occB_1.00A.ccp4 \
      --resolution  1.0 \
      --selections-csv it_opt_scratch/paper_maxrmsd_selections.csv \
      --protein 2YL0 --out 2YL0_rscc.csv

Every number-determining step is written out in this file on purpose -- the alignment, the
occupancy/B values, the window cropping. They match scripts/eval/rscc_grid_search_script.py and
score_paper_rscc.py; changing any of them changes the metric, so the comments say why each is
the way it is.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
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
from sampleworks.utils.density_utils import build_density_transformer, run_density_transformer
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.framework_utils import match_batch

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    windows = read_windows(args.selections_csv, args.protein)
    if not windows:
        sys.exit(f"no selections for {args.protein} in {args.selections_csv}")

    require_unit_cell(args.prediction)

    # The observed map, plus the forward model that turns coordinates into a calculated one.
    observed = XMap.fromfile(str(args.map), resolution=args.resolution).canonical_unit_cell()
    transformer, _ = build_density_transformer(observed, em_mode=False, device=device)

    # Two reads of the reference, for two different jobs:
    #   - windows are defined over ALL altlocs, so the mask covers both conformations;
    #   - alignment needs NaN-free coordinates.
    # load_any, not parse: parse() reconciles against pdbx_poly_seq_scheme and, without a CCD
    # mirror, turns most atoms into NaN placeholders. load_any reads atom_site as written.
    reference_all_altlocs = load_any(str(args.reference))
    alignment_target = remove_atoms_with_any_nan_coords(load_any(str(args.reference)))

    window_coords = {}
    for selection in windows:
        coords = extract_selection_coordinates(reference_all_altlocs, selection)
        if len(coords) and np.isfinite(coords).all():
            window_coords[selection] = coords
        else:
            logger.warning(f"{args.protein}: window {selection!r} empty or non-finite, skipping")

    prediction = remove_atoms_with_any_nan_coords(load_any(str(args.prediction)))
    prediction = add_density_annotations(prediction)
    prediction = align_to_reference(alignment_target, prediction)

    calculated = copy.copy(observed)
    calculated.array = run_density_transformer(transformer, prediction).cpu().numpy()
    if calculated.array.shape != observed.array.shape:
        sys.exit(f"density shape {calculated.array.shape} != observed {observed.array.shape}")

    rows = [
        score_window(args.protein, selection, coords, observed, calculated)
        for selection, coords in window_coords.items()
    ]

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    scored = int(table["rscc"].notna().sum())
    logger.info(f"wrote {args.out}: {scored}/{len(table)} windows scored")
    if scored:
        logger.info(f"median rscc {table['rscc'].median():.4f}")
    return 0 if scored == len(table) else 1


def require_unit_cell(cif_path: Path) -> None:
    """Stop early if the CIF has no crystallographic header.

    X-ray density is calculated in the full crystal frame, so the forward model needs the unit
    cell and space group. A freshly generated refined.cif carries neither -- the header is added
    afterwards by scripts/patch_output_cif_files.py, which fetches it from the PDB entry. Without
    this check, scoring an unpatched file yields numbers that look reasonable and are not.
    """
    text = cif_path.read_text()
    if "_cell." not in text:
        sys.exit(
            f"{cif_path} has no unit cell -- it looks like an unpatched refined.cif.\n"
            "Add the crystallographic header first:\n"
            "  python scripts/patch_output_cif_files.py --input-dir <dir> "
            "--grid-search-input-dir <processed structures> --rcsb-pattern '<dir>/([0-9][A-Za-z0-9]{3})'\n"
            "then score the resulting refined-patched.cif."
        )


def read_windows(csv_path: Path, protein: str) -> list[str]:
    """The paper's windows for one protein: the semicolon-joined `selection` column of its row."""
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["protein"].strip().upper() == protein.upper():
                return [s.strip() for s in row["selection"].split(";") if s.strip()]
    return []


def align_to_reference(reference, prediction):
    """Global uniform-weight Kabsch of prediction onto reference, applied to every atom.

    Alignment is deliberately GLOBAL: fitting on the 3-residue window itself would let a wrong
    local conformation be rotated into apparent agreement. The transform is fitted on the atoms
    common to both structures, then applied to the whole predicted array.
    """
    try:
        reference_common, prediction_common = filter_to_common_atoms(reference, prediction)
    except RuntimeError:
        # The prediction relabels every chain to 'A' and renumbers residues from 1, while the
        # reference keeps the deposited chain id and numbering (e.g. chain 'P', res 5-234).
        # Strict (chain, res, name) matching then finds nothing, so fall back to sequential
        # per-chain matching. This only runs when strict matching raises, so proteins that
        # already align keep their exact matched-atom set.
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


def add_density_annotations(atom_array):
    """Add the occupancy / b_factor annotations the density forward model requires.

    load_any does not populate these even when the CIF carries the columns. The values are
    occupancy 1.0 and B 20.0, which is what parse() supplied to the published scorer.
    """
    n_atoms = atom_array.coord.shape[-2]
    for name, value in (("occupancy", 1.0), ("b_factor", 20.0)):
        if name not in atom_array.get_annotation_categories():
            atom_array.set_annotation(name, np.full(n_atoms, value))
    return atom_array


def score_window(protein: str, selection: str, coords, observed, calculated) -> dict:
    """RSCC over one window: crop both maps to it, then correlate.

    Cropping is `extract_tight` around the window's reference coordinates, so observed and
    calculated are compared on exactly the same voxels.
    """
    row = {"protein": protein, "selection": selection, "rscc": None, "error": None}
    try:
        _, observed_crop = observed.extract_tight(coords, padding=DEFAULT_SELECTION_PADDING)
        _, calculated_crop = calculated.extract_tight(coords, padding=DEFAULT_SELECTION_PADDING)
        if observed_crop is None or observed_crop.shape[0] == 0:
            raise ValueError("observed crop empty")
        if calculated_crop is None or calculated_crop.shape[0] == 0:
            raise ValueError("calculated crop empty")
        row["rscc"] = rscc(observed_crop, calculated_crop)
    except Exception as err:  # noqa: BLE001 - one bad window should not lose the others
        row["error"] = str(err)
    return row


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prediction", type=Path, required=True, help="refined.cif to score")
    ap.add_argument("--reference", type=Path, required=True, help="deposited .cif")
    ap.add_argument("--map", type=Path, required=True, help="observed .ccp4")
    ap.add_argument("--resolution", type=float, default=1.0, help="map resolution, Angstrom")
    ap.add_argument("--selections-csv", type=Path, required=True, help="the paper's windows")
    ap.add_argument("--protein", required=True, help="which protein's windows, e.g. 2YL0")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(main())
