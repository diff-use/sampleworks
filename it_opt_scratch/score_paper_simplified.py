"""Score one run tree with both paper metrics, in one command and one CSV.

This is a thin driver, not a new metric. RSCC comes from score_paper_rscc.py and min-altloc-RMSD
from score_paper_rmsd.py, both called unchanged -- so the numbers are identical to running those
two scripts separately. It exists so you remember one command instead of two with eight matching
flags, and get one table instead of two you have to join by hand.

  IN    --runs-dir         tree of <dir-template>/<arm>/<target-filename>
        --dir-template     per-protein dir name, e.g. '{protein}_0.5occA_0.5occB'
        --arms             which arm sub-dirs to score
        --inputs-dir       holds processed/{PROTEIN}/{PROTEIN}_single_001_density_input.cif
        --maps-dir         holds the density maps (default <inputs-dir>/density_maps)
        --map-template     map filename, e.g. '{protein}_0.5occA_0.5occB_1.00A.ccp4'
        --selections-csv   the paper's per-protein 3-residue max-RMSD windows

  OUT   --out             one row per (protein, arm, selection):
                          rscc, min_rmsd_to_A, min_rmsd_to_B, n_atoms_A, n_atoms_B,
                          rscc_error, rmsd_error, base_map_path
                          rewritten after every protein, so a long sweep is resumable-by-eye

Run it on the 11-protein regen tree:
  pixi run -e analysis python it_opt_scratch/score_paper_simplified.py \
      --runs-dir it_opt_scratch/targets_out_11_regen_0.5occA_0.5occB_ens8 \
      --dir-template '{protein}_0.5occA_0.5occB' \
      --maps-dir it_opt_scratch/targets_out_11_regen_0.5occA_0.5occB/density_maps \
      --map-template '{protein}_0.5occA_0.5occB_1.00A.ccp4' \
      --inputs-dir /home/dev/test_data \
      --selections-csv it_opt_scratch/paper_maxrmsd_selections.csv \
      --arms s_plus_z \
      --out it_opt_scratch/regen11_scores.csv

Each prediction is loaded and aligned once per metric rather than once in total. That is a
deliberate trade: calling the two scorers as they are keeps this file honest about the numbers,
and scoring is seconds per arm against minutes per generation run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from loguru import logger

# Same directory as this script, which is sys.path[0] when run as `python it_opt_scratch/...`.
import score_paper_rmsd
import score_paper_rscc

KEY = ["protein", "arm", "selection"]


def main() -> None:
    args = parse_args()

    selections = score_paper_rscc.read_selections(args.selections_csv)
    if args.proteins:
        wanted = {p.upper() for p in args.proteins}
        selections = {k: v for k, v in selections.items() if k in wanted}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"{len(selections)} proteins, {sum(len(v) for v in selections.values())} selections, "
        f"arms={args.arms}, device={device}"
    )

    scored = []
    for i, (protein, sels) in enumerate(sorted(selections.items()), 1):
        logger.info(f"[{i}/{len(selections)}] {protein} ({len(sels)} selections)")
        scored.append(score_one_protein(protein, sels, args, device))
        pd.concat(scored).to_csv(args.out, index=False)  # checkpoint after every protein

    table = pd.concat(scored)
    table.to_csv(args.out, index=False)
    logger.info(f"wrote {args.out}: {len(table)} rows")
    report(table)


def score_one_protein(protein: str, sels: list[str], args, device) -> pd.DataFrame:
    """Both metrics for one protein, joined on (protein, arm, selection).

    An outer join because the two scorers can disagree on which selections are scoreable: the
    RSCC side drops a selection whose reference coordinates are empty or non-finite, while the
    RMSD side still emits a row for it.
    """
    rscc_rows = score_paper_rscc.score_protein(
        protein, sels, args.runs_dir, args.inputs_dir, args.arms, device,
        target_filename=args.target_filename,
        dir_template=args.dir_template,
        maps_dir=args.maps_dir,
        map_template=args.map_template,
    )
    rmsd_rows = score_paper_rmsd.score_protein(
        protein, sels, args.runs_dir, args.inputs_dir, args.arms,
        args.target_filename, args.dir_template,
    )
    rscc_table = pd.DataFrame(rscc_rows).rename(columns={"error": "rscc_error"})
    rmsd_table = pd.DataFrame(rmsd_rows).rename(columns={"error": "rmsd_error"})
    return pd.merge(rscc_table, rmsd_table, on=KEY, how="outer")


def report(table: pd.DataFrame) -> None:
    """Per-arm summary, in the same terms the two original scorers print."""
    rscc_ok = table[table["rscc"].notna()]
    if not rscc_ok.empty:
        summary = rscc_ok.groupby("arm")["rscc"].agg(
            n="size",
            median="median",
            frac_ge_08=lambda s: (s >= 0.8).mean(),
            frac_ge_09=lambda s: (s >= 0.9).mean(),
        )
        logger.info(f"RSCC\n{summary.to_string()}")

    both = ["min_rmsd_to_A", "min_rmsd_to_B"]
    rmsd_ok = table[table[both].notna().all(axis=1)]
    if not rmsd_ok.empty:
        nearer = rmsd_ok[both].min(axis=1)  # the altloc the ensemble reached
        worse = rmsd_ok[both].max(axis=1)  # the one it had to also reach to score well
        summary = pd.DataFrame(
            {
                "n": rmsd_ok.groupby("arm").size(),
                "med_nearer": nearer.groupby(rmsd_ok.arm).median(),
                "med_max": worse.groupby(rmsd_ok.arm).median(),
                "max_le_2A": worse.le(2.0).groupby(rmsd_ok.arm).mean(),
                "max_le_1A": worse.le(1.0).groupby(rmsd_ok.arm).mean(),
            }
        )
        logger.info(f"min-altloc-RMSD\n{summary.to_string()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--inputs-dir", type=Path, required=True)
    ap.add_argument("--selections-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["baseline", "s_only", "s_plus_z", "z_only"])
    ap.add_argument("--proteins", nargs="+", default=None, help="subset; default all in the CSV")
    ap.add_argument("--dir-template", default="{protein}_native_occ",
                    help="per-protein dir under --runs-dir; '{protein}' is substituted")
    ap.add_argument("--target-filename", default="refined.cif",
                    help="CIF to score in each arm dir; refined-patched.cif after patching")
    ap.add_argument("--maps-dir", type=Path, default=None,
                    help="dir holding the maps; default <inputs-dir>/density_maps")
    ap.add_argument("--map-template", default="{protein}_uniform_1.00A.ccp4",
                    help="map filename; use '{protein}_0.5occA_0.5occB_1.00A.ccp4' for the "
                         "paper's 0.5/0.5 occupancy maps")
    return ap.parse_args()


if __name__ == "__main__":
    main()
