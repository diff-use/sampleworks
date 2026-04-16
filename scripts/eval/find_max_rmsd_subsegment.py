"""Find the maximum RMSD subsegment within each altloc selection.

This script consumes the output of ``scripts/eval/find_altloc_selections.py``
and, for each contiguous altloc span longer than ``--window-size`` residues,
identifies the contiguous subsegment of that size with the highest
RMSD between any pair of alternate conformations.

Only residues with identical residue names across altlocs are considered.
Windows containing compositional heterogeneity (different residue names, e.g. CYS vs CSO)
are skipped.

The primary output CSV preserves the setup expected by
``rscc_grid_search_script.py`` (one row per protein, semicolon joined
selections).  An optional diagnostic CSV provides per selection detail.

Usage: find_altloc_selections.py -> this script -> rscc_grid_search_script.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArrayStack, get_residues, rmsd as biotite_rmsd
from loguru import logger
from sampleworks.eval.grid_search_eval_utils import resolve_cif_path
from sampleworks.eval.structure_utils import (
    ATOMWORKS_COMPARISON_OPS,
    parse_selection_string,
)
from sampleworks.utils.atom_array_utils import (
    build_pairwise_altloc_arrays,
    detect_altlocs,
    load_structure_with_altlocs,
)


def _has_compositional_heterogeneity(arr_i, arr_j, mask: np.ndarray) -> bool:
    """Check if residues under *mask* have different names between two altloc arrays."""
    _, names_i = get_residues(arr_i[mask])
    _, names_j = get_residues(arr_j[mask])
    return not np.array_equal(names_i, names_j)


def _find_max_rmsd_window(
    pair_arrays: dict[tuple[str, str], tuple[AtomArrayStack, AtomArrayStack]],
    chain: str,
    residues: list[int],
    window_size: int = 3,
) -> tuple[list[int], float, str] | None:
    """Slide a window over residues and find the subsegment with maximum all atom RMSD.

    Parameters
    ----------
    pair_arrays
        Output of ``build_pairwise_altloc_arrays``.
    chain
        Chain ID to filter on.
    residues
        Sorted list of actual residue IDs in the span.
    window_size
        Number of consecutive residues per window.

    Returns
    -------
    tuple or None
        ``(best_window_residues, best_rmsd, best_pair_str)`` or ``None``
        if no valid RMSD could be computed for any window.
    """
    best_rmsd = -np.inf
    best_window: list[int] | None = None
    best_pair = ""

    for w in range(len(residues) - window_size + 1):
        window_res = residues[w : w + window_size]

        for (alt_i, alt_j), (stack_i, stack_j) in pair_arrays.items():
            arr_i = stack_i[0]
            arr_j = stack_j[0]

            mask = (arr_i.chain_id == chain) & np.isin(arr_i.res_id, window_res)
            if mask.sum() == 0:
                continue

            if _has_compositional_heterogeneity(arr_i, arr_j, mask):
                continue

            rmsd_val = float(biotite_rmsd(arr_i[mask], arr_j[mask]))
            if np.isfinite(rmsd_val) and rmsd_val > best_rmsd:
                best_rmsd = rmsd_val
                best_window = window_res
                best_pair = f"{alt_i}-{alt_j}"

    if best_window is None:
        return None
    return best_window, float(best_rmsd), best_pair


def _process_structure(
    row: pd.Series,
    cif_root: Path | None,
    window_size: int = 3,
) -> list[dict]:
    """Load a structure and narrow all its selections to max RMSD subsegments."""
    protein = str(row["protein"])
    cif_path = resolve_cif_path(row, cif_root)
    if not cif_path.exists():
        logger.error(f"[{protein}] CIF file not found: {cif_path}")
        return []

    selection_field = row.get("selection", "")
    if not isinstance(selection_field, str) or not selection_field.strip():
        logger.warning(f"[{protein}] no selections in CSV row")
        return []

    logger.info(f"[{protein}] loading {cif_path}")
    atom_array = load_structure_with_altlocs(cif_path)
    altloc_info = detect_altlocs(atom_array)
    if len(altloc_info.altloc_ids) < 2:
        logger.warning(
            f"[{protein}] structure has <2 altloc IDs ({altloc_info.altloc_ids}); skipping"
        )
        return []

    pair_arrays = build_pairwise_altloc_arrays(atom_array, altloc_info.altloc_ids)
    if not pair_arrays:
        logger.warning(f"[{protein}] no valid altloc pairs could be built; skipping")
        return []

    output_rows: list[dict] = []
    for sel_str in [s.strip() for s in selection_field.split(";") if s.strip()]:
        # Skip atomworks-style catchall selections # NOTE: This will need to be addressed when we
        # migrate to atomworks-style selections for everything, similar to classify_altloc_regions
        if any(op in sel_str for op in ATOMWORKS_COMPARISON_OPS):
            continue

        chain, start, end = parse_selection_string(sel_str)
        if chain is None or start is None or end is None:
            logger.warning(f"[{protein}] cannot parse selection: {sel_str}")
            continue

        # Get resids present in the structure for this range
        range_mask = (
            (atom_array.chain_id == chain)
            & (atom_array.res_id >= start)
            & (atom_array.res_id <= end)
        )
        actual_res_ids = sorted(set(int(r) for r in atom_array.res_id[range_mask]))

        if not actual_res_ids:
            logger.warning(f"[{protein}] selection matched no residues: {sel_str}")
            continue

        out: dict = {
            "protein": protein,
            "structure_pattern": row.get("structure_pattern", ""),
            "map_pattern": row.get("map_pattern", ""),
            "base_map_dir": row.get("base_map_dir", ""),
            "resolution": row.get("resolution", ""),
            "original_selection": sel_str,
        }

        if len(actual_res_ids) <= window_size:
            out["selection"] = sel_str
            out["max_rmsd"] = float("nan")
            out["altloc_pair"] = ""
        else:
            result = _find_max_rmsd_window(pair_arrays, chain, actual_res_ids, window_size)
            if result is None:
                logger.warning(f"[{protein}] no valid RMSD window for {sel_str}; keeping original")
                out["selection"] = sel_str
                out["max_rmsd"] = float("nan")
                out["altloc_pair"] = ""
            else:
                best_res, best_rmsd, best_pair = result
                out["selection"] = f"chain {chain} and resi {best_res[0]}-{best_res[-1]}"
                out["max_rmsd"] = best_rmsd
                out["altloc_pair"] = best_pair

        output_rows.append(out)

    return output_rows


def main(args: argparse.Namespace) -> None:
    input_df = pd.read_csv(args.input_csv)
    required = {"protein", "selection"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    all_rows: list[dict] = []
    for _, row in input_df.iterrows():
        all_rows.extend(
            _process_structure(row=row, cif_root=args.cif_root, window_size=args.window_size)
        )

    detail_df = pd.DataFrame(all_rows)

    # Write diagnostic csv
    if args.diagnostic_file:
        args.diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(args.diagnostic_file, index=False)
        logger.info(f"Wrote {len(detail_df)} rows to {args.diagnostic_file}")

    if detail_df.empty:
        final_df = pd.DataFrame(
            columns=pd.Index(
                [
                    "protein",
                    "selection",
                    "structure_pattern",
                    "map_pattern",
                    "base_map_dir",
                    "resolution",
                ]
            )
        )
    else:
        final_rows = []
        for protein, group in detail_df.groupby("protein", sort=False):
            final_rows.append(
                {
                    "protein": protein,
                    "selection": ";".join(group["selection"]),
                    "structure_pattern": group["structure_pattern"].iloc[0],
                    "map_pattern": group["map_pattern"].iloc[0],
                    "base_map_dir": group["base_map_dir"].iloc[0],
                    "resolution": group["resolution"].iloc[0],
                }
            )
        final_df = pd.DataFrame(final_rows)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_file, index=False)
    logger.info(f"Wrote {len(final_df)} proteins to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "For each altloc selection spanning more than --window-size residues, "
            "find the contiguous subsegment with maximum pairwise all atom RMSD "
            "between altloc conformations. Narrows selections for downstream RSCC evaluation."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Output CSV from find_altloc_selections.py (columns: protein, selection, "
        "structure_pattern, map_pattern, base_map_dir, resolution).",
    )
    parser.add_argument(
        "--cif-root",
        type=Path,
        default=None,
        help="Root directory to resolve structure_pattern entries against.",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-file",
        type=Path,
        default=None,
        help="Optional per-selection diagnostic CSV with RMSD details.",
    )
    parser.add_argument("--window-size", type=int, default=3)
    args = parser.parse_args()
    main(args)
