"""Classify altloc regions.

This script consumes the output of ``scripts/eval/find_altloc_selections.py``
and classifies each contiguous altloc span into one of four bins:

1. ``side_chain_only`` : altloc atoms exist in the residue, but none of its
   backbone atoms have altlocs.
2. ``small_loop`` : a contiguous backbone altloc span whose mean per-residue
   backbone lDDT score (defined below) between altlocs is above
   ``--loop-lddt-threshold`` (default 0.75).
3. ``large_loop`` : a contiguous backbone-altloc span whose mean per-residue
   backbone lDDT score between altlocs is below ``--loop-lddt-threshold``.
4. ``domain_shift`` : a single contiguous backbone-altloc span longer than
   ``--domain-shift-min-span`` residues (default 50). Classified before the
   loop lDDT test.

Score definition (important, slightly different from canonical lDDT):

    For a given pair of altlocs, the score is the **equal-weighted arithmetic
    mean** of per residue backbone lDDT scores across the span:

        score = (1 / N_span_residues) * sum_k score_k

    Each ``score_k`` is the standard per-residue local lDDT from
    :class:`sampleworks.metrics.lddt.AllAtomLDDT`, which is the fraction of residue
    k's neighbor distances (within 15 Å) that are preserved between altlocs across
    the four lDDT thresholds (0.5, 1, 2, 4 Å).

    The canonical atom pair weighted lDDT would instead aggregate as
    ``sum_k(score_k * n_pairs_k) / sum_k(n_pairs_k)``. This script's
    equal residue mean is equivalent with that only when every span residue
    has the same neighbor count. The 0.75 default is calibrated for this specific
    calculation.

Altloc pairing: when > 2 altlocs are present, the score above is
computed for every combination of altloc pairs and the span is
classified by the *minimum* score over pair combinations.

Use ``find_altloc_selections.py --min-span 1`` to ensure single-residue side only
selections.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from loguru import logger
from sampleworks.eval.grid_search_eval_utils import resolve_cif_path
from sampleworks.eval.structure_utils import (
    ATOMWORKS_COMPARISON_OPS,
    get_mask_from_old_selection_string,
    parse_selection_string,
)
from sampleworks.metrics.lddt import AllAtomLDDT
from sampleworks.utils.atom_array_utils import (
    BACKBONE_ATOM_TYPES,
    BLANK_ALTLOC_IDS,
    detect_altlocs,
    filter_to_common_atoms,
    load_structure_with_altlocs,
    select_altloc,
)


_ATOMWORKS_CHAIN_RE = re.compile(r"chain_id\s*==\s*['\"]([^'\"]+)['\"]")


OUTPUT_COLUMNS = [
    "protein",
    "selection",
    "chain",
    "start_res",
    "end_res",
    "span_length",
    "classification",
    "worst_pair_mean_backbone_lddt",
    "n_backbone_altloc_residues",
    "n_altlocs",
    "pair_lddts",
]


def _max_contiguous_run(sorted_res_ids: np.ndarray | list[int]) -> int:
    """Return the length of the longest contiguous run of integers in a sorted list."""
    arr = np.asarray(sorted_res_ids, dtype=int)
    if arr.size == 0:
        return 0
    breaks = np.concatenate(([0], np.nonzero(np.diff(arr) != 1)[0] + 1, [arr.size]))
    return int(np.diff(breaks).max())


def _chain_from_selection(selection: str) -> str | None:
    """Extract the chain_id named by a selection string, or None if absent.

    Handles atomworks-style (``chain_id == 'A'``) and the legacy ``chain A``
    syntax accepted by ``parse_selection_string``.

    TODO: deprecate when we move all to atomworks style selections.
    """
    m = _ATOMWORKS_CHAIN_RE.search(selection)
    if m is not None:
        return m.group(1)
    if any(op in selection for op in ATOMWORKS_COMPARISON_OPS):
        # Atomworks style selection without a chain_id
        return None
    chain_id, _, _ = parse_selection_string(selection)
    return chain_id


def _build_pairwise_altloc_arrays(
    atom_array, altloc_ids: list[str]
) -> dict[tuple[str, str], tuple[AtomArray, AtomArray]]:
    """Return ``{(id_i, id_j): (array_i, array_j)}`` pre-filtered to common atoms.

    For each unordered altloc pair we build the two per-altloc AtomArrays
    (via ``select_altloc(return_full_array=True)``, which includes blank-altloc
    atoms as shared context) and then run ``filter_to_common_atoms`` so the two
    inputs have identical atom order and count.

    We build per-pair rather than using ``map_altlocs_to_stack`` so residues whose
    altloc set is a subset of those in the whole structure (e.g. 2YL0 res 60–64
    carry only altlocs A and B, not C) still get scored for the pairs where they
    exist. A stack level ``filter_to_common_atoms`` would drop them entirely.

    TODO: this helper hits the broader issue in how we
    handle structures with >2 altlocs.
    Fixing that upstream would let us replace this helper
    with a direct ``map_altlocs_to_stack`` call and remove a source of
    duplication.
    """
    pairs: dict[tuple[str, str], tuple[AtomArray, AtomArray]] = {}
    for i in range(len(altloc_ids)):
        for j in range(i + 1, len(altloc_ids)):
            a_i = select_altloc(atom_array, altloc_ids[i], return_full_array=True)
            a_j = select_altloc(atom_array, altloc_ids[j], return_full_array=True)
            try:
                f_i, f_j = filter_to_common_atoms(a_i, a_j)
            except RuntimeError as e:
                logger.warning(
                    f"could not match atoms between altlocs "
                    f"{altloc_ids[i]} and {altloc_ids[j]}: {e}"
                )
                continue
            pairs[(altloc_ids[i], altloc_ids[j])] = (f_i, f_j)
    return pairs


def _mean_residue_lddt_for_pair(
    gt_array: AtomArray,
    pred_array: AtomArray,
    chain: str,
    residues: list[int],
) -> float:
    """Equal weighted arithmetic mean of per residue lDDT across the span."""
    if gt_array is None or pred_array is None or not residues:
        return float("nan")

    res_clause = " or ".join(f"res_id == {r}" for r in residues)
    selection = f"chain_id == '{chain}' and ({res_clause}) and atom_name in ['C','CA','N','O']"
    try:
        result = AllAtomLDDT().compute(
            predicted_atom_array_stack=pred_array,
            ground_truth_atom_array_stack=gt_array,
            selection=selection,
        )
    except Exception as e:
        logger.warning(f"lDDT compute failed for chain {chain} residues {residues}: {e}")
        return float("nan")

    residue_scores = result.get("residue_lddt_scores", {})
    keys = [f"{chain}{r}" for r in residues]
    missing = [k for k in keys if k not in residue_scores]
    if missing:
        logger.warning(
            f"lDDT result missing residues {missing} for chain {chain}. This means the result"
            f"averaged only over the {len(keys) - len(missing)} residues it returned"
        )
    flat = [residue_scores[k][0] for k in keys if k in residue_scores]
    if not flat:
        return float("nan")
    return float(np.mean(flat))


def _classify_selection(
    atom_array: AtomArray,
    pair_arrays: dict[tuple[str, str], tuple[AtomArray, AtomArray]],
    altloc_ids: list[str],
    selection_str: str,
    protein: str,
    structure_altloc_mask: np.ndarray,
    structure_backbone_mask: np.ndarray,
    domain_shift_min_span: int,
    loop_lddt_threshold: float,
) -> tuple[dict, set[tuple[str, int]]] | None:
    """Classify one contiguous altloc selection into a conformational type.

    1. If the span has no backbone altlocs anywhere, it is classified as ``side_chain_only``.
    2. Else if the longest contiguous backbone altloc run exceeds
       ``domain_shift_min_span``, it is classified as ``domain_shift``.
    3. Else compute the per residue backbone lDDT for every altloc pair over
       the backbone altloc residues in the span and take the minimum
       pair mean. Compare against ``loop_lddt_threshold``, if it is above is is classified as
       ``small_loop``. If it is below, it is classified as ``large_loop``.

    Returns ``(row_dict, covered_altloc_residues)`` on success or ``None`` if the
    selection could not be applied.

    ``row_dict`` has the keys:
    ``protein``, ``selection``, ``chain``, ``start_res``, ``end_res``,
    ``span_length``, ``classification``, ``worst_pair_mean_backbone_lddt``,
    ``n_backbone_altloc_residues``, ``n_altlocs``, and ``pair_lddts`` (a
    JSON encoded ``{pair_label: mean_lddt}`` map so the dict can be loaded
    through the CSV intact via ``json.loads``).

    ``covered_altloc_residues`` is the set of ``(chain_id, res_id)`` pairs in the
    span that carry any altloc, used for the caller's residue-coverage invariant
    check.
    """
    try:
        if not any(op in selection_str for op in ATOMWORKS_COMPARISON_OPS):
            sel_mask = get_mask_from_old_selection_string(atom_array, selection_str)
        else:
            sel_mask = atom_array.mask(selection_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"[{protein}] failed to apply selection '{selection_str}': {e}")
        return None

    if not sel_mask.any():
        logger.warning(f"[{protein}] selection matched no atoms: {selection_str}")
        return None

    sel_res_ids = np.unique(atom_array.res_id[sel_mask])
    sel_chain_ids = np.unique(atom_array.chain_id[sel_mask])

    # Chain is taken from the selection string. Fall back to the
    # mask-matched atoms when the selection has no chain clause.
    chain_from_sel = _chain_from_selection(selection_str)
    if chain_from_sel is None:
        if len(sel_chain_ids) != 1:
            logger.warning(
                f"{protein} selection '{selection_str}' did not specify a chain and "
                f"matched atoms that exist in these chains {sel_chain_ids.tolist()}, skipping"
            )
            return None
        chain = str(sel_chain_ids[0])
    else:
        if not (len(sel_chain_ids) == 1 and str(sel_chain_ids[0]) == chain_from_sel):
            logger.warning(
                f"{protein} selection '{selection_str}' has chain "
                f"'{chain_from_sel}' but mask matched atoms exist in chains "
                f"{sel_chain_ids.tolist()} skipping"
            )
            return None
        chain = chain_from_sel

    sel_altloc_mask = sel_mask & structure_altloc_mask
    covered_altloc_residues: set[tuple[str, int]] = {
        (str(c), int(r))
        for c, r in zip(atom_array.chain_id[sel_altloc_mask], atom_array.res_id[sel_altloc_mask])
    }

    backbone_altloc_mask = sel_altloc_mask & structure_backbone_mask
    backbone_altloc_res_ids = sorted(
        int(r) for r in np.unique(atom_array.res_id[backbone_altloc_mask])
    )
    n_backbone = len(backbone_altloc_res_ids)

    row = {
        "protein": protein,
        "selection": selection_str,
        "chain": chain,
        "start_res": int(sel_res_ids.min()),
        "end_res": int(sel_res_ids.max()),
        "span_length": int(len(sel_res_ids)),
        "n_backbone_altloc_residues": n_backbone,
        "n_altlocs": len(altloc_ids),
        # JSON encoded so the pair calculation can be loaded back through the CSV
        "pair_lddts": json.dumps({}),
        "worst_pair_mean_backbone_lddt": float("nan"),
        "classification": "",
    }

    # Side chain only: no backbone altlocs anywhere in the span.
    if n_backbone == 0:
        row["classification"] = "side_chain_only"
        return row, covered_altloc_residues

    # Domain shift: contiguous backbone-altloc run exceeds threshold (default 50).
    if _max_contiguous_run(backbone_altloc_res_ids) > domain_shift_min_span:
        row["classification"] = "domain_shift"
        return row, covered_altloc_residues

    # Loop classification via pairwise lDDT across all altloc pairs
    pair_lddts: dict[str, float] = {}
    for i in range(len(altloc_ids)):
        for j in range(i + 1, len(altloc_ids)):
            pair = pair_arrays.get((altloc_ids[i], altloc_ids[j]))
            gt, pred = pair if pair is not None else (None, None)
            pair_lddts[f"{altloc_ids[i]}-{altloc_ids[j]}"] = _mean_residue_lddt_for_pair(
                gt, pred, chain, backbone_altloc_res_ids
            )
    row["pair_lddts"] = json.dumps(pair_lddts)

    finite_vals = [v for v in pair_lddts.values() if np.isfinite(v)]
    if not finite_vals:
        raise RuntimeError(
            f"[{protein}] could not compute lDDT for any altloc pair in span "
            f"'{selection_str}' (backbone-altloc residues: {backbone_altloc_res_ids}). "
            "Refusing to emit an indeterminate classification."
        )

    worst = float(min(finite_vals))
    row["worst_pair_mean_backbone_lddt"] = worst
    row["classification"] = "small_loop" if worst > loop_lddt_threshold else "large_loop"
    return row, covered_altloc_residues


def _process_structure(
    row: pd.Series,
    cif_root: Path | None,
    domain_shift_min_span: int,
    loop_lddt_threshold: float,
) -> list[dict]:
    protein = str(row["protein"])
    cif_path = resolve_cif_path(row, cif_root)
    if not cif_path.exists():
        logger.error(f"[{protein}] CIF file not found: {cif_path}")
        return []

    selection_field = row.get("selection", "")
    if not isinstance(selection_field, str) or not selection_field.strip():
        logger.warning(f"[{protein}] no selections in CSV row for {cif_path}")
        return []

    logger.info(f"[{protein}] loading {cif_path}")
    atom_array = load_structure_with_altlocs(cif_path)
    altloc_info = detect_altlocs(atom_array)
    if len(altloc_info.altloc_ids) < 2:
        logger.warning(
            f"[{protein}] structure has <2 altloc IDs ({altloc_info.altloc_ids}); skipping"
        )
        return []

    pair_arrays = _build_pairwise_altloc_arrays(atom_array, altloc_info.altloc_ids)

    structure_altloc_mask = ~np.isin(atom_array.altloc_id, list(BLANK_ALTLOC_IDS))
    structure_backbone_mask = np.isin(atom_array.atom_name, BACKBONE_ATOM_TYPES)

    rows: list[dict] = []
    classified_res_ids: set[tuple[str, int]] = set()
    for selection_str in [s.strip() for s in selection_field.split(";") if s.strip()]:
        # find_altloc_selections.py appends a combined all altloc selection
        # (atomworks-style with " or " clauses) at the end of each row. That one is
        # a union over every span we already processed individually, so skip it.
        if " or " in selection_str:
            continue
        out = _classify_selection(
            atom_array=atom_array,
            pair_arrays=pair_arrays,
            altloc_ids=altloc_info.altloc_ids,
            selection_str=selection_str,
            protein=protein,
            structure_altloc_mask=structure_altloc_mask,
            structure_backbone_mask=structure_backbone_mask,
            domain_shift_min_span=domain_shift_min_span,
            loop_lddt_threshold=loop_lddt_threshold,
        )
        if out is None:
            continue
        row, covered = out
        rows.append(row)
        classified_res_ids.update(covered)

    # residues across all classified spans should equal total unique
    # (chain, res_id) pairs that carry any altloc in the structure.
    all_altloc_res_ids: set[tuple[str, int]] = {
        (str(c), int(r))
        for c, r in zip(
            atom_array.chain_id[structure_altloc_mask],
            atom_array.res_id[structure_altloc_mask],
        )
    }
    if classified_res_ids != all_altloc_res_ids:
        missing = all_altloc_res_ids - classified_res_ids
        extra = classified_res_ids - all_altloc_res_ids
        logger.warning(
            f"[{protein}] residue coverage invariant not satisfied: "
            f"{len(missing)} altloc residues missing from classification, "
            f"{len(extra)} classified residues not in full altloc set. "
            "This typically means --min-span > 1 was used upstream."
        )

    return rows


def main(args: argparse.Namespace) -> None:
    input_df = pd.read_csv(args.input_csv)
    required = {"protein", "selection"}
    missing = required - set(input_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    all_rows: list[dict] = []
    for _, row in input_df.iterrows():
        all_rows.extend(
            _process_structure(
                row=row,
                cif_root=args.cif_root,
                domain_shift_min_span=args.domain_shift_min_span,
                loop_lddt_threshold=args.loop_lddt_threshold,
            )
        )

    out_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_file, index=False)
    logger.info(f"Wrote {len(out_df)} classified spans to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify altloc regions into side_chain_only / small_loop / "
            "large_loop / domain_shift bins. Consumes the CSV produced by "
            "find_altloc_selections.py (run with --min-span 1 to include "
            "side-chain-only regions)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Output CSV from find_altloc_selections.py (must contain 'protein' "
        "and 'selection'. May contain 'structure' or 'structure_pattern').",
    )
    parser.add_argument(
        "--cif-root",
        type=Path,
        default=None,
        help="Optional root directory to resolve 'structure_pattern' entries against.",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--domain-shift-min-span", type=int, default=50)
    parser.add_argument("--loop-lddt-threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args)
