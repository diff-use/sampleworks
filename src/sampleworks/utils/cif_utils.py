import itertools
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from atomworks.io.utils.io_utils import load_any
from loguru import logger

from sampleworks.utils.atom_array_utils import find_all_altloc_ids, select_altloc


def find_altloc_selections(
    cif_file: Path | str, altloc_label: str = "label_alt_id", min_span: int = 5
) -> Iterable[str]:
    """
    Find alternative location selections in a CIF file. Individual spans at least min_span
    residues long are yielded as selection strings. A final batch of selection strings
    is also yielded that contains all residues with altlocs, one selection per chain.

    Parameters:
    - cif_file (Path | str): Path to the CIF file.
    - altloc_label (str):
        Label for alternative location identifier. Default is 'label_alt_id'.
        If you don't know it, search for "_atom_site" in your CIF file to identify it.
    - min_span (int): Minimum number of consecutive residues to consider an altloc selection.
        spans of altlocs shorter than this are not yielded as selection strings, but ARE
        included in the final selections which includes all residues with altlocs in each chain.

    Yields:
    - Iterable[str]: Iterable of alternative location selections, keyed by altloc ID.

    Example: for RCSB PDB entry 5SOP, this should yield items like:
    ['chain A and resi 3-6', 'chain A and resi 10-12', 'chain A and resi 20-26', ...,
     'chain_id == 'A' and (res_id == 3 or res_id == 10 or res_id == 20 or ...)]

    """
    cif_file = Path(cif_file)
    logger.info(f"Finding altloc selections for {cif_file}")
    structure = load_any(cif_file, altloc="all", extra_fields=["occupancy", altloc_label])

    # our other methods rely on the annotation "altloc_id" being present, so we'll add it here.
    structure.set_annotation("altloc_id", structure.get_annotation(altloc_label))

    altlocs = OrderedDict()
    for altloc_id in find_all_altloc_ids(structure):
        altk = select_altloc(structure, altloc_id=altloc_id)
        unique_altk = set((ch, res) for ch, res in zip(altk.chain_id, altk.res_id))
        # probably unnecessary but making sure these are consistently ordered
        # FIXME? This is a little clunky. Perhaps should be hierarchical by chain then altloc?
        #   At some point though we'll do altloc selections using correlations/contacts
        #   so this is probably not a big deal.
        altlocs[altloc_id] = sorted(list(unique_altk))

    all_altloc_selections = {}
    for chain, start, end, _ in find_consecutive_residues(altlocs):
        if end - start >= min_span:
            # FIXME use new style selection https://github.com/diff-use/sampleworks/issues/56
            yield f"chain {chain} and resi {start}-{end}"  # old style, more compact, selection

        if chain not in all_altloc_selections:
            all_altloc_selections[chain] = []
        if start == end:
            all_altloc_selections[chain].append(f"(res_id == {start})")
        else:
            all_altloc_selections[chain].append(f"(res_id >= {start} and res_id <= {end})")

    for chain, selections in all_altloc_selections.items():
        yield f"chain_id == '{chain}' and ({' or '.join(selections)})"


def find_consecutive_residues(
    altlocs: dict[str, list[tuple[str, int]]],  # Ex: {'A': [('X', 1), ('X', 2), ('X', 3)]}
) -> Iterable[tuple[str, int, int, set[str]]]:
    """
    Find and yield spans of consecutive residues with the same set of
    alternate location identifiers.

    This function processes a dictionary mapping alternate location identifiers (altlocs)
    to (chain_id, residue_id) tuples having that altloc. For each chain_id in the structure,
    it yields spans of consecutive residues when membership in altlocs changes
    or where a break in residue numbering occurs. The yieled spans include information about
    the chain, start residue, end residue, and the corresponding membership.

    Arguments:
        altlocs (dict[str, list[tuple[str, int]]]): A dictionary where keys are
        alternate location identifiers and values are lists of tuples representing
        chain identifiers (str) and residue IDs (int).

    Yields:
        tuple: A tuple containing the chain (str), start residue ID (int), end residue
        ID (int), and a set of alternate location identifiers representing the
        membership in the span.

    Example: for RCSB PDB entry 5SOP, this should yield:
        [('A', 3, 6, {'A', 'B'}),
         ('A', 10, 12, {'A', 'B'}),
         ('A', 20, 26, {'A', 'B'}),
         ('A', 28, 31, {'A', 'B'}),
         ('A', 38, 38, {'A', 'B'}),
         ('A', 42, 42, {'A', 'B'}),
         ('A', 44, 59, {'A', 'B'}),
         ('A', 87, 88, {'A', 'B'}),
         ('A', 97, 108, {'A', 'B'}),
         ('A', 113, 113, {'A', 'B'}),
         ('A', 125, 137, {'A', 'B', 'C'}),
         ('A', 138, 141, {'A', 'B'}),
         ('A', 155, 169, {'A', 'B'})]

    """
    # TODO create test cases from 5SOP and 7Z0E, low priority since this isn't a critical function
    #   and will likely change in the future anyway.
    #   https://github.com/diff-use/sampleworks/issues/111

    # First find the chains
    all_chains = {res[0] for altloc in altlocs.values() for res in altloc}

    # iterating over chains, check each residue's membership in altlocs.
    # Yield spans when membership changes or there is a break in the residue number
    for chain in all_chains:
        chain_altlocs = {
            altloc_id: {res[1] for res in altlocs[altloc_id] if res[0] == chain}
            for altloc_id in altlocs
        }
        all_res_ids = sorted(list(set.union(*chain_altlocs.values())))
        if not all_res_ids:
            continue

        start = all_res_ids[0]
        next_res_id = None
        current_membership = {k for k in chain_altlocs if start in chain_altlocs[k]}
        start = start if len(current_membership) > 1 else None
        for current_res_id, next_res_id in itertools.pairwise(all_res_ids):
            res_membership = {k for k in chain_altlocs if next_res_id in chain_altlocs[k]}
            if res_membership != current_membership or next_res_id - current_res_id > 1:
                if start is not None:
                    yield chain, start, current_res_id, current_membership

                start = next_res_id if len(res_membership) > 1 else None
                current_membership = res_membership if len(res_membership) > 1 else None
        if start is not None and next_res_id:
            yield chain, start, next_res_id, current_membership
