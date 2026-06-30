import itertools
import tempfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx.cif import CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.atom_array_utils import (
    find_all_altloc_ids,
    save_structure_to_cif,
    select_altloc,
)


def find_altloc_selections(
    cif_file: Path | str,
    altloc_label: str = "label_alt_id",
    min_span: int = 5,
    include_all_altlocs: bool = True,
) -> Iterable[str]:
    """Find alternative location selections in a CIF file.

    Individual spans at least ``min_span`` residues long are yielded as selection strings.
    Optionally, a final batch of selection strings is also yielded that contains all residues
    with altlocs, one selection per chain.

    Parameters
    ----------
    cif_file : Path | str
        Path to the CIF file.
    altloc_label : str
        Label for alternative location identifier. Default is ``'label_alt_id'``.
        If you don't know it, search for ``"_atom_site"`` in your CIF file to identify it.
    min_span : int
        Minimum number of consecutive residues to consider an altloc selection.
        Spans of altlocs shorter than this are not yielded as selection strings, but ARE
        included in the final selections which includes all residues with altlocs in each chain when
        ``include_all_altlocs=True``.
    include_all_altlocs : bool
        If True (default), yield a final per-chain selection string containing all residues
        with altlocs regardless of span length.

    Yields
    ------
    str
        Alternative location selections, keyed by altloc ID.

    Examples
    --------
    For RCSB PDB entry 5SOP, this should yield items like:
    ``['chain A and resi 125-137', "chain_id == 'A' and ((res_id >= 3 and res_id <= 6) or ...)"]``
    """
    cif_file = Path(cif_file)
    logger.info(f"Finding altloc selections for {cif_file}")
    structure = load_any(cif_file, altloc="all", extra_fields=["occupancy", altloc_label])

    # our other methods rely on the annotation "altloc_id" being present, so we'll add it here.
    structure.set_annotation("altloc_id", structure.get_annotation(altloc_label))

    altlocs = OrderedDict()
    for altloc_id in find_all_altloc_ids(structure):
        altk = select_altloc(structure, altloc_id=altloc_id)
        unique_altk = set((ch, res) for ch, res in zip(altk.chain_id, altk.res_id, strict=True))
        # probably unnecessary but making sure these are consistently ordered
        # FIXME? This is a little clunky. Perhaps should be hierarchical by chain then altloc?
        #   At some point though we'll do altloc selections using correlations/contacts
        #   so this is probably not a big deal.
        altlocs[altloc_id] = sorted(list(unique_altk))

    all_altloc_selections = {}
    for chain, start, end, _ in find_consecutive_residues(altlocs):
        if end - start >= min_span - 1:
            # FIXME use new style selection https://github.com/diff-use/sampleworks/issues/56
            yield f"chain {chain} and resi {start}-{end}"  # old style, more compact, selection

        if include_all_altlocs:
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
    """Find and yield spans of consecutive residues with the same set of altloc identifiers.

    This function processes a dictionary mapping alternate location identifiers (altlocs)
    to (chain_id, residue_id) tuples having that altloc. For each chain_id in the structure,
    it yields spans of consecutive residues when membership in altlocs changes
    or where a break in residue numbering occurs. The yielded spans include information about
    the chain, start residue, end residue, and the corresponding membership.

    Parameters
    ----------
    altlocs : dict[str, list[tuple[str, int]]]
        A dictionary where keys are alternate location identifiers and values are
        lists of tuples representing chain identifiers (str) and residue IDs (int).

    Yields
    ------
    tuple[str, int, int, set[str]]
        A tuple containing the chain, start residue ID, end residue ID, and a set
        of alternate location identifiers representing the membership in the span.

    Examples
    --------
    For RCSB PDB entry 5SOP, this should yield::

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


def resolve_mixed_hetatm_atom_altlocs(cif_path: Path | str) -> Path:
    """Pre-process a CIF file where ATOM and HETATM records with different residue names
    share the same (chain, residue) position via different altloc IDs.

    This occurs when a residue has a modified form (e.g. CSO, cysteic acid) as some
    altlocs and the canonical form (e.g. CYS) as another altloc at the same sequence
    position. Atomworks treats these as two sequential residues rather than alternates,
    inserting a spurious extra residue into the sequence fed to Boltz2.

    Should Atomworks fix the underlying issue in the future, we should remove this method.

    The fix: for each affected position, remove the HETATM (modified) records and keep
    only the ATOM (canonical) records. Also cleans up the ``_struct_conn`` covale bonds
    referencing the removed residues, since ``save_structure_to_cif`` only writes
    ``_atom_site``.

    A warning is logged for every affected (chain, residue) position.

    Parameters
    ----------
    cif_path
        Path to the input CIF file.

    Returns
    -------
    Path
        Path to a fixed temporary CIF file if any positions were modified, or the
        original ``cif_path`` unchanged if no issues were found.
    """
    cif_path = Path(cif_path)
    atom_array = load_any(cif_path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]

    chain_id = atom_array.chain_id
    res_id = atom_array.res_id
    res_name = atom_array.res_name
    hetero = atom_array.hetero

    keep_mask = np.ones(len(atom_array), dtype=bool)
    found_any = False

    for chain in np.unique(chain_id):
        for rid in np.unique(res_id[chain_id == chain]):
            pos_mask = (chain_id == chain) & (res_id == rid)
            has_no_hetatm = np.any(~hetero[pos_mask])
            has_hetatm = np.any(hetero[pos_mask])

            if not (has_no_hetatm and has_hetatm):
                # there are either only HETATM or only ATOM records at this position, or none at all
                continue

            atom_res_names = np.unique(res_name[pos_mask & ~hetero])
            hetatm_res_names = np.unique(res_name[pos_mask & hetero])

            if set(atom_res_names) == set(hetatm_res_names):
                continue  # Same residue name on both — not the case we're fixing

            logger.warning(
                f"Chain {chain}, residue {rid}: found mixed ATOM {list(atom_res_names)} "
                f"and HETATM {list(hetatm_res_names)} records with different residue names "
                f"at the same sequence position. Removing HETATM records to prevent "
                f"atomworks from inserting a duplicate residue into the Boltz2 input sequence."
            )
            keep_mask[pos_mask & hetero] = False
            found_any = True

    if not found_any:
        return cif_path

    fixed_array = atom_array[keep_mask]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", prefix="sampleworks_fixed_cif_", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    save_structure_to_cif(fixed_array, tmp_path)
    logger.info(f"Wrote altloc-fixed CIF to temporary file: {tmp_path}")
    return tmp_path


def _resolve_block(ciffile: CIFFile, block_name: str | None):
    """Return the target CIFBlock, defaulting to the sole block when block_name is None.

    Raises ValueError if the file is empty, or is multi-block and no block_name was given,
    or the named block does not exist.
    """
    if block_name is None:
        # CIFFile is a Mapping, so inherits .keys(), which ultimately iterates over blocks
        blocks = list(ciffile.keys())
        if len(blocks) == 0:
            raise ValueError("CIFFile has no blocks. Cannot add category.")
        if len(blocks) > 1:
            raise ValueError(
                f"CIFFile has multiple blocks: {blocks}. Please specify block_name parameter."
            )
        return ciffile[blocks[0]]
    if block_name not in ciffile:
        raise ValueError(f"Block '{block_name}' not found in CIFFile.")
    return ciffile[block_name]


def add_category_to_cif(
    ciffile: CIFFile,
    data: dict[str, Any],
    category_name: str,
    overwrite: bool = False,
    block_name: str | None = None,
) -> None:
    """Add a custom category in-place to a CIFFile.

    Parameters
    ----------
    ciffile : CIFFile
        The CIF file object to modify.
    data : dict[str, Any]
        Dictionary with column names as keys and column data as values.
    category_name : str
        Name of the category to add (e.g., "custom_data").
    overwrite : bool, optional
        If False and the category already exists, raise RuntimeError. Default is False.
    block_name : str | None, optional
        Name of the block to add the category to. If None, check that there is only
        one block and add to that block. Default is None.

    Raises
    ------
    RuntimeError
        If category already exists and overwrite is False.
    ValueError
        If block_name is None but the file has multiple blocks, or if the specified
        block_name does not exist.

    Examples
    --------
    >>> from biotite.structure.io.pdbx.cif import CIFFile
    >>> ciffile = CIFFile.read("example.cif")  # assuming it contains a single block
    >>> data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
    >>> add_category_to_cif(ciffile, data, "my_custom_data")
    >>> print(ciffile.block["my_custom_data"].serialize())
    loop_
    _my_custom_data.id
    _my_custom_data.value
    1 a
    2 b
    3 c
    >>> data = {"sampleworks_version": "0.4.0", "pdb_id": "1L63"}
    >>> add_category_to_cif(ciffile, data, "sampleworks_metadata")
    >>> print(ciffile.block["sampleworks_metadata"].serialize())
    _sampleworks_metadata.sampleworks_version 0.4.0
    _sampleworks_metadata.pdb_id              1L63
    """
    block = _resolve_block(ciffile, block_name)

    # Check if a category with name category_name already exists
    if category_name in block and not overwrite:
        raise RuntimeError(
            f"Category '{category_name}' already exists in block with value: {block[category_name]}"
        )

    # Create and add the category--remove any None values, CIF requires non-null values
    category = CIFCategory(
        columns={k: _normalize_nulls(v) for k, v in data.items()}, name=category_name
    )
    block[category_name] = category


def _normalize_nulls(value: Any) -> Any:
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return ["?" if item is None else item for item in value]
    return "?" if value is None else value


# Categories that biotite's ``set_structure`` plus ``add_completeness_categories`` regenerate
# from the AtomArray / box. These are deliberately NOT copied on a metadata-preserving round-trip:
# the structural ones are rewritten from the fresh ``_atom_site``, and the completeness parents are
# regenerated against it (copying them stale would reference the old atom_site).
_STRUCTURAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "atom_site",
        "struct_conn",
        "chem_comp_bond",
        "cell",
        "atom_type",
        "chem_comp",
        "entity",
        "struct_conn_type",
        "entry",
    }
)


def read_category_from_cif(
    source: str | Path | CIFFile,
    category_name: str,
    block_name: str | None = None,
    strict: bool = False,
) -> dict[str, str | list[str]] | None:
    """Read a custom category from a CIF file into a plain dict.

    The inverse of ``add_category_to_cif``. A single-row category (such as our ``_sampleworks``
    metadata) yields scalar string values; a multi-row category yields lists. Everything comes
    back as **strings** -- numeric / null coercion is the caller's job. A CIF null is the literal
    ``"?"`` and is returned verbatim.

    Parameters
    ----------
    source : str | Path | CIFFile
        Path to a CIF file, or an already-open ``CIFFile``.
    category_name : str
        Category to read, without the leading underscore (e.g. ``"sampleworks"``).
    block_name : str | None, optional
        Block to read from. If None, the sole block is used (error if multi-block).
    strict : bool, optional
        If False (default), any failure (missing file, multi-block ambiguity, resolve error)
        returns None so callers can fall back cleanly. If True, the underlying error is raised.

    Returns
    -------
    dict[str, str | list[str]] | None
        The category's columns, or None if the category (or file/block) is absent.
    """
    try:
        ciffile = source if isinstance(source, CIFFile) else CIFFile.read(str(source))
        block = _resolve_block(ciffile, block_name)
        if category_name not in block:
            return None
        category = block[category_name]
        result: dict[str, str | list[str]] = {}
        for key in category:
            values = category[key].as_array(str)
            result[key] = str(values[0]) if len(values) == 1 else [str(v) for v in values]
        return result
    except Exception:
        if strict:
            raise
        return None


def copy_custom_categories(
    src: CIFFile,
    dst: CIFFile,
    block_name: str | None = None,
    exclude: Iterable[str] = _STRUCTURAL_CATEGORIES,
    overwrite: bool = True,
) -> list[str]:
    """Copy non-structural (custom) categories from one CIF block into another, in place.

    Used to carry custom metadata (e.g. ``_sampleworks``) across a
    ``load_any -> set_structure -> write`` round-trip, which otherwise drops everything but the
    structural categories biotite regenerates from the AtomArray. Categories in ``exclude`` are
    skipped (see ``_STRUCTURAL_CATEGORIES``).

    Parameters
    ----------
    src, dst : CIFFile
        Source and destination CIF files. The matching block in each is used.
    block_name : str | None, optional
        Block to operate on. If None, the sole block of each file is used.
    exclude : Iterable[str], optional
        Category names to skip. Defaults to the structural / completeness set.
    overwrite : bool, optional
        If True (default), replace categories already present in dst; if False, skip them.

    Returns
    -------
    list[str]
        Names of the categories copied.
    """
    src_block = _resolve_block(src, block_name)
    dst_block = _resolve_block(dst, block_name)
    exclude_set = set(exclude)
    copied: list[str] = []
    for name in src_block:
        if name in exclude_set:
            continue
        if name in dst_block and not overwrite:
            continue
        dst_block[name] = src_block[name]
        copied.append(name)
    return copied


def _unique_preserve(values: Iterable[Any]) -> list[str]:
    """Unique string values, preserving first-seen order (handles numpy str scalars)."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = str(v)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def add_completeness_categories(
    ciffile: CIFFile,
    block_name: str | None = None,
    overwrite: bool = False,
) -> None:
    """Add the minimal parent categories required to pass the PDBe mmcif-validator.

    The validator errors when an ``_atom_site`` item references a parent category that
    is absent. For a sampleworks output CIF (which carries only ``_atom_site`` and the
    custom ``_sampleworks`` category) the three blocking parents are:

    ===========================  ==================  ====================
    child item                   parent category     parent item written
    ===========================  ==================  ====================
    ``_atom_site.type_symbol``   ``atom_type``       ``_atom_type.symbol``
    ``_atom_site.label_comp_id`` ``chem_comp``       ``_chem_comp.id``
    ``_atom_site.label_entity_id`` ``entity``        ``_entity.id``
    ===========================  ==================  ====================

    Values are read from the ``_atom_site`` loop *as written* so the parent ids match the
    child references exactly. This deliberately sidesteps the writer's ``chain_entity`` vs
    ``label_entity_id`` discrepancy: whatever entity id ``set_structure`` actually emitted
    into ``_atom_site`` is what ``_entity.id`` will list.

    Additionally, when the structure was built from the input array, the
    writer also emits ``_struct_conn`` (from bonds) and ``_cell`` (from the box). Those are
    completed too: ``struct_conn_type`` is synthesized as the parent of
    ``_struct_conn.conn_type_id``, and ``_cell.entry_id`` (plus its ``_entry`` parent) is
    added. Both are skipped when the categories are absent, so the bare-predictor path is
    unaffected.

    Must be called AFTER ``set_structure`` has populated ``_atom_site``.

    Parameters
    ----------
    ciffile : CIFFile
        The CIF file object to modify in place.
    block_name : str | None, optional
        Block to operate on. If None, the sole block is used (error if multi-block).
    overwrite : bool, optional
        If False and any of the categories already exist, raise (via add_category_to_cif).
    """
    block = _resolve_block(ciffile, block_name)
    if "atom_site" not in block:
        raise ValueError(
            "CIFFile block has no 'atom_site' category; call set_structure before "
            "add_completeness_categories."
        )
    atom_site = block["atom_site"]

    symbols = _unique_preserve(atom_site["type_symbol"].as_array(str))
    comp_ids = _unique_preserve(atom_site["label_comp_id"].as_array(str))
    entity_ids = _unique_preserve(atom_site["label_entity_id"].as_array(str))

    add_category_to_cif(
        ciffile, {"symbol": symbols}, "atom_type", overwrite=overwrite, block_name=block_name
    )
    add_category_to_cif(
        ciffile, {"id": comp_ids}, "chem_comp", overwrite=overwrite, block_name=block_name
    )
    add_category_to_cif(
        ciffile, {"id": entity_ids}, "entity", overwrite=overwrite, block_name=block_name
    )

    # atomworks/biotite writer serializes the input's connectivity (_struct_conn, from
    # the BondList) and unit cell (_cell, from the array box). Both are incomplete as written
    # and fail the validator. Complete them here. These blocks are no-ops when the categories
    # are absent (e.g. bare predictor output), so the non-renumber path is unchanged.
    if (
        "struct_conn" in block
        and "conn_type_id" in block["struct_conn"]
        and "struct_conn_type" not in block
    ):
        conn_types = _unique_preserve(block["struct_conn"]["conn_type_id"].as_array(str))
        add_category_to_cif(
            ciffile,
            {"id": conn_types},
            "struct_conn_type",
            overwrite=overwrite,
            block_name=block_name,
        )

    if "cell" in block:
        # _cell.entry_id is mandatory and is a child of _entry.id, so provide both with a
        # consistent value (the data block name).
        resolved_block_name = block_name if block_name is not None else list(ciffile.keys())[0]
        entry_id = resolved_block_name or "sampleworks"
        if "entry" not in block:
            add_category_to_cif(
                ciffile, {"id": [entry_id]}, "entry", overwrite=overwrite, block_name=block_name
            )
        if "entry_id" not in block["cell"]:
            cell = block["cell"]
            # Preserve the existing cell parameters (length/angle) and add entry_id.
            cell_data = {"entry_id": [entry_id]}
            cell_data.update({key: list(cell[key].as_array(str)) for key in cell.keys()})
            add_category_to_cif(ciffile, cell_data, "cell", overwrite=True, block_name=block_name)


# The depositor's polymer-entity categories that tortoize (libcifpp) needs to score a chain:
# ``_entity`` (typed), ``_entity_poly`` (sequence), ``_entity_poly_seq`` (numbered residues).
# These are the categories the carve strips and ``add_completeness_categories`` only stubs.
_POLYMER_ENTITY_CATEGORIES: tuple[str, ...] = ("entity", "entity_poly", "entity_poly_seq")

_RCSB_CACHE = Path("~/.sampleworks/rcsb")


def fetch_rcsb_cif(pdb_id: str, cache: str | Path = _RCSB_CACHE) -> Path:
    """Download (and cache) the original mmCIF deposit for ``pdb_id`` from RCSB.

    Reuses the on-disk cache the retired ``patch_output_cif_files.py`` used. biotite's ``fetch``
    only hits the network when the file is not already present, so once the cache is warm this is
    offline-friendly and cheap to call per generation.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB id (case-insensitive), e.g. ``"1vme"``.
    cache : str | Path, optional
        Directory to download into. Defaults to ``~/.sampleworks/rcsb``.

    Returns
    -------
    Path
        Path to the cached ``.cif`` file.
    """
    from biotite.database.rcsb import fetch

    cache_dir = Path(cache).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(fetch(pdb_id, format="cif", target_path=str(cache_dir)))


def _select_polymer_entity_index(entity_poly: CIFCategory, kept_chains: set[str]) -> int:
    """Pick the row of ``_entity_poly`` whose strands cover ``kept_chains``.

    Falls back to the sole row when there is exactly one polymer entity (
    sufficient for single-chain inputs). Raises when neither resolves, so the caller can degrade.
    """
    entity_ids = entity_poly["entity_id"].as_array(str)
    if "pdbx_strand_id" in entity_poly:
        strands = entity_poly["pdbx_strand_id"].as_array(str)
        for i, strand_field in enumerate(strands):
            strand_set = {s.strip() for s in str(strand_field).split(",")}
            if strand_set & kept_chains:
                return i
    if len(entity_ids) == 1:
        return 0
    raise ValueError(
        f"could not match a single polymer entity to chains {sorted(kept_chains)} "
        f"(found {len(entity_ids)} polymer entities)"
    )


def _present_residues(dst_block, kept_chains: set[str]) -> tuple[list[int], list[str]]:
    """Return the kept-chain residues actually present in the dst ``_atom_site``.

    Yields ``(sorted label_seq_id ints, comp_id per residue)`` -- the residues tortoize will try
    to map onto ``_entity_poly_seq``.
    """
    atom_site = dst_block["atom_site"]
    chain_col = "auth_asym_id" if "auth_asym_id" in atom_site else "label_asym_id"
    chains = atom_site[chain_col].as_array(str)
    seq_ids = atom_site["label_seq_id"].as_array(str)
    comps = atom_site["label_comp_id"].as_array(str)
    present: dict[int, str] = {}
    for chain, sid, comp in zip(chains, seq_ids, comps, strict=True):
        if chain in kept_chains and str(sid) not in (".", "?", ""):
            present[int(sid)] = str(comp)
    nums = sorted(present)
    return nums, [present[n] for n in nums]


def _seq_renumber_offset(
    dst_block, ref_nums: list[str], ref_mons: list[str], kept_chains: set[str]
) -> int:
    """Offset to renumber ``_entity_poly_seq.num`` into the dst ``label_seq_id`` scheme.

    The output's present residues should form a contiguous block of the canonical sequence; we
    locate that block by matching the residue-name sequence and return the integer shift that
    lines ``num`` up with ``label_seq_id``. Raises if the match is not unique (caller degrades).
    """
    dst_nums, dst_comps = _present_residues(dst_block, kept_chains)
    if not dst_nums:
        raise ValueError("no kept-chain residues with a label_seq_id in the output")
    ref_pairs = sorted(zip((int(n) for n in ref_nums), ref_mons, strict=True))
    ref_num_seq = [n for n, _ in ref_pairs]
    ref_mon_seq = [m for _, m in ref_pairs]
    span = len(dst_comps)
    starts = [
        i for i in range(len(ref_mon_seq) - span + 1) if ref_mon_seq[i : i + span] == dst_comps
    ]
    if len(starts) != 1:
        raise ValueError(
            f"cannot uniquely align {span} output residues to the reference sequence "
            f"({len(starts)} candidate alignments)"
        )
    return dst_nums[0] - ref_num_seq[starts[0]]


def carry_entity_categories(
    dst: CIFFile,
    reference: str | Path | CIFFile,
    *,
    output_entity_id: str = "0",
    kept_chains: Iterable[str] | None = None,
    reconcile: bool = False,
    block_name: str | None = None,
) -> list[str]:
    """Graft the depositor's real polymer-entity categories onto an output CIF.

    Replaces the minimal ``_entity`` stub from :func:`add_completeness_categories` with the real
    ``_entity`` (typed) / ``_entity_poly`` (sequence) / ``_entity_poly_seq`` (numbered residues)
    read from ``reference`` (a full RCSB deposit), so tortoize (G2) can produce real z-scores while
    G1 stays satisfied. Also writes a fresh ``_struct_asym`` mapping each kept chain to the entity.

    The reference entity is subset to the chain(s) actually present in the output and relabelled to
    ``output_entity_id`` (the id ``set_structure`` emitted into the output ``_atom_site``). When
    ``reconcile`` is set, ``_entity_poly_seq.num`` is shifted into the output's ``label_seq_id``
    numbering (needed when the raw deposit numbers its sequence differently from the SW input).

    Parameters
    ----------
    dst : CIFFile
        The output CIF to modify in place. Must already carry ``_atom_site``.
    reference : str | Path | CIFFile
        The full deposit (e.g. from :func:`fetch_rcsb_cif`) carrying the real entity categories.
    output_entity_id : str, optional
        Entity id to relabel onto, matching the output ``_atom_site.label_entity_id``. Default "0".
    kept_chains : Iterable[str] | None, optional
        Chains to keep. Defaults to the chains present in the dst ``_atom_site``.
    reconcile : bool, optional
        If True, renumber ``_entity_poly_seq.num`` to the output's ``label_seq_id`` scheme.
    block_name : str | None, optional
        Block of ``dst`` to operate on. If None, its sole block is used.

    Returns
    -------
    list[str]
        Names of the categories written (``entity``, ``entity_poly``, ``entity_poly_seq``,
        ``struct_asym``).

    Raises
    ------
    ValueError
        If the reference lacks the polymer-entity categories, no single polymer entity matches the
        kept chains, or (with ``reconcile``) the sequence cannot be uniquely aligned. Callers wrap
        this to degrade to the minimal synthesis.
    """
    ref_file = reference if isinstance(reference, CIFFile) else CIFFile.read(str(reference))
    dst_block = _resolve_block(dst, block_name)
    ref_block = _resolve_block(ref_file, None)

    missing = [c for c in _POLYMER_ENTITY_CATEGORIES if c not in ref_block]
    if missing:
        raise ValueError(f"reference CIF lacks {missing}; cannot carry entity categories")

    if kept_chains is None:
        atom_site = dst_block["atom_site"]
        chain_col = "auth_asym_id" if "auth_asym_id" in atom_site else "label_asym_id"
        kept = set(_unique_preserve(atom_site[chain_col].as_array(str)))
    else:
        kept = {str(c) for c in kept_chains}

    entity_poly = ref_block["entity_poly"]
    poly_idx = _select_polymer_entity_index(entity_poly, kept)
    ref_entity_id = str(entity_poly["entity_id"].as_array(str)[poly_idx])

    # _entity: the single row for the chosen entity, relabelled.
    entity = ref_block["entity"]
    ent_mask = np.asarray(entity["id"].as_array(str)) == ref_entity_id
    entity_data = {k: list(np.asarray(entity[k].as_array(str))[ent_mask]) for k in entity}
    entity_data["id"] = [output_entity_id]

    # _entity_poly: the chosen row, relabelled and re-stranded to the kept chains.
    entity_poly_data = {k: [str(entity_poly[k].as_array(str)[poly_idx])] for k in entity_poly}
    entity_poly_data["entity_id"] = [output_entity_id]
    if "pdbx_strand_id" in entity_poly_data:
        entity_poly_data["pdbx_strand_id"] = [",".join(sorted(kept))]

    # _entity_poly_seq: all residues for the chosen entity, relabelled (and optionally renumbered).
    eps = ref_block["entity_poly_seq"]
    eps_mask = np.asarray(eps["entity_id"].as_array(str)) == ref_entity_id
    eps_data = {k: list(np.asarray(eps[k].as_array(str))[eps_mask]) for k in eps}
    eps_data["entity_id"] = [output_entity_id] * int(eps_mask.sum())
    if reconcile:
        offset = _seq_renumber_offset(dst_block, eps_data["num"], eps_data["mon_id"], kept)
        eps_data["num"] = [str(int(n) + offset) for n in eps_data["num"]]

    # _struct_asym: one row per kept chain -> the entity (built fresh, not copied).
    struct_asym_data = {"id": sorted(kept), "entity_id": [output_entity_id] * len(kept)}

    # _chem_comp must list every residue the carried _entity_poly_seq references (mon_id is a
    # foreign key into _chem_comp.id), not just the modeled residues add_completeness_categories
    # saw in _atom_site. The canonical sequence can include residues absent from the modeled subset
    # (e.g. an N-terminal MET), so extend _chem_comp to the union or mmcif-validator (G1) errors.
    atom_site = dst_block["atom_site"]
    comp_ids = _unique_preserve(
        list(atom_site["label_comp_id"].as_array(str)) + list(eps_data["mon_id"])
    )

    add_category_to_cif(dst, entity_data, "entity", overwrite=True, block_name=block_name)
    add_category_to_cif(dst, entity_poly_data, "entity_poly", overwrite=True, block_name=block_name)
    add_category_to_cif(dst, eps_data, "entity_poly_seq", overwrite=True, block_name=block_name)
    add_category_to_cif(dst, struct_asym_data, "struct_asym", overwrite=True, block_name=block_name)
    add_category_to_cif(dst, {"id": comp_ids}, "chem_comp", overwrite=True, block_name=block_name)
    return ["entity", "entity_poly", "entity_poly_seq", "struct_asym", "chem_comp"]


def model_numbers(ciffile: CIFFile, block_name: str | None = None) -> list[str]:
    """Distinct ``_atom_site.pdbx_PDB_model_num`` values, first-seen order.

    Returns ``["1"]`` when the column is absent (a single-model file).
    """
    block = _resolve_block(ciffile, block_name)
    atom_site = block["atom_site"]
    if "pdbx_PDB_model_num" not in atom_site:
        return ["1"]
    return _unique_preserve(atom_site["pdbx_PDB_model_num"].as_array(str))


def extract_model(ciffile: CIFFile, model_num: str | int, block_name: str | None = None) -> None:
    """Filter ``_atom_site`` to a single model in place, preserving all other categories.

    Single-model tools (tortoize) load zero atoms from a stacked ensemble. Extracting via
    ``set_structure`` (i.e. ``save_structure_to_cif``) would rebuild the block and drop the carried
    ``_entity``/``_entity_poly``/``_entity_poly_seq``; this keeps them by filtering the
    ``_atom_site`` loop directly. A no-op when ``pdbx_PDB_model_num`` is absent.

    Parameters
    ----------
    ciffile : CIFFile
        File to filter in place.
    model_num : str | int
        The ``pdbx_PDB_model_num`` value to keep.
    block_name : str | None, optional
        Block to operate on. If None, the sole block is used.

    Raises
    ------
    ValueError
        If ``model_num`` is not present in ``_atom_site``.
    """
    block = _resolve_block(ciffile, block_name)
    atom_site = block["atom_site"]
    if "pdbx_PDB_model_num" not in atom_site:
        return
    model_col = np.asarray(atom_site["pdbx_PDB_model_num"].as_array(str))
    mask = model_col == str(model_num)
    if not mask.any():
        raise ValueError(f"model {model_num} not in atom_site (have {_unique_preserve(model_col)})")
    cols = {k: list(np.asarray(atom_site[k].as_array(str))[mask]) for k in atom_site}
    block["atom_site"] = CIFCategory(columns=cols, name="atom_site")
