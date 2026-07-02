"""Reproducible preparation of model-input CIFs from RCSB deposits.

A sampleworks input CIF is a *carve* of an RCSB deposit: one or more polymer chains, with
waters / ligands dropped and a chosen residue numbering, written as ``_atom_site`` plus the
carried unit cell / connectivity and the minimal completeness categories that keep it
mmcif-validator clean. (The depositor's real entity sequence is carried onto the prediction
*output* CIF, not the input.)

This module makes that carve reproducible and declarative. A :class:`CarveSpec` names the
chains to keep, what to drop, and the numbering policy; :func:`prepare_input` fetches the
deposit, applies the spec, stamps the result with carve provenance, and caches it under a
stable hash of the spec. :func:`carve` is the pure transform underneath (no network, no write).

The carve is model-agnostic: every predictor wrapper consumes the same ``atomworks.parse`` of
the input (its per-chain canonical sequence and atom array), and all model-specific preparation
happens downstream of that parse. So the correctness gate for a carve is "atomworks parses it
into the right per-chain polymer sequence," not running each model. Modified polymer residues
(e.g. selenomethionine, pyroglutamate) are kept with their deposit names -- atomworks
canonicalizes the sequence for the models, and dropping them here would lose information.

Registry format
---------------
A registry is a JSON object keyed by (lower-case) PDB id, each value a spec dict::

    {"4ole": {"chains": ["B"], "drop": ["water", "ligand"], "numbering": "preserve_auth"}}

:func:`load_registry` reads it into ``{pdb_id: CarveSpec}``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger

from sampleworks.utils.cif_utils import (
    add_category_to_cif,
    add_completeness_categories,
    fetch_rcsb_cif,
)


# Altloc ids that mark an atom as having no alternate conformer (always kept).
_BLANK_ALTLOCS = (".", "?", "")

# Recognised ``drop`` tokens. This cut keeps polymer entities and drops everything else, so the
# only supported configuration is dropping both; the tokens remain a declared, hashable record of
# intent and the anchor for a future richer drop vocabulary (e.g. retaining a ligand).
_VALID_DROP_TOKENS = frozenset({"water", "ligand"})
_DEFAULT_DROP = frozenset({"water", "ligand"})

_NUMBERING_POLICIES = ("preserve_auth", "preserve_label", "from_one")
_DEFAULT_INPUT_CACHE = Path("~/.sampleworks/inputs")

# The standard _atom_site coordinate columns an input CIF carries (the shape the existing inputs
# and the output writer use). Deposit-specific extras (e.g. pdbx_tls_group_id, which references a
# refinement category the input does not carry) are dropped so the file stays mmcif-validator clean.
_STANDARD_ATOM_SITE_COLUMNS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "pdbx_formal_charge",
    "auth_seq_id",
    "auth_comp_id",
    "auth_asym_id",
    "auth_atom_id",
    "pdbx_PDB_model_num",
)


@dataclass
class CarveSpec:
    """A declared transform from an RCSB deposit to a sampleworks input CIF.

    Parameters
    ----------
    chains : list[str]
        Author chain ids (``auth_asym_id``) to keep, in order. Each becomes its own output entity.
    drop : frozenset[str]
        Entity classes to drop. ``"water"`` drops water entities; ``"ligand"`` drops
        non-polymer / branched / macrolide entities. Polymer chains are never dropped, so modified
        polymer residues (HETATM that belong to a polymer entity) are kept. Default
        ``{"water", "ligand"}``.
    numbering : str
        How the kept residues are numbered (output ``label_seq_id == auth_seq_id``):
        ``"preserve_auth"`` uses the deposit author numbering; ``"preserve_label"`` uses the deposit
        canonical (``label_seq_id``) numbering; ``"from_one"`` renumbers ``1..N`` per chain.
    selection : str | None
        Reserved for an atom-selection refinement; not yet implemented (raises if set).
    """

    chains: list[str]
    drop: frozenset[str] = field(default_factory=lambda: frozenset(_DEFAULT_DROP))
    numbering: str = "preserve_auth"
    selection: str | None = None

    def __post_init__(self) -> None:
        if not self.chains:
            raise ValueError("CarveSpec.chains must name at least one chain")
        self.chains = [str(c) for c in self.chains]
        self.drop = frozenset(self.drop)
        unknown = self.drop - _VALID_DROP_TOKENS
        if unknown:
            raise ValueError(f"CarveSpec.drop has unknown tokens {sorted(unknown)}")
        if self.numbering not in _NUMBERING_POLICIES:
            raise ValueError(
                f"CarveSpec.numbering must be one of {_NUMBERING_POLICIES}, got {self.numbering!r}"
            )

    def to_dict(self) -> dict:
        """Serialize to a registry dict (sorted ``drop`` for stable round-trips)."""
        return {
            "chains": list(self.chains),
            "drop": sorted(self.drop),
            "numbering": self.numbering,
            "selection": self.selection,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CarveSpec:
        """Build from a registry dict; missing ``drop``/``numbering`` fall back to defaults."""
        return cls(
            chains=list(data["chains"]),
            drop=frozenset(data.get("drop", _DEFAULT_DROP)),
            numbering=data.get("numbering", "preserve_auth"),
            selection=data.get("selection"),
        )

    def spec_hash(self) -> str:
        """A stable 12-hex-char digest of the spec.

        Canonical: keys sorted, ``chains`` and ``drop`` sorted, so the hash is independent of the
        order chains were listed in. The cache key for a prepared input is ``<pdb_id>__<hash>``.
        """
        canonical = {
            "chains": sorted(self.chains),
            "drop": sorted(self.drop),
            "numbering": self.numbering,
            "selection": self.selection,
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_registry(path: str | Path) -> dict[str, CarveSpec]:
    """Read a JSON carve registry into ``{lower-case pdb_id: CarveSpec}``."""
    raw = json.loads(Path(path).expanduser().read_text())
    return {pdb_id.lower(): CarveSpec.from_dict(spec) for pdb_id, spec in raw.items()}


def _longest_consecutive(nums: list[int]) -> list[int]:
    """Longest run of consecutive integers in a sorted unique list."""
    if not nums:
        return []
    best = cur = [nums[0]]
    for value in nums[1:]:
        cur = cur + [value] if value == cur[-1] + 1 else [value]
        if len(cur) > len(best):
            best = cur
    return best


def _sole_block(cif: CIFFile) -> CIFBlock:
    return cif[list(cif.keys())[0]]


def _entity_types(block: CIFBlock) -> dict[str, str]:
    """Map ``_entity.id -> _entity.type`` from a deposit block (empty if no ``_entity``)."""
    if "entity" not in block:
        return {}
    entity = block["entity"]
    ids = [str(i) for i in entity["id"].as_array(str)]
    if "type" in entity:
        types = [str(t) for t in entity["type"].as_array(str)]
    else:
        types = ["polymer"] * len(ids)
    return dict(zip(ids, types))


def _is_int(value: str) -> bool:
    return value.lstrip("-").isdigit()


def _resolve_altlocs(
    candidate: np.ndarray,
    chains: np.ndarray,
    auth_seq: np.ndarray,
    alt: np.ndarray,
    n: int,
) -> np.ndarray:
    """Pick one conformer per atom among ``candidate`` atoms.

    Keeps blank (``.``/``?``/empty) altlocs plus, for atoms that carry an altloc, the residue's
    preferred conformer -- ``"A"`` when present, otherwise the lowest altloc id in that residue.
    Resolving per residue (rather than a flat "keep altloc A") means a residue whose only conformer
    is labeled (e.g.) ``"B"`` is kept rather than dropped, which would otherwise fragment the chain.
    """
    preferred: dict[tuple[str, str], str] = {}
    for i in np.nonzero(candidate)[0]:
        a = str(alt[i])
        if a in _BLANK_ALTLOCS:
            continue
        key = (str(chains[i]), str(auth_seq[i]))
        if key not in preferred or a < preferred[key]:
            preferred[key] = a
    keep = np.zeros(n, dtype=bool)
    for i in np.nonzero(candidate)[0]:
        a = str(alt[i])
        keep[i] = a in _BLANK_ALTLOCS or a == preferred.get((str(chains[i]), str(auth_seq[i])))
    return keep


def _carry_cell_and_struct_conn(
    dst: CIFFile,
    src_block: CIFBlock,
    numbering_map: dict[tuple[str, str], str],
    kept_by_chain: dict[str, set[str]],
    block_name: str,
) -> None:
    """Carry ``_cell`` (whole) and ``_struct_conn`` (subset to kept residues) from the deposit.

    ``_struct_conn`` rows survive only when *both* partners reference a kept chain and a kept
    residue (so disulfides among kept residues are retained, conns to dropped ligands/waters are
    not). Partner sequence ids are remapped through ``numbering_map`` so they match the carved
    ``_atom_site`` numbering (an identity map under ``preserve_auth``).
    """
    if "cell" in src_block:
        cell = src_block["cell"]
        # Drop the deposit's entry_id so add_completeness_categories can supply one consistent with
        # the output _entry.id (the block name); keeping the deposit's would dangle the foreign key.
        add_category_to_cif(
            dst,
            {key: list(cell[key].as_array(str)) for key in cell if key != "entry_id"},
            "cell",
            overwrite=True,
            block_name=block_name,
        )

    if "struct_conn" not in src_block:
        return
    struct_conn = src_block["struct_conn"]
    cols = {key: list(struct_conn[key].as_array(str)) for key in struct_conn}
    nrows = len(next(iter(cols.values()))) if cols else 0

    def _col(*keys: str) -> list[str]:
        for key in keys:
            if key in cols:
                return cols[key]
        return [""] * nrows

    def _partner(prefix: str, idx: int) -> tuple[str, str]:
        chain = _col(f"{prefix}_auth_asym_id", f"{prefix}_label_asym_id")
        seq = _col(f"{prefix}_auth_seq_id", f"{prefix}_label_seq_id")
        return chain[idx], seq[idx]

    def _kept(chain: str, seq: str) -> bool:
        return chain in kept_by_chain and seq in kept_by_chain[chain]

    keep = []
    for i in range(nrows):
        c1, s1 = _partner("ptnr1", i)
        c2, s2 = _partner("ptnr2", i)
        if _kept(c1, s1) and _kept(c2, s2):
            keep.append(i)
    if not keep:
        return

    new_cols = {key: [values[i] for i in keep] for key, values in cols.items()}
    # Rewrite each partner onto the carved atom_site: identify the residue by its deposit author
    # (chain, seq), then set both the label and auth asym/seq ids to the carved chain and number
    # (label == auth in the carve) so the connectivity resolves against the new _atom_site.
    def _new_col(*keys: str) -> list[str] | None:
        for key in keys:
            if key in new_cols:
                return new_cols[key]
        return None

    for prefix in ("ptnr1", "ptnr2"):
        chains_here = _new_col(f"{prefix}_auth_asym_id", f"{prefix}_label_asym_id")
        seqs_here = _new_col(f"{prefix}_auth_seq_id", f"{prefix}_label_seq_id")
        if chains_here is None or seqs_here is None:
            continue
        renumbered = [numbering_map[(chains_here[j], seqs_here[j])] for j in range(len(keep))]
        for col in (f"{prefix}_auth_seq_id", f"{prefix}_label_seq_id"):
            if col in new_cols:
                new_cols[col] = list(renumbered)
        for col in (f"{prefix}_auth_asym_id", f"{prefix}_label_asym_id"):
            if col in new_cols:
                new_cols[col] = list(chains_here)
    add_category_to_cif(dst, new_cols, "struct_conn", overwrite=True, block_name=block_name)


def carve(original: str | Path | CIFFile, spec: CarveSpec, *, source: str = "input") -> CIFFile:
    """Apply ``spec`` to an RCSB deposit, returning a prepared input ``CIFFile`` (no write).

    Keeps the selected polymer chains (entity-aware, so modified polymer residues survive), drops
    waters / ligands per ``spec.drop``, restricts each chain to its longest contiguous modeled run,
    applies the numbering policy, carries ``_cell`` / ``_struct_conn``, synthesizes the minimal
    completeness categories so the file is mmcif-validator clean, and stamps ``_sampleworks`` carve
    provenance. (The depositor's real entity *sequence* is carried onto the prediction output, not
    the input -- the input needs only the validator's stub parents.)

    Parameters
    ----------
    original : str | Path | CIFFile
        The RCSB deposit (path or already-open ``CIFFile``).
    spec : CarveSpec
        The declared transform.
    source : str
        Provenance label (typically the PDB id); also used as the output block / entry name.
    """
    if spec.selection is not None:
        raise NotImplementedError("CarveSpec.selection is reserved and not yet implemented")
    if spec.drop != _DEFAULT_DROP:
        # Keeping a non-polymer entity needs end-to-end handling (entity typing, numbering,
        # connectivity) the longest-run polymer carve does not yet do -- fail loudly rather than
        # silently drop a ligand the spec asked to keep.
        raise NotImplementedError(
            f"this cut keeps polymer entities only; CarveSpec.drop must be {set(_DEFAULT_DROP)}"
        )

    src = original if isinstance(original, CIFFile) else CIFFile.read(str(original))
    src_block = _sole_block(src)
    atom_site = src_block["atom_site"]
    cols = {key: np.asarray(atom_site[key].as_array(str)) for key in atom_site}
    n = len(next(iter(cols.values())))
    if "label_entity_id" not in cols:
        raise ValueError("deposit _atom_site lacks label_entity_id; cannot apply entity-aware drop")

    entity_types = _entity_types(src_block)
    chain_col = "auth_asym_id" if "auth_asym_id" in atom_site else "label_asym_id"
    chains = cols[chain_col]
    entity_ids = cols["label_entity_id"]
    alt = cols.get("label_alt_id", np.array(["."] * n))
    auth_seq = cols["auth_seq_id"]
    label_seq = cols["label_seq_id"]

    def _entity_kept(entity_id: str) -> bool:
        # Keep polymer entities (which carry modified residues like MSE/PCA as HETATM); drop water
        # and non-polymer. Unknown entity ids default to kept.
        return entity_types.get(entity_id, "polymer") == "polymer"

    type_keep = np.array([_entity_kept(str(e)) for e in entity_ids])
    candidate = type_keep & np.isin(chains, spec.chains)
    if "pdbx_PDB_model_num" in cols:
        first_model = next(iter(dict.fromkeys(cols["pdbx_PDB_model_num"].tolist())))
        candidate &= cols["pdbx_PDB_model_num"] == first_model
    base = candidate & _resolve_altlocs(candidate, chains, auth_seq, alt, n)

    total = np.zeros(n, dtype=bool)
    entity_out = np.array(["?"] * n, dtype=object)
    new_seq = np.array(["?"] * n, dtype=object)
    numbering_map: dict[tuple[str, str], str] = {}
    kept_by_chain: dict[str, set[str]] = {}

    for entity_index, chain in enumerate(spec.chains, start=1):
        mask = base & (chains == chain)
        if not mask.any():
            raise ValueError(f"no kept polymer atoms for chain {chain!r} in {source}")
        present = sorted({int(r) for r in auth_seq[mask] if _is_int(r)})
        run = set(_longest_consecutive(present))
        mask &= np.array([_is_int(r) and int(r) in run for r in auth_seq])

        # Map each kept residue's deposit auth id -> its output sequence id under the policy.
        if spec.numbering == "from_one":
            ordered = sorted({int(r) for r in auth_seq[mask]})
            remap = {str(orig): str(rank) for rank, orig in enumerate(ordered, start=1)}
        elif spec.numbering == "preserve_label":
            remap = {str(auth_seq[i]): str(label_seq[i]) for i in np.nonzero(mask)[0]}
        else:  # preserve_auth
            remap = {str(orig): str(orig) for orig in {int(r) for r in auth_seq[mask]}}

        total |= mask
        entity_out[mask] = str(entity_index)
        kept_by_chain[chain] = set()
        for i in np.nonzero(mask)[0]:
            orig = str(auth_seq[i])
            new_seq[i] = remap[orig]
            numbering_map[(chain, orig)] = remap[orig]
            kept_by_chain[chain].add(orig)

        logger.info(
            f"[carve] {source} chain {chain} -> entity {entity_index}: {int(mask.sum())} atoms "
            f"over res {min(run)}-{max(run)} (numbering={spec.numbering})"
        )

    # Emit the standard coordinate columns only; deposit-specific extras (e.g. pdbx_tls_group_id)
    # reference categories the input does not carry and would fail the validator.
    kept_count = int(total.sum())
    new_cols = {key: list(cols[key][total]) for key in _STANDARD_ATOM_SITE_COLUMNS if key in cols}
    new_cols["label_asym_id"] = list(chains[total])
    new_cols["auth_asym_id"] = list(chains[total])
    new_cols["label_entity_id"] = list(entity_out[total])
    new_cols["label_seq_id"] = list(new_seq[total])
    new_cols["auth_seq_id"] = list(new_seq[total])
    if "id" in new_cols:
        new_cols["id"] = [str(i) for i in range(1, kept_count + 1)]

    block_name = source or "input"
    out = CIFFile()
    out[block_name] = CIFBlock()
    out[block_name]["atom_site"] = CIFCategory(columns=new_cols, name="atom_site")

    # Carry the unit cell and connectivity, then synthesize the minimal parent categories the
    # mmcif-validator requires (a typeless _entity stub, _atom_type, _chem_comp, _struct_conn_type,
    # _entry). No real entity sequence is carried -- that is an output-CIF concern.
    _carry_cell_and_struct_conn(out, src_block, numbering_map, kept_by_chain, block_name)
    add_completeness_categories(out, block_name=block_name, overwrite=True)

    provenance = {
        "sampleworks_carve_source": source,
        "sampleworks_carve_chains": ",".join(spec.chains),
        "sampleworks_carve_drop": ",".join(sorted(spec.drop)),
        "sampleworks_carve_numbering": spec.numbering,
        "sampleworks_carve_spec_hash": spec.spec_hash(),
    }
    add_category_to_cif(out, provenance, "sampleworks", overwrite=True, block_name=block_name)
    return out


def prepare_input(
    pdb_id: str,
    spec: CarveSpec,
    *,
    cache_dir: str | Path = _DEFAULT_INPUT_CACHE,
    rcsb_cache: str | Path | None = None,
) -> Path:
    """Fetch ``pdb_id`` from RCSB, carve per ``spec``, and cache the result; return its path.

    The cache key is ``<pdb_id>__<spec_hash>.cif``. If that file already exists it is returned as-is
    (no re-fetch, no re-carve), mirroring :func:`fetch_rcsb_cif`'s cache-if-present behavior.
    """
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    out_path = cache / f"{pdb_id.lower()}__{spec.spec_hash()}.cif"
    if out_path.exists():
        return out_path

    original = (
        fetch_rcsb_cif(pdb_id, rcsb_cache) if rcsb_cache is not None else fetch_rcsb_cif(pdb_id)
    )
    cif = carve(original, spec, source=pdb_id.lower())
    cif.write(str(out_path))
    return out_path
