"""Check the entity-carry + reconcile against tortoize, per PDB.

Builds four single-model variants of a "modeled output" and runs tortoize on each:

  bare            _atom_site only (entity family stripped) -- does CCD inference alone suffice?
  minimal         completeness stub only (add_completeness_categories: a typeless _entity.id)
  carried         real entity grafted from the deposit, num carried verbatim   (reconcile=False)
  carried+reconc  real entity grafted, num shifted onto label_seq_id           (reconcile=True)

A variant "passes" when tortoize returns non-null ramachandran-z / torsion-z. The output tells you
(a) whether the re-fetch works, (b) whether reconcile is needed, (c) whether bare alone suffices.
The per-PDB "reconcile offset" is how far reconcile shifted _entity_poly_seq.num (0 == no-op).

Two ways to supply the modeled output:

  --refined <cif>   use an existing refined.cif (a real prediction output) as the coordinate source.
  --from-deposit    derive the modeled output from the RCSB deposit itself (one polymer chain,
                    primary conformer), re-basing label_seq_id := auth_seq_id so the numbering
                    matches what save_everything emits (auth == label == res_id). Exercises the
                    carry + reconcile without a prediction run; the offset a real run produces
                    depends on the input numbering.

tortoize is not importable; it is shelled out -- point --tortoize at your install. Run this in an
environment with sampleworks + biotite + atomworks.

Example
-------
    python scripts/verify_entity_carry_tortoize.py --from-deposit --pdb-id 6b8x --tortoize tortoize
    python scripts/verify_entity_carry_tortoize.py --refined path/to/refined.cif --pdb-id 6b8x
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx.cif import CIFBlock, CIFCategory, CIFFile
from loguru import logger
from sampleworks.utils.cif_utils import (
    add_completeness_categories,
    carry_entity_categories,
    extract_model,
    fetch_rcsb_cif,
)


_ENTITY_FAMILY = ("entity", "entity_poly", "entity_poly_seq", "struct_asym")

# Primary-conformer altloc ids kept when carving a chain out of a deposit.
_PRIMARY_ALTLOCS = (".", "?", "", "A")


def _block(cif: CIFFile):
    return cif[list(cif.keys())[0]]


def _strip_entity_family(cif: CIFFile) -> None:
    """Remove the entity-family categories in place, leaving a bare-ish structure."""
    block = _block(cif)
    for cat in _ENTITY_FAMILY:
        if cat in block:
            del block[cat]


def _run_tortoize(tortoize_cmd: str, cif_path: Path, out_json: Path) -> dict:
    """Run tortoize, returning the per-model protein-level z-scores keyed by model id."""
    cmd = shlex.split(tortoize_cmd) + [str(cif_path), str(out_json)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    result = json.loads(out_json.read_text())
    models = result.get("model", {})
    return {
        mid: {
            "ramachandran-z": mdata.get("ramachandran-z"),
            "torsion-z": mdata.get("torsion-z"),
        }
        for mid, mdata in models.items()
    }


def _bare_single_model(refined: Path, model_num: int, workdir: Path) -> Path:
    """Write a bare single-model CIF (entity stripped, one model) from an existing refined.cif."""
    cif = CIFFile.read(str(refined))
    _strip_entity_family(cif)
    extract_model(cif, model_num)
    out = workdir / "bare_single.cif"
    cif.write(str(out))
    return out


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


def _bare_single_model_from_deposit(
    reference: Path,
    workdir: Path,
    *,
    chain: str | None = None,
    entity_id: str = "0",
    inject_offset: int = 0,
) -> Path:
    """Derive a bare single-model coordinate source from an RCSB deposit.

    Keeps one polymer chain's primary conformer (group_PDB == ATOM, single altloc, first model)
    restricted to its largest gap-free residue run (so the residues form a contiguous block that
    ``_seq_renumber_offset`` can match), strips the entity family, makes the chain/entity ids
    self-consistent, and sets ``label_seq_id := auth_seq_id + inject_offset`` so the numbering
    matches what ``save_everything`` emits (auth == label == res_id).

    The deposit's ``_entity_poly_seq.num`` stays canonical 1-based, so carrying it back with
    ``reconcile`` recovers ``(auth + inject_offset) - canonical``. With ``inject_offset == 0``
    this is the deposit's natural author-vs-canonical gap; a non-zero ``inject_offset`` forces a
    known offset on real coordinates.
    """
    ref = CIFFile.read(str(reference))
    atom_site = _block(ref)["atom_site"]
    cols = {k: np.asarray(atom_site[k].as_array(str)) for k in atom_site}
    n = len(next(iter(cols.values())))

    group = cols.get("group_PDB", np.array(["ATOM"] * n))
    chain_col = "auth_asym_id" if "auth_asym_id" in atom_site else "label_asym_id"
    chains = cols[chain_col]
    is_atom = group == "ATOM"
    if not is_atom.any():
        raise ValueError(f"no ATOM (polymer) rows in {reference}")
    if chain is None:
        chain = str(chains[np.argmax(is_atom)])  # first polymer chain
    alt = cols.get("label_alt_id", np.array(["."] * n))

    mask = is_atom & (chains == chain) & np.isin(alt, _PRIMARY_ALTLOCS)
    if "pdbx_PDB_model_num" in cols:
        first_model = list(dict.fromkeys(cols["pdbx_PDB_model_num"].tolist()))[0]
        mask &= cols["pdbx_PDB_model_num"] == first_model
    if not mask.any():
        raise ValueError(f"no primary-conformer ATOM rows for chain {chain!r} in {reference}")

    auth_seq = cols["auth_seq_id"]
    present = sorted({int(r) for r in auth_seq[mask] if r.lstrip("-").isdigit()})
    run = set(_longest_consecutive(present))
    in_run = np.array([r.lstrip("-").isdigit() and int(r) in run for r in auth_seq])
    mask &= in_run
    kept = int(mask.sum())

    new_cols = {k: list(v[mask]) for k, v in cols.items()}
    # Self-consistent single chain/entity so libcifpp links _struct_asym -> _atom_site cleanly.
    new_cols["label_asym_id"] = [chain] * kept
    new_cols["auth_asym_id"] = [chain] * kept
    new_cols["label_entity_id"] = [entity_id] * kept
    new_cols["label_seq_id"] = [str(int(r) + inject_offset) for r in auth_seq[mask]]

    out = CIFFile()
    out["model"] = CIFBlock()
    out["model"]["atom_site"] = CIFCategory(columns=new_cols, name="atom_site")
    bare = workdir / "bare_single.cif"
    out.write(str(bare))
    logger.info(
        f"[deposit->modeled] chain {chain}: {kept} atoms over res "
        f"{min(run) if run else '?'}-{max(run) if run else '?'} "
        f"(label_seq_id <- auth_seq_id{f' + {inject_offset}' if inject_offset else ''})"
    )
    return bare


def _summarize(scores: dict) -> str:
    if not scores:
        return "NO MODELS (0 atoms loaded)"
    parts = []
    for mid, s in scores.items():
        rama, tors = s["ramachandran-z"], s["torsion-z"]
        real = rama is not None and tors is not None
        flag = "REAL" if real else "NULL"
        parts.append(f"model {mid}: rama-z={rama} torsion-z={tors} [{flag}]")
    return "; ".join(parts)


def _reconcile_offset(variants: dict[str, CIFFile]) -> int | None:
    """How far reconcile shifted _entity_poly_seq.num (carried+reconc minus carried), or None.

    reconcile shifts every num by a constant, so the first-num difference is the offset.
    """

    def first_num(cif: CIFFile) -> int | None:
        block = _block(cif)
        if "entity_poly_seq" not in block:
            return None
        nums = block["entity_poly_seq"]["num"].as_array(str)
        return int(nums[0]) if len(nums) else None

    if "carried" in variants and "carried+reconc" in variants:
        verbatim, reconciled = first_num(variants["carried"]), first_num(variants["carried+reconc"])
        if verbatim is not None and reconciled is not None:
            return reconciled - verbatim
    return None


def _run_one(pdb_id: str, args: argparse.Namespace, workdir_root: Path) -> tuple[dict, int | None]:
    """Build the four variants for one PDB, score each with tortoize, return (results, offset)."""
    workdir = workdir_root / pdb_id
    workdir.mkdir(parents=True, exist_ok=True)
    reference = fetch_rcsb_cif(pdb_id)
    logger.info(f"[{pdb_id}] reference deposit: {reference}")

    if args.from_deposit:
        bare = _bare_single_model_from_deposit(
            reference,
            workdir,
            chain=args.chain,
            entity_id=args.entity_id,
            inject_offset=args.inject_offset,
        )
    else:
        if args.refined is None:
            raise ValueError("--refined is required unless --from-deposit is set")
        refined = args.refined.expanduser()
        if not refined.exists():
            raise FileNotFoundError(f"--refined not found: {refined}")
        bare = _bare_single_model(refined, args.model_num, workdir)

    def fresh() -> CIFFile:
        return CIFFile.read(str(bare))

    variants: dict[str, CIFFile] = {"bare": fresh()}

    minimal = fresh()
    add_completeness_categories(minimal, overwrite=True)
    variants["minimal"] = minimal

    for name, reconcile in (("carried", False), ("carried+reconc", True)):
        cif = fresh()
        try:
            written = carry_entity_categories(
                cif, reference, output_entity_id=args.entity_id, reconcile=reconcile
            )
            logger.info(f"[{pdb_id}][{name}] wrote {written}")
            variants[name] = cif
        except Exception as e:  # noqa: BLE001 - report and continue
            logger.warning(f"[{pdb_id}][{name}] carry failed: {e}")

    results: dict[str, str] = {}
    for name, cif in variants.items():
        cif_path = workdir / f"{name.replace('+', '_')}.cif"
        cif.write(str(cif_path))
        out_json = workdir / f"{name.replace('+', '_')}.tortoize.json"
        try:
            results[name] = _summarize(_run_tortoize(args.tortoize, cif_path, out_json))
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace").strip().splitlines()
            tail = stderr[-1] if stderr else f"exit {e.returncode}"
            results[name] = f"tortoize FAILED: {tail}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"ERROR: {e}"

    return results, _reconcile_offset(variants)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refined",
        type=Path,
        default=None,
        help="An existing refined.cif (ensemble or single-model) as the coordinate source. "
        "Required unless --from-deposit.",
    )
    parser.add_argument(
        "--from-deposit",
        action="store_true",
        help="Derive the modeled output from the RCSB deposit instead of --refined.",
    )
    parser.add_argument(
        "--pdb-id",
        nargs="+",
        required=True,
        help="One or more PDB ids to re-fetch from RCSB.",
    )
    parser.add_argument(
        "--chain",
        default=None,
        help="(--from-deposit) which chain to keep; default the first polymer chain.",
    )
    parser.add_argument(
        "--inject-offset",
        type=int,
        default=0,
        help="(--from-deposit) shift the modeled label_seq_id by this much to force a known "
        "non-zero reconcile offset on real coordinates. Default 0 (the deposit's natural gap).",
    )
    parser.add_argument("--model-num", type=int, default=1, help="Model to extract and score.")
    parser.add_argument("--entity-id", default="0", help="Output label_entity_id to relabel onto.")
    parser.add_argument(
        "--tortoize",
        default="tortoize",
        help="Command that runs tortoize; the cif and output-json paths are appended.",
    )
    parser.add_argument("--workdir", type=Path, default=None, help="Where to write variant CIFs.")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="verify_entity_carry_") if args.workdir is None else None
    workdir_root = args.workdir or Path(tmp)
    workdir_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"workdir: {workdir_root}")

    all_results: dict[str, tuple[dict, int | None] | None] = {}
    for pdb_id in args.pdb_id:
        try:
            all_results[pdb_id] = _run_one(pdb_id, args, workdir_root)
        except Exception as e:  # noqa: BLE001 - one bad deposit must not abort the rest
            logger.error(f"[{pdb_id}] run failed: {e}")
            all_results[pdb_id] = None

    print("\n==================== verify entity-carry / tortoize ====================")
    print(f"mode    : {'from-deposit' if args.from_deposit else f'refined={args.refined}'}")
    for pdb_id in args.pdb_id:
        print(f"\n--- {pdb_id} ---")
        out = all_results[pdb_id]
        if out is None:
            print("  (run failed -- see log above)")
            continue
        results, offset = out
        print(f"  reconcile offset : {offset}{' (no-op)' if offset == 0 else ''}")
        for name in ("bare", "minimal", "carried", "carried+reconc"):
            print(f"  {name:16s} {results.get(name, '(skipped)')}")
    print("========================================================================")
    print(
        "A 'carried' variant should show REAL z-scores. If only carried+reconc is REAL,\n"
        "reconcile is needed for this deposit's numbering. If bare is also REAL, tortoize\n"
        "can score the bare model and the carried entity only makes the file carry a typed _entity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
