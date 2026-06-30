"""
It builds four single-model variants from an existing refined.cif and runs tortoize on each:

  bare            _atom_site only (entity family stripped) -- does CCD inference alone suffice?
  minimal         today's output: add_completeness_categories (stub _entity.id) -- expected to break
  carried         real entity grafted from RCSB, num carried verbatim   (reconcile=False)
  carried+reconc  real entity grafted from RCSB, num shifted to label_seq_id (reconcile=True)

A variant "passes" when tortoize returns non-null ramachandran-z / torsion-z. The outcome tells us
(a) whether re-fetch works at all, (b) whether reconcile is needed, and (c) whether bare would do.

tortoize is NOT importable; it is shelled out. Point --tortoize at your install, e.g. the default
``conda run -n cifval tortoize``. Run this script in an env with sampleworks + biotite + atomworks.

Example
-------
    python scripts/verify_entity_carry_tortoize.py \
        --refined ~/dev/sampleworks_scratch/r3_entitycarry_prototype/refined_entitycarry.cif \
        --pdb-id 1vme
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from biotite.structure.io.pdbx.cif import CIFFile
from loguru import logger
from sampleworks.utils.cif_utils import (
    add_completeness_categories,
    carry_entity_categories,
    extract_model,
    fetch_rcsb_cif,
)


_ENTITY_FAMILY = ("entity", "entity_poly", "entity_poly_seq", "struct_asym")

# The validated lab-file prototype baseline (1vme, model 1) for eyeballing.
_PROTOTYPE_BASELINE = {"ramachandran-z": 0.20, "torsion-z": 3.51}


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
    """Write a bare single-model CIF (entity stripped, one model) reused across variants."""
    cif = CIFFile.read(str(refined))
    _strip_entity_family(cif)
    extract_model(cif, model_num)
    out = workdir / "bare_single.cif"
    cif.write(str(out))
    return out


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refined",
        type=Path,
        default=Path(
            "~/dev/sampleworks_scratch/r3_entitycarry_prototype/refined_entitycarry.cif"
        ).expanduser(),
        help="An existing refined.cif (ensemble or single-model) to use as the coordinate source.",
    )
    parser.add_argument("--pdb-id", default="1vme", help="PDB id to re-fetch from RCSB.")
    parser.add_argument("--model-num", type=int, default=1, help="Model to extract and score.")
    parser.add_argument("--entity-id", default="0", help="Output label_entity_id to relabel onto.")
    parser.add_argument(
        "--tortoize",
        default="conda run -n cifval tortoize",
        help="Command that runs tortoize; the cif and output-json paths are appended.",
    )
    parser.add_argument("--workdir", type=Path, default=None, help="Where to write variant CIFs.")
    args = parser.parse_args()

    if not args.refined.exists():
        logger.error(f"--refined not found: {args.refined}")
        return 1

    tmp = tempfile.mkdtemp(prefix="verify_entity_carry_") if args.workdir is None else None
    workdir = args.workdir or Path(tmp)
    workdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"workdir: {workdir}")

    reference = fetch_rcsb_cif(args.pdb_id)
    logger.info(f"reference deposit: {reference}")

    bare = _bare_single_model(args.refined, args.model_num, workdir)

    def fresh() -> CIFFile:
        return CIFFile.read(str(bare))

    variants: dict[str, CIFFile] = {}

    # bare: entity family already stripped.
    variants["bare"] = fresh()

    # minimal: today's add_completeness_categories synthesis (stub _entity.id).
    minimal = fresh()
    add_completeness_categories(minimal, overwrite=True)
    variants["minimal"] = minimal

    # carried (verbatim num) and carried+reconcile.
    for name, reconcile in (("carried", False), ("carried+reconc", True)):
        cif = fresh()
        try:
            written = carry_entity_categories(
                cif, reference, output_entity_id=args.entity_id, reconcile=reconcile
            )
            logger.info(f"[{name}] wrote {written}")
            variants[name] = cif
        except Exception as e:  # noqa: BLE001 - report and continue
            logger.warning(f"[{name}] carry failed: {e}")

    results: dict[str, str] = {}
    for name, cif in variants.items():
        cif_path = workdir / f"{name.replace('+', '_')}.cif"
        cif.write(str(cif_path))
        out_json = workdir / f"{name.replace('+', '_')}.tortoize.json"
        try:
            scores = _run_tortoize(args.tortoize, cif_path, out_json)
            results[name] = _summarize(scores)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode(errors="replace").strip().splitlines()
            tail = stderr[-1] if stderr else f"exit {e.returncode}"
            results[name] = f"tortoize FAILED: {tail}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"ERROR: {e}"

    print("\n==================== verify entity-carry / tortoize ====================")
    print(f"refined : {args.refined}")
    print(f"pdb id  : {args.pdb_id}   model: {args.model_num}")
    print(f"baseline (lab-file prototype): {_PROTOTYPE_BASELINE}")
    print("------------------------------------------------------------------------")
    for name in ("bare", "minimal", "carried", "carried+reconc"):
        print(f"  {name:16s} {results.get(name, '(skipped)')}")
    print("========================================================================")
    print(
        "Gate: a 'carried' variant must show REAL z-scores. If only carried+reconc is REAL,\n"
        "set reconcile=True in production. If bare is also REAL, the harness can score bare\n"
        "single models and the carry is only for G1 / provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
