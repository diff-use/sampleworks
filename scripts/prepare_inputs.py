"""Regenerate sampleworks input CIFs from the carve registry.

Reads a JSON carve registry (``pdb_id -> CarveSpec``) and, for each requested pdb id, fetches the
RCSB deposit, applies the declared carve, and writes the cached input CIF at
``<cache-dir>/<pdb_id>__<spec_hash>.cif`` -- the file the generation pipeline's ``structure``
column already expects. One ``<pdb_id>\t<path>`` line is printed per prepared input.

This is a decoupled prep step: it *produces* the input file; ``run_grid_search.py`` and
``ProteinInput.from_csv`` are unchanged. Point a proteins-CSV ``structure`` column at a printed
path -- use the absolute path, since ``from_csv`` resolves relative paths against the CSV's own
directory, not the working directory.

Run with sampleworks importable (e.g. ``PYTHONPATH=src`` or inside the pixi env). The RCSB deposit
is fetched from the network unless ``--rcsb-cache`` already holds ``<pdb_id>.cif``.

Examples
--------
    python scripts/prepare_inputs.py                       # every pdb id in the default registry
    python scripts/prepare_inputs.py 1vme 2yl0             # just these two
    python scripts/prepare_inputs.py --cache-dir /tmp/in --registry my_registry.json
"""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files
from pathlib import Path

from sampleworks.utils.input_prep import prepare_from_registry

_DEFAULT_REGISTRY = files("sampleworks.data") / "input_registry.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare sampleworks input CIFs from a carve registry."
    )
    parser.add_argument(
        "pdb_ids",
        nargs="*",
        help="PDB ids to prepare (default: every id in the registry).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=_DEFAULT_REGISTRY,
        help="Carve registry JSON keyed by pdb id (default: sampleworks/data/input_registry.json).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Output cache directory (default: ~/.sampleworks/inputs).",
    )
    parser.add_argument(
        "--rcsb-cache",
        type=Path,
        default=None,
        help="Directory of <pdb_id>.cif deposit originals, for offline reproduction.",
    )
    args = parser.parse_args(argv)

    extra = {} if args.cache_dir is None else {"cache_dir": args.cache_dir}
    try:
        prepared = prepare_from_registry(
            args.registry, args.pdb_ids or None, rcsb_cache=args.rcsb_cache, **extra
        )
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for pdb_id, path in prepared.items():
        print(f"{pdb_id}\t{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
