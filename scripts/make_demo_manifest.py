#!/usr/bin/env python
"""Write a truncated copy of a Sampleworks ``proteins.csv`` manifest.

This exists purely so that demo/smoke presets can run the real pipeline over a
single protein without anyone having to hand-maintain a second dataset. It is
stdlib-only on purpose: it runs as a preset ``pre_job`` and must import cleanly
in whichever pixi environment the demo job happens to use.

Example
-------
::

    python scripts/make_demo_manifest.py \\
        --proteins /data/inputs/proteins.csv \\
        --out /data/results/demo/_demo_inputs/proteins.csv \\
        --limit 1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def subset_manifest(source: Path, destination: Path, limit: int) -> int:
    """Copy the header and the first ``limit`` data rows of ``source``.

    Parameters
    ----------
    source : pathlib.Path
        Existing manifest CSV with a header row.
    destination : pathlib.Path
        Path to write the truncated manifest to. Parent directories are
        created if they do not exist.
    limit : int
        Maximum number of data rows (excluding the header) to keep. Must be
        positive.

    Returns
    -------
    int
        Number of data rows actually written, which is ``min(limit, available)``.

    Raises
    ------
    FileNotFoundError
        If ``source`` does not exist.
    ValueError
        If ``limit`` is not positive, or if ``source`` has no header row.
    """
    if limit <= 0:
        raise ValueError(f"--limit must be positive, got {limit}")
    if not source.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source}")

    with open(source, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Source manifest is empty: {source}") from exc
        rows = [row for _, row in zip(range(limit), reader)]

    if not rows:
        raise ValueError(f"Source manifest has a header but no data rows: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed ``proteins``, ``out``, and ``limit`` values.
    """
    parser = argparse.ArgumentParser(
        description="Truncate a proteins.csv manifest to its first N rows for demo runs."
    )
    parser.add_argument("--proteins", required=True, help="Source proteins.csv to subset")
    parser.add_argument("--out", required=True, help="Path to write the truncated manifest to")
    parser.add_argument(
        "--limit", type=int, default=1, help="Number of data rows to keep (default: 1)"
    )
    return parser.parse_args()


def main() -> int:
    """Write the truncated manifest and report where it landed.

    Returns
    -------
    int
        Process exit code; 0 on success, 2 on a user-facing input error.
    """
    args = parse_args()
    source = Path(args.proteins)
    destination = Path(args.out)
    try:
        written = subset_manifest(source, destination, args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"make_demo_manifest: {exc}", file=sys.stderr)
        return 2
    print(f"make_demo_manifest: wrote {written} row(s) from {source} to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
