# utility script to put all header information from original PDB entry into our CIF files
import fnmatch
import joblib
import re
from argparse import ArgumentParser
from pathlib import Path
from shutil import copy

import numpy as np
from atomworks.io.parser import parse
from biotite.structure import AtomArrayStack
from biotite.structure.io.pdbx import CIFFile, set_structure, CIFColumn
from biotite.database.rcsb import fetch

SAMPLEWORKS_CACHE = Path("~/.sampleworks/rcsb").expanduser()
SAMPLEWORKS_CACHE.mkdir(parents=True, exist_ok=True)

def crawl_dir_by_depth(
        root_dir: str | Path,
        target_pattern: str,
        n_levels: int,
) -> list[Path]:
    """
    Recursively crawl `root_dir` up to `n_levels` directory levels deep and return
    all files whose *name* matches `target_pattern` (fnmatch-style, e.g. "*.cif").

    Depth meaning:
      - n_levels = 0: only files directly in root_dir
      - n_levels = 1: root_dir + its immediate subdirectories
      - etc.
    """
    root = Path(root_dir)
    if n_levels < 0:
        return []

    results: list[Path] = []

    def _crawl(current: Path, levels_left: int) -> None:
        try:
            for entry in current.iterdir():
                if entry.is_file():
                    if fnmatch.fnmatch(entry.name, target_pattern):
                        results.append(entry)
                elif entry.is_dir() and levels_left > 0:
                    _crawl(entry, levels_left - 1)
        except (PermissionError, FileNotFoundError):
            # Skip unreadable or transient directories
            return

    _crawl(root, n_levels)
    return results


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--cif-pattern", default='refined.cif')
    parser.add_argument(
        "--rcsb-pattern", default='grid_search_results/(.{4})',
        help="Regex pattern for rcsb ids in file paths. "
             "Must have only one group, surrounding the id"
    )
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth to search the directory tree below input-dir")
    args = parser.parse_args()
    return args

def main(
        input_dir: str | Path,
        target_pattern: str,
        rcsb_regex: str = r"grid_search_results/(.{4})",
        depth: int = 4
) -> None:
    cif_files_to_patch = crawl_dir_by_depth(input_dir, target_pattern, n_levels=depth)
    results = joblib.Parallel()(
        joblib.delayed(patch_individual_cif_file)(f, rcsb_regex) for f in cif_files_to_patch
    )
    results = [r for r in results if r]
    if results:
        print("The following files could not be patched:")
        for r in results:
            print(r)


def patch_individual_cif_file(cif_file: Path, rcsb_regex: str):
    cif_path = Path(cif_file)
    # write a backup version of the input cif file
    copy(cif_path, cif_path.parent / (cif_path.name + ".bak"))

    m = re.search(rcsb_regex, str(cif_path))
    rcsb_id = m.group(1) if m else None
    if not m:
        print(
            f"Unable to parse an RCSB structure id: from path {cif_file} with pattern {rcsb_regex}"
        )
        return cif_file

    # fetch only downloads the file if it isn't already present.
    rcsb_path = fetch(rcsb_id, format="cif", target_path=str(SAMPLEWORKS_CACHE))

    # load the copy, and the new coordinates for it.
    template = CIFFile.read(rcsb_path)
    cif_to_patch = parse(cif_file)
    asym_unit: AtomArrayStack = cif_to_patch["asym_unit"]

    # make sure entity ids match in atom_site and entity_poly
    ep = template.block["entity_poly"]
    # fixme for now I'm using a hack--if there's one polymer entity, just assert that
    #   polymers in atom_site have to be that one. Otherwise do nothing.
    if len(ep["entity_id"]) == 1:
        entity_id = ep["entity_id"].as_item()
        if not "label_entity_id" in asym_unit.get_annotation_categories():
            asym_unit.add_annotation("label_entity_id", int)
        asym_unit.label_entity_id = np.ones_like(asym_unit.label_entity_id) * int(entity_id)

    # now set the structure with correct entity ids
    set_structure(template, asym_unit)

    # If there's a pdbx_poly_seq_scheme, make sure the seq nums all agree, as
    # the numbers in our outputs will all agree. We appear to use the one called ndb_seq_num
    nsm = template.block["pdbx_poly_seq_scheme"]["ndb_seq_num"]
    template.block["pdbx_poly_seq_scheme"]["pdb_seq_num"] = nsm
    template.block["pdbx_poly_seq_scheme"]["auth_seq_num"] = nsm

    # Make sure the id field is unique to each atom
    template.block["atom_site"]["id"] = CIFColumn(np.arange(np.prod(asym_unit.shape)))

    template.block.name = cif_path.stem
    template.write(cif_file)
    print(f"Wrote {cif_file}")
    return None


if __name__ == "__main__":

    args = parse_args()
    main(args.input_dir, args.cif_pattern, args.rcsb_pattern, args.depth)