# utility script to put all header information from original PDB entry into our CIF files
import fnmatch
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.database.rcsb import fetch
from biotite.structure.io.pdbx import CIFColumn, CIFFile, set_structure
from loguru import logger
from sampleworks.utils.atom_array_utils import remove_atoms_with_any_nan_coords
from sampleworks.utils.cif_utils import add_category_to_cif

SAMPLEWORKS_CACHE = Path("~/.sampleworks/rcsb").expanduser()


def crawl_dir_by_depth(root_dir: str | Path, target_pattern: str, n_levels: int) -> list[Path]:
    root = Path(root_dir)
    if n_levels < 0:
        return []

    results: list[Path] = []

    def _crawl(current: Path, levels_left: int) -> None:
        try:
            for entry in current.iterdir():
                if entry.is_file():
                    if fnmatch.fnmatch(entry.name, target_pattern):
                        logger.info(f"Found matching file: {entry}")
                        results.append(entry)
                elif entry.is_dir() and levels_left > 0:
                    _crawl(entry, levels_left - 1)
        except (PermissionError, FileNotFoundError):
            return

    _crawl(root, n_levels)
    return results


#  FIXED: added help descriptions + removed duplicates
def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing CIF files"
    )

    parser.add_argument(
        "--cif-pattern",
        default="refined.cif",
        help="Pattern for CIF files (default: refined.cif)"
    )

    parser.add_argument(
        "--rcsb-pattern",
        default="grid_search_results/(.{4})",
        help="Regex pattern to extract RCSB ID (must have one group)"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth to search directory tree"
    )

    parser.add_argument(
        "--grid-search-input-dir",
        required=True,
        help="Directory containing input PDB files"
    )

    parser.add_argument(
        "--input-pdb-pattern",
        default="{pdb_id}/{pdb_id}_single_001_density_input.cif",
        help="Pattern for input PDB files"
    )

    return parser.parse_args()


def main(input_dir, grid_search_input_dir, target_pattern, rcsb_regex, depth, input_pdb_pattern):
    SAMPLEWORKS_CACHE.mkdir(parents=True, exist_ok=True)

    cif_files_to_patch = crawl_dir_by_depth(input_dir, target_pattern, n_levels=depth)

    results = joblib.Parallel()(
        joblib.delayed(patch_individual_cif_file)(
            f, rcsb_regex, Path(grid_search_input_dir), input_pdb_pattern
        )
        for f in cif_files_to_patch
    )

    results = [r for r in results if r]
    if results:
        logger.error("The following errors occurred:")
        for r in results:
            print(r)


def patch_individual_cif_file(cif_file, rcsb_regex, reference_dir, input_pdb_pattern):
    cif_path = Path(cif_file)
    m = re.search(rcsb_regex, str(cif_path))
    rcsb_id = m.group(1) if m else None

    if not m:
        msg = f"Unable to parse RCSB id from {cif_file}"
        logger.warning(msg)
        return msg

    try:
        reference_path = reference_dir / input_pdb_pattern.format(pdb_id=rcsb_id)
        rcsb_path = fetch(rcsb_id, format="cif", target_path=str(SAMPLEWORKS_CACHE))

        reference = load_any(reference_path)
        asym_unit = ensure_atom_array_stack(load_any(cif_file))

    #  FIXED: better error handling
    except Exception as e:
        logger.exception(f"Error reading {cif_file} or {reference_path}: {e}")
        return str(e)

    if reference.res_id is None or asym_unit.res_id is None:
        return "Missing residue numbers"

    ref_keys = list(dict.fromkeys(zip(reference.chain_id.tolist(), reference.res_id.tolist())))
    cif_keys = list(dict.fromkeys(zip(asym_unit.chain_id.tolist(), asym_unit.res_id.tolist())))

    if len(ref_keys) != len(cif_keys):
        return "Residue mapping mismatch"

    mapping = {}

    # FIXED: chain consistency check
    for cif_key, ref_key in zip(cif_keys, ref_keys):
        if cif_key[0] != ref_key[0]:
            return "Chain mismatch in residue mapping"
        mapping[cif_key] = ref_key[1]

    atom_keys = list(zip(asym_unit.chain_id.tolist(), asym_unit.res_id.tolist()))

    #  FIXED: preserve dtype
    asym_unit.res_id = np.array(
        [mapping[k] for k in atom_keys],
        dtype=asym_unit.res_id.dtype
    )

    template = CIFFile.read(rcsb_path)
    set_structure(template, asym_unit)

   
    n_atoms = len(template.block["atom_site"]["type_symbol"])
    template.block["atom_site"]["id"] = CIFColumn(np.arange(n_atoms))

    
    # OCCUPANCY FIX (SAFE)
   

    if "occupancy" not in template.block["atom_site"]:
        template.block["atom_site"]["occupancy"] = CIFColumn([1.0] * n_atoms)

    occupancy_values = template.block["atom_site"]["occupancy"]

    for val in occupancy_values:
        try:
            value = float(val)
        except (ValueError, TypeError):
            error = f"Invalid occupancy value '{val}' in {cif_path}"
            logger.error(error)
            return error

        if not (0.0 <= value <= 1.0):
            error = f"Invalid occupancy value {value} in {cif_path} (must be 0–1)"
            logger.error(error)
            return error

   
    # B_iso FIX


    if "B_iso_or_equiv" not in template.block["atom_site"]:
        template.block["atom_site"]["B_iso_or_equiv"] = CIFColumn([20.0] * n_atoms)

    template.block.name = cif_path.stem
    output_file = cif_path.parent / f"{cif_path.stem}-patched.cif"
    template.write(output_file)

    logger.info(f"Wrote {output_file}")
    return None


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_dir,
        args.grid_search_input_dir,
        args.cif_pattern,
        args.rcsb_pattern,
        args.depth,
        args.input_pdb_pattern,
    )