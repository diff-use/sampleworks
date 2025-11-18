#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import csv

import iotbx.pdb
import mmtbx.model
import mmtbx.f_model
from iotbx import reflection_file_reader
import libtbx.load_env 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute R_work and R_free from a PDB + MTZ and append to pdb_rvalues.csv."
    )
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("mtz", help="Input MTZ file")
    parser.add_argument(
        "--free-flag-value",
        type=int,
        default=0,
        help="Value in FREE array that indicates the free set (default: 0)",
    )
    return parser.parse_args()


def find_arrays(miller_arrays):
    """Pick out F_obs and R-free flags from the MTZ."""
    f_obs = None
    r_free_flags = None

    for ma in miller_arrays:
        label = ma.info().label_string()
        print("  ", label)

        # Try common Fobs labels
        if label in ("FOBS_X,SIGFOBS_X", "FP,SIGFP"):
            f_obs = ma

        # Try common R-free labels
        if label in ("R-free-flags", "FREE"):
            r_free_flags = ma

    return f_obs, r_free_flags


def main():
    args = parse_args()
    pdb_file = args.pdb
    mtz_file = args.mtz
    csv_file = args.csv
    free_flag_value = args.free_flag_value

    # Basic sanity checks
    if not os.path.isfile(pdb_file):
        sys.exit(f"ERROR: PDB file not found: {pdb_file}")
    if not os.path.isfile(mtz_file):
        sys.exit(f"ERROR: MTZ file not found: {mtz_file}")

    print(f"PDB file: {pdb_file}")
    print(f"MTZ file: {mtz_file}")

    # Read PDB and build model
    pdb_inp = iotbx.pdb.input(file_name=pdb_file)
    model = mmtbx.model.manager(model_input=pdb_inp)

    # Read MTZ -> Miller arrays
    hkl_inp = reflection_file_reader.any_reflection_file(file_name=mtz_file)
    miller_arrays = hkl_inp.as_miller_arrays()

    f_obs, r_free_flags = find_arrays(miller_arrays)

    if f_obs is None:
        sys.exit(
            "ERROR: Could not find F_obs array in MTZ. "
            "Expected one of: 'FOBS_X,SIGFOBS_X' or 'FP,SIGFP'."
        )
    if r_free_flags is None:
        sys.exit(
            "ERROR: Could not find R-free flags array in MTZ. "
            "Expected one of: 'R-free-flags' or 'FREE'."
        )

    # Put on common set of Miller indices
    f_obs, r_free_flags = f_obs.common_sets(r_free_flags)

    # Convert to boolean mask: True = free-set reflections
    data = r_free_flags.data()
    free_mask = (data == free_flag_value)
    r_free_flags = r_free_flags.array(data=free_mask)

    # Build f_model manager and compute R-values
    fmodel = mmtbx.f_model.manager(
        f_obs=f_obs,
        r_free_flags=r_free_flags,
        xray_structure=model.get_xray_structure(),
    )
    fmodel.update_all_scales()

    r_work = fmodel.r_work()
    r_free = fmodel.r_free()


    csv_exists = os.path.isfile({args.pdb}_rvalues.csv)
    with open(csv_file, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not csv_exists:
            writer.writerow(["pdb", "mtz", "r_work", "r_free"])
        writer.writerow(
            [
                os.path.basename(pdb_file),
                os.path.basename(mtz_file),
                f"{r_work:.4f}",
                f"{r_free:.4f}",
            ]
        )


if __name__ == "__main__":
    main()
