import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sampleworks.utils.cif_utils import find_altloc_selections


def _process_row(
    row: pd.Series, altloc_label: str, min_span: int, include_all_altlocs: bool
) -> pd.Series:
    cif_file = row["structure"]
    selections = ";".join(
        find_altloc_selections(cif_file, altloc_label, min_span, include_all_altlocs)
    )
    if not selections:
        logger.warning(
            f"No selections emitted for {cif_file} "
            f"(min_span={min_span}, include_all_altlocs={include_all_altlocs}) "
            f"altlocs may exist but none met these criteria."
        )

    # The column names here are defined by the input requirements of scripts like
    # rscc_grid_search_script.py
    output = {
        "protein": row["name"],  # this should be the RSCB ID
        "selection": selections,
        "structure_pattern": Path(cif_file).name,
        "map_pattern": Path(row["density"]).name,
        "base_map_dir": Path(row["density"]).parent.name,
        "resolution": row["resolution"],
    }
    return pd.Series(output)


def main(args):
    """
    Write an output file with rows consisting of
    RCSB ID, selections for altlocs, and path to the corresponding CIF file,

    This script will likely change until we settle on a final input/output
    results directory structure.
    """
    input_df = pd.read_csv(args.input_csv)
    output = input_df.apply(
        _process_row,
        altloc_label=args.altloc_label,
        min_span=args.min_span,
        include_all_altlocs=args.include_all_altlocs,
        axis=1,
    )
    output.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to the input CSV config file used for grid search",
    )
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--min-span", type=int, default=5)
    parser.add_argument("--altloc-label", type=str, default="label_alt_id")
    parser.add_argument(
        "--no-all-altlocs",
        dest="include_all_altlocs",
        action="store_false",
        help="Omit the final per-chain selection string that includes all altloc residues",
    )
    args = parser.parse_args()
    main(args)
