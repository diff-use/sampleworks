import argparse
from pathlib import Path

from loguru import logger
from sampleworks.utils.cif_utils import find_altloc_selections


def main(args):
    """
    Write an output file with rows consisting of
    RCSB ID, selections for altlocs, and path to the corresponding CIF file,

    This script will likely change until we settle on a final input/output
    results directory structure.
    """
    output_lines = []
    warnings = []  # output these all at the end so they're easy to see.
    for line in open(args.input_csv):
        if not output_lines:
            keys = line.strip().split(",")
            if not all(k in keys for k in ("name", "structure", "resolution", "density")):
                raise ValueError(
                    "Input CSV file must keys 'name', 'structure', 'resolution', and 'density'"
                )
            output_lines.append(
                "protein,selection,structure_pattern,map_pattern,base_map_dir,resolution"
            )
            continue

        datum = dict(zip(keys, line.strip().split(",")))  # pyright: ignore

        cif_file = datum["structure"]
        structure_pattern = Path(cif_file).name
        pdb_id = datum["name"]
        resolution = datum["resolution"]
        density = Path(datum["density"])
        base_map_dir = density.parent.name  # we want the path relative to the input directory
        map_pattern = density.name

        selections = ";".join(find_altloc_selections(cif_file, args.altloc_label, args.min_span))
        if not selections:
            warnings.append(f"No altlocs found for {cif_file}")

        output_lines.append(
            f'{pdb_id},"{selections}",{structure_pattern},{map_pattern},{base_map_dir},{resolution}'
        )

    with open(args.output_file, "w") as f:
        f.write("\n".join(output_lines))

    for warning in warnings:
        logger.warning(warning)


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
    parser.add_argument("--cif-pattern", type=str, default="[0-9]*.cif")
    args = parser.parse_args()
    main(args)
