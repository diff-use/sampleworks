import argparse
from pathlib import Path

from loguru import logger

from sampleworks.eval.constants import OCCUPANCY_LEVELS
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.eval.grid_search_eval_utils import scan_grid_search_results, parse_args
from sampleworks.eval.structure_utils import get_reference_structure_coords, \
    get_reference_atomarraystack


def main(args: argparse.Namespace):
    workspace_root = Path(args.workspace_root)
    grid_search_dir = workspace_root / "grid_search_results"  # TODO make more general

    # Protein configurations: base map paths, structure selections, and resolutions
    protein_inputs_dir = args.grid_search_inputs_path or workspace_root
    protein_configs = ProteinConfig.from_csv(protein_inputs_dir, args.protein_configs_csv)

    logger.info(f"Grid search directory: {grid_search_dir}")
    logger.info(f"Proteins configured: {list(protein_configs.keys())}")

    # Scan for experiments (look for refined.cif files)
    all_experiments = scan_grid_search_results(grid_search_dir)
    logger.info(f"Found {len(all_experiments)} experiments with refined.cif files")

    if all_experiments:
        all_experiments.summarize()  # Prints some summary stats, e.g. number of unique proteins

    logger.info("Pre-loading reference structures for each protein for coordinate extraction")
    ref_coords = {}
    for protein_key, protein_config in protein_configs.items():
        # TODO: need to rework get_reference_structure so it returns coordinates separately
        #  for each occupancy, since we'll use them directly here.
        for occ in OCCUPANCY_LEVELS:
            protein_ref_coords = get_reference_atomarraystack(protein_config, occ)
            if protein_ref_coords is not None:
                ref_coords[(protein_key, occ)] = protein_ref_coords


if __name__ == "__main__":
    args = parse_args("Evaluate LDDT on grid search results.")
    main(args)
