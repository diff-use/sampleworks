import argparse
import json
import subprocess
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sampleworks.eval.eval_dataclasses import Experiment
from sampleworks.eval.grid_search_eval_utils import scan_grid_search_results


def parse_args(description: str | None = None) -> argparse.Namespace:
    """
    Return a common set of arguments for grid search evaluation scripts,
    with a custom description, which is passed to argparse.ArgumentParser.

    All eval scripts should use this same framework
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        required=True,
        help="Path containing the grid search results directory, e.g. if results are "
        "at $HOME/grid_search_results, $HOME should be what you pass",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Number of parallel jobs to run. -1 uses all CPUs.",
        default=16,
    )
    return parser.parse_args()

def main(args) -> None:
    # TODO check that phenix is installed and commands are available, bail early if not.
    workspace_root = Path(args.workspace_root)
    grid_search_dir = workspace_root / "grid_search_results"  # TODO make more general
    all_experiments = scan_grid_search_results(grid_search_dir)
    logger.info(f"Found {len(all_experiments)} experiments with refined.cif files")

    # Now loop over experiments with joblib and get back tuples of experiment level metrics
    clashscore_metrics = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(process_one_experiment)(experiment)
        for experiment in all_experiments
    )
    clashscore_df = pd.concat(clashscore_metrics)  # pyright: ignore
    clashscore_df.to_csv(
        workspace_root / "grid_search_results"/ "clashscore_metrics.csv", index=False
    )


def process_one_experiment(experiment: Experiment) -> pd.DataFrame:
    # make sure there are no nan lines in the CIF file  # TODO fix this elsewhere!
    file_with_no_nans = experiment.refined_cif_path.parent / "nonan.cif"
    json_output = experiment.refined_cif_path.parent / "clashscore.json"
    logfile = experiment.refined_cif_path.parent / "clashscore.log"
    logger.info(f"Removing nans from {experiment.refined_cif_path}")
    retcode = subprocess.call(
        f"grep -v nan {experiment.refined_cif_path}".split(),
        stdout=file_with_no_nans.open("w")
    )
    if retcode != 0:
        raise RuntimeError(f"grep failed, see {logfile} for details")

    # phenix needs to be installed and on path for this to work. Also sh won't work with
    # phenix.clashscore because of that pesky period in the name.
    retcode = subprocess.call(
        f"phenix.clashscore {str(file_with_no_nans)} --json-filename {str(json_output)}".split(),
        stderr=logfile.open("w")
    )
    if retcode != 0:
        logger.error(f"phenix.clashscore failed, see {logfile} for details")
        return pd.DataFrame()
    return process_clashscore_json_output(json_output)


def process_clashscore_json_output(json_output: Path) -> pd.DataFrame:
    """
    Opens the json output file `json_output` and parses out the
    "summary_results", flattening it into rows which include the "model_name" field

    """
    with open(json_output) as f:
        json_data = json.load(f)

    model_name = json_data.get("model_name")
    summary_results = json_data.get("summary_results", {})

    rows = []
    for model_id, results in summary_results.items():
        row = {
            "model_name": model_name,
            "model_id": model_id,
            "clashscore": results.get("clashscore"),
            "num_clashes": results.get("num_clashes")
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    args = parse_args()
    main(args)
