from __future__ import annotations

import argparse


# TODO: this is used when resolving arguments from run_grid_search.py. We should get rid of it.
# It has to be here because it gets used in some otherwise circular imports.
def get_checkpoint(model: str, args: argparse.Namespace) -> str | None:
    if model == "boltz1":
        return args.boltz1_checkpoint
    elif model == "boltz2":
        return args.boltz2_checkpoint
    elif model == "protenix":
        return args.protenix_checkpoint
    return None
