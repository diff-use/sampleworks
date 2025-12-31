from __future__ import annotations

import argparse

from sampleworks.utils.guidance_constants import StructurePredictor


# TODO: this is used when resolving arguments from run_grid_search.py. We should get rid of it.
# It has to be here because it gets used in some otherwise circular imports.
def get_checkpoint(model: str, args: argparse.Namespace) -> str | None:
    if model == StructurePredictor.BOLTZ_1:
        return args.boltz1_checkpoint
    elif model == StructurePredictor.BOLTZ_2:
        return args.boltz2_checkpoint
    elif model == StructurePredictor.PROTENIX:
        return args.protenix_checkpoint
    elif model == StructurePredictor.RF3:
        return args.rf3_checkpoint
    return None
