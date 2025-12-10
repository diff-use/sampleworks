"""
Run pure guidance with real-space density reward on the Boltz2 model.
"""

from sampleworks.utils.guidance_constants import PURE_GUIDANCE, BOLTZ_2
from sampleworks.utils.guidance_script_utils import setup_guidance_worker, run_guidance
from sampleworks.utils.guidance_script_arguments import parse_boltz2_pure_guidance_args


def main(args):

    # new (old?)-fangled way to get the model wrapper that makes it easier to work with process pools.
    global device, model_wrapper
    setup_guidance_worker(args, BOLTZ_2)
    run_guidance(args, PURE_GUIDANCE)


if __name__ == "__main__":
    args = parse_boltz2_pure_guidance_args()
    main(args)
