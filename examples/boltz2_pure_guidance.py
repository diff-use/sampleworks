"""
Run pure guidance with real-space density reward on the Boltz2 model.
"""

from sampleworks.utils.guidance_constants import PURE_GUIDANCE, BOLTZ_2
from sampleworks.utils.guidance_script_utils import run_guidance, get_model_and_device
from sampleworks.utils.guidance_script_arguments import parse_boltz2_pure_guidance_args


def main(args):

    # new (old?)-fangled way to get the model wrapper that makes it easier to work with process pools.
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, BOLTZ_2, method=args.method)
    run_guidance(args, PURE_GUIDANCE, device, model_wrapper)


if __name__ == "__main__":
    guidance_args = parse_boltz2_pure_guidance_args()
    main(guidance_args)
