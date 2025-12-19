"""
Run pure guidance with real-space density reward on the Boltz1 model.
"""

from sampleworks.utils.guidance_constants import BOLTZ_1, PURE_GUIDANCE
from sampleworks.utils.guidance_script_arguments import parse_boltz1_pure_guidance_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, BOLTZ_1)
    run_guidance(args, PURE_GUIDANCE, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_boltz1_pure_guidance_args()
    main(guidance_args)
