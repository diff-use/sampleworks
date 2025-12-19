"""
Run pure guidance with real-space density reward on the Protenix model.
"""

from sampleworks.utils.guidance_constants import PROTENIX, PURE_GUIDANCE
from sampleworks.utils.guidance_script_arguments import parse_protenix_pure_guidance_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, PROTENIX)
    run_guidance(args, PURE_GUIDANCE, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_protenix_pure_guidance_args()
    main(guidance_args)
