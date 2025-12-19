"""
Run pure guidance with real-space density reward on the RF3 model.
"""

from sampleworks.utils.guidance_constants import PURE_GUIDANCE, RF3
from sampleworks.utils.guidance_script_arguments import parse_rf3_pure_guidance_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, RF3)
    run_guidance(args, PURE_GUIDANCE, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_rf3_pure_guidance_args()
    main(guidance_args)
