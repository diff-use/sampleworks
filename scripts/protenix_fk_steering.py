"""
Run FK steering with real-space density reward on the Protenix model.
"""

from sampleworks.utils.guidance_constants import FK_STEERING, PROTENIX
from sampleworks.utils.guidance_script_arguments import parse_protenix_fk_steering_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, PROTENIX)
    run_guidance(args, FK_STEERING, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_protenix_fk_steering_args()
    main(guidance_args)
