"""
Run FK steering with real-space density reward on the RF3 model.
"""

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import parse_rf3_fk_steering_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device("", args.model_checkpoint, StructurePredictor.RF3)
    run_guidance(args, GuidanceType.FK_STEERING, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_rf3_fk_steering_args()
    main(guidance_args)
