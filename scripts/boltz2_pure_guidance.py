"""
Run pure guidance with real-space density reward on the Boltz2 model.
"""

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import parse_boltz2_pure_guidance_args
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(args):
    device, model_wrapper = get_model_and_device(
        "", args.model_checkpoint, StructurePredictor.BOLTZ_2, method=args.method
    )
    run_guidance(args, GuidanceType.PURE_GUIDANCE, model_wrapper, device)


if __name__ == "__main__":
    guidance_args = parse_boltz2_pure_guidance_args()
    main(guidance_args)
