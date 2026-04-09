"""Run FK steering with real-space density reward on the Boltz1 model."""

import sys

from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main():
    config = GuidanceConfig.from_cli(model="boltz1", guidance_type="fk_steering")
    device, model_wrapper = get_model_and_device(
        config.device,
        getattr(config, "model_checkpoint", None),
        config.model,
    )
    result = run_guidance(config, config.guidance_type, model_wrapper, device)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
