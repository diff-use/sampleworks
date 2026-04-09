"""Run pure guidance with real-space density reward on the Protenix model."""

from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main():
    config = GuidanceConfig.from_cli(model="protenix", guidance_type="pure_guidance")
    device, model_wrapper = get_model_and_device(
        config.device,
        getattr(config, "model_checkpoint", None),
        config.model,
    )
    run_guidance(config, config.guidance_type, model_wrapper, device)


if __name__ == "__main__":
    main()
