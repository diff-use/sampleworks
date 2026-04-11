"""Unified CLI for running diffusion guidance with sampleworks."""

from __future__ import annotations

import sys

from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance


def main(argv: list[str] | None = None) -> int:
    config = GuidanceConfig.from_cli(argv)
    device, model_wrapper = get_model_and_device(
        config.device,
        getattr(config, "model_checkpoint", None),
        config.model,
        method=getattr(config, "method", None),
    )
    result = run_guidance(config, config.guidance_type, model_wrapper, device)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
