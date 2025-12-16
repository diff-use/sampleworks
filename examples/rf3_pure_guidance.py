"""Run pure guidance with real-space density reward on the RF3 model."""

from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.utils.guidance_script_arguments import parse_rf3_pure_guidance_args
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
    save_everything,
)


def main(args):
    device, model_wrapper = get_model_and_device(
        device_str=args.device,
        model_checkpoint_path=args.model_checkpoint,
        model_type="rf3",
    )

    reward_function, structure = get_reward_function_and_structure(
        args.density,
        args.device,
        args.em,
        args.loss_order,
        args.resolution,
        args.structure,
    )

    print("Initializing pure guidance")
    guidance = PureGuidance(
        model_wrapper=model_wrapper,
        reward_function=reward_function,
    )

    print("Running pure guidance")
    refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
        structure,
        msa_path=args.msa_path,
        guidance_start=args.guidance_start,
        step_size=args.step_size,
        gradient_normalization=args.gradient_normalization,
        use_tweedie=args.use_tweedie,
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        partial_diffusion_step=args.partial_diffusion_step,
        out_dir=args.output_dir,
        alignment_reverse_diffusion=False,  # RF3 not trained with this
        ensemble_size=args.ensemble_size,
    )

    save_everything(
        args.output_dir,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        "pure_guidance",
    )


if __name__ == "__main__":
    main(parse_rf3_pure_guidance_args())
