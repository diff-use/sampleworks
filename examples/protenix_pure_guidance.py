"""
Run pure guidance with real-space density reward on the Protenix model.
"""

from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
    save_everything,
)
from sampleworks.utils.guidance_script_arguments import parse_protenix_fk_steering_args


def main(args):

    device, model_wrapper = get_model_and_device(
        args.device, args.model_checkpoint, model_type="protenix"
    )

    reward_function, structure = get_reward_function_and_structure(
        args.density,  # actually a str/path to a map file.
        args.device,
        args.em,
        args.loss_order,
        args.resolution,
        args.structure  # also actually a path or string
    )

    print("Initializing pure guidance")
    guidance = PureGuidance(
        model_wrapper=model_wrapper,
        reward_function=reward_function,
    )

    print("Running pure guidance")
    refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
        structure,
        guidance_start=args.guidance_start,
        step_size=args.step_size,
        gradient_normalization=args.gradient_normalization,
        use_tweedie=args.use_tweedie,
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        partial_diffusion_step=args.partial_diffusion_step,
        out_dir=args.output_dir,
        alignment_reverse_diffusion=False,  # Protenix was not trained with this
        ensemble_size=args.ensemble_size,
    )

    save_everything(
        args.output_dir,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        "pure_guidance"
    )


if __name__ == "__main__":
    args = parse_protenix_fk_steering_args()
    main(args)
