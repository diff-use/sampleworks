"""Run FK steering with real-space density reward on the RF3 model."""

from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.utils.guidance_script_arguments import parse_rf3_fk_steering_args
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

    print("Initializing Feynman-Kac steering")
    guidance = FKSteering(
        model_wrapper=model_wrapper,
        reward_function=reward_function,
    )

    print("Running FK steering")
    refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
        structure,
        msa_path=args.msa_path,
        num_particles=args.num_particles,
        ensemble_size=args.ensemble_size,
        fk_resampling_interval=args.fk_resampling_interval,
        fk_lambda=args.fk_lambda,
        num_gd_steps=args.num_gd_steps,
        guidance_weight=args.guidance_weight,
        gradient_normalization=args.gradient_normalization,
        guidance_interval=args.guidance_interval,
        guidance_start=args.guidance_start,
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        partial_diffusion_step=args.partial_diffusion_step,
        out_dir=args.output_dir,
        alignment_reverse_diffusion=False,  # RF3 not trained with this
    )

    save_everything(
        args.output_dir,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        "fk_steering",
    )


if __name__ == "__main__":
    main(parse_rf3_fk_steering_args())
