"""
Run FK steering with real-space density reward on the Protenix model.
"""

from pathlib import Path

import torch
from atomworks.io.parser import parse
from biotite.structure import stack
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    XMap,
)
from sampleworks.core.rewards.real_space_density import (
    RewardFunction,
    setup_scattering_params,
)
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.models.protenix.wrapper import ProtenixWrapper
from sampleworks.utils.grid_search_utils import save_losses, save_fk_steering_trajectory
from sampleworks.utils.guidance_script_utils import parse_protenix_fk_steering_args
from sampleworks.utils.torch_utils import try_gpu


def main(args):

    device = torch.device(args.device) if args.device else try_gpu()
    print(f"Using device: {device}")

    print(f"Loading structure from {args.structure}")
    structure = parse(
        args.structure,
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )

    print(f"Loading density map from {args.density}")
    xmap = XMap.fromfile(args.density, resolution=args.resolution)

    print("Setting up scattering parameters")
    scattering_params = setup_scattering_params(structure, em=args.em)

    atom_array = structure["asym_unit"]
    selection_mask = atom_array.occupancy > 0
    n_selected = selection_mask.sum()
    print(f"Selected {n_selected} atoms with occupancy > 0")

    print("Creating reward function")
    reward_function = RewardFunction(
        xmap,
        scattering_params,
        selection_mask,
        em=args.em,
        loss_order=args.loss_order,
        device=device,
    )

    print(f"Loading Protenix model from {args.model_checkpoint}")
    model_wrapper = ProtenixWrapper(
        checkpoint_path=args.model_checkpoint,
        device=device,
    )

    print("Initializing Feynman-Kac steering")
    guidance = FKSteering(
        model_wrapper=model_wrapper,
        reward_function=reward_function,
    )

    print("Running FK steering")
    refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
        structure,
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
        alignment_reverse_diffusion=False,  # Protenix was not trained with this
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    final_structure = CIFFile()
    set_structure(final_structure, refined_structure["asym_unit"])
    final_structure.write(str(output_dir / "refined.cif"))

    save_fk_steering_trajectory(
        traj_denoised,
        refined_structure["asym_unit"],
        output_dir,
        refined_structure["asym_unit"].occupancy > 0,
        "denoised",
        save_every=10,
    )
    save_fk_steering_trajectory(
        traj_next_step,
        refined_structure["asym_unit"],
        output_dir,
        refined_structure["asym_unit"].occupancy > 0,
        "next_step",
        save_every=10,
    )
    save_losses(losses, output_dir)

    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        print(f"\nFinal loss: {valid_losses[-1]:.6f}")
        print(f"Initial loss: {valid_losses[0]:.6f}")
        print(f"Loss reduction: {valid_losses[0] - valid_losses[-1]:.6f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    args = parse_protenix_fk_steering_args()
    main(args)
