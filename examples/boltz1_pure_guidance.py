"""
Run pure guidance with real-space density reward on the Boltz1 model.
"""

from pathlib import Path

import torch
from atomworks.io.parser import parse
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    XMap,
)
from sampleworks.core.rewards.real_space_density import (
    RewardFunction,
    setup_scattering_params,
)
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.utils.grid_search_utils import save_trajectory, save_losses
from sampleworks.utils.guidance_script_utils import parse_boltz1_pure_guidance_args
from sampleworks.models.boltz.wrapper import Boltz1Wrapper
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

    print(f"Loading Boltz1 model from {args.model_checkpoint}")
    model_wrapper = Boltz1Wrapper(
        checkpoint_path=args.model_checkpoint,
        use_msa_server=True,
        device=device,
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
        alignment_reverse_diffusion=True,  # Boltz was trained with this
        ensemble_size=args.ensemble_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    final_structure = CIFFile()
    set_structure(final_structure, refined_structure["asym_unit"])
    final_structure.write(str(output_dir / "refined.cif"))

    save_trajectory(
        traj_denoised, atom_array, output_dir, selection_mask, "denoised", save_every=10
    )
    save_trajectory(
        traj_next_step,
        atom_array,
        output_dir,
        selection_mask,
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
    args = parse_boltz1_pure_guidance_args()
    main(args)
