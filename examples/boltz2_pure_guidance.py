"""
Run pure guidance with real-space density reward on the Boltz2 model.
"""

import argparse
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
from sampleworks.models.boltz.wrapper import Boltz2Wrapper
from sampleworks.utils.setup import try_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with Boltz-2 and real-space density"
    )
    parser.add_argument("--structure", type=str, required=True, help="Input structure")
    parser.add_argument("--density", type=str, required=True, help="Input density map")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--partial-diffusion-step",
        type=int,
        default=0,
        help="Diffusion step to start from",
    )
    parser.add_argument("--step-size", type=float, default=0.1, help="Gradient step")
    parser.add_argument(
        "--loss-order", type=int, default=2, choices=[1, 2], help="L1 or L2 loss"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Map resolution in Angstroms (required for CCP4/MRC/MAP)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="~/.boltz/boltz2_conf.ckpt",
        help="Path to Boltz2 checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)"
    )
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
    parser.add_argument("--em", action="store_true", help="Use EM scattering factors")
    parser.add_argument(
        "--guidance-start",
        type=int,
        default=-1,
        help="Step to start guidance (default: -1, starts immediately)",
    )
    parser.add_argument(
        "--use-tweedie",
        action="store_true",
        help="Use Tweedie's formula for gradient computation "
        "(enables augmentation/alignment)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--align-to-input",
        action="store_true",
        help="Enable alignment to input",
    )
    return parser.parse_args()


def save_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, save_every=10
):
    from biotite.structure.io import save_structure

    output_dir = Path(output_dir / "trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(trajectory):
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy.coord[:, reward_param_mask] = coords.numpy()
        save_structure(output_dir / f"trajectory_{i}.cif", array_copy)


def save_losses(losses, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "losses.txt", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(losses):
            if loss is not None:
                f.write(f"{i},{loss}\n")
            else:
                f.write(f"{i},NA\n")


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else try_gpu()
    print(f"Using device: {device}")

    print(f"Loading structure from {args.structure}")
    structure = parse(args.structure, hydrogen_policy="remove", add_missing_atoms=False)

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

    print(f"Loading Boltz2 model from {args.model_checkpoint}")
    model_wrapper = Boltz2Wrapper(
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
    refined_structure, trajectory, losses = guidance.run_guidance(
        structure,
        guidance_start=args.guidance_start,
        step_size=args.step_size,
        gradient_normalization=args.gradient_normalization,
        use_tweedie=args.use_tweedie,
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        partial_diffusion_step=args.partial_diffusion_step,
        out_dir=args.output_dir,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    final_structure = CIFFile()
    set_structure(final_structure, refined_structure["asym_unit"])
    final_structure.write(str(output_dir / "refined.cif"))

    save_trajectory(trajectory, atom_array, output_dir, selection_mask, save_every=5)
    save_losses(losses, output_dir)

    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        print(f"\nFinal loss: {valid_losses[-1]:.6f}")
        print(f"Initial loss: {valid_losses[0]:.6f}")
        print(f"Loss reduction: {valid_losses[0] - valid_losses[-1]:.6f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
