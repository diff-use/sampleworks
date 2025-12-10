import time
from typing import Any

import numpy as np
from atomworks import parse
from loguru import logger
from pathlib import Path

import torch

from biotite.structure import AtomArray, stack
from biotite.structure.io import save_structure

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import RewardFunction, setup_scattering_params
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper
from sampleworks.models.protenix.wrapper import ProtenixWrapper
from sampleworks.utils.guidance_constants import FK_STEERING, PURE_GUIDANCE
from sampleworks.utils.torch_utils import try_gpu


def save_trajectory(
    scaler_type: str,
    trajectory,
    atom_array,
    output_dir,
    reward_param_mask,
    subdir_name,
    save_every=10
):
    if scaler_type == "pure_guidance":
        _save_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=save_every
        )
    elif scaler_type == "fk_steering":
        _save_fk_steering_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=save_every
        )
    else:  # we shouldn't ever get here, since we can't have run guidance w/o this!
        raise ValueError(f"Invalid scaler type: {scaler_type}")


def _save_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=10
):

    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: @marcus.collins swap try/except for if/else in next PR
    try:
        assert isinstance(atom_array, AtomArray)
    except AssertionError:
        atom_array = atom_array[0]

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[0]
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        array_copy.coord[:, reward_param_mask] = coords.detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def _save_fk_steering_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=10
):
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        assert isinstance(atom_array, AtomArray)
    except AssertionError:
        atom_array = atom_array[0]

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[1]  # first dim is the particle dim
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        # TODO: k.chripens can you add a comment here explaining why you take coords[0]?
        #   are we only saving the first particle?
        array_copy.coord[:, reward_param_mask] = coords[0].detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
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


def get_model_and_device(
        device: str,
        model_checkpoint_path: str,
        model_type: str,
        method: str | None = None,
) -> tuple[torch.device, ProtenixWrapper | Boltz1Wrapper | Boltz2Wrapper]:
    device = torch.device(device) if device else try_gpu()
    logger.debug(f"Using device: {device}")
    if model_type == "protenix":
        logger.debug(f"Loading Protenix model from {model_checkpoint_path}")
        model_wrapper = ProtenixWrapper(
            checkpoint_path=model_checkpoint_path,
            device=device,
        )
    elif model_type == "boltz1":
        print(f"Loading Boltz1 model from {model_checkpoint_path}")
        model_wrapper = Boltz1Wrapper(
            checkpoint_path=model_checkpoint_path,
            use_msa_server=True,
            device=device,
        )
    elif model_type == "boltz2":
        if method is None:
            # TODO: make a useful error msg that includes options for method
            raise ValueError("Method must be specified for Boltz2")
        print(f"Loading Boltz2 model from {model_checkpoint_path}")
        model_wrapper = Boltz2Wrapper(
            checkpoint_path=model_checkpoint_path,
            use_msa_server=True,
            device=device,
            method=method.upper(),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return device, model_wrapper


# TODO: further atomize for easier testing.
def get_reward_function_and_structure(
        density: str | Path,
        device: torch.device,
        em,
        loss_order,
        resolution,
        structure: str | Path
) -> tuple[RewardFunction, dict[str, Any]]:
    print(f"Loading structure from {structure}")
    structure = parse(
        structure,
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )

    print(f"Loading density map from {density}")
    xmap = XMap.fromfile(density, resolution=resolution)

    print("Setting up scattering parameters")
    scattering_params = setup_scattering_params(structure, em=em)

    atom_array = structure["asym_unit"]
    selection_mask = atom_array.occupancy > 0
    n_selected = selection_mask.sum()
    print(f"Selected {n_selected} atoms with occupancy > 0")

    print("Creating reward function")
    reward_function = RewardFunction(
        xmap,
        scattering_params,
        selection_mask,
        em=em,
        loss_order=loss_order,
        device=device,
    )
    return reward_function, structure


def save_everything(
        output_dir: str | Path,
        losses: list[Any],
        refined_structure: dict,
        traj_denoised: list[Any],
        traj_next_step: list[Any],
        scaler_type: str
        ) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    final_structure = CIFFile()
    set_structure(final_structure, refined_structure["asym_unit"])
    final_structure.write(str(output_dir / "refined.cif"))

    # Two calls to save_trajectory, very similar, but saving different trajectories!
    save_trajectory(
        scaler_type,
        traj_denoised,  # <--- the difference is here!
        refined_structure["asym_unit"],  # this is just used as a dummy structure
        output_dir,
        refined_structure["asym_unit"].occupancy > 0,
        "denoised",
        save_every=10,
    )
    save_trajectory(
        scaler_type,
        traj_next_step,  # <--- and here!
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


#####################
# Methods for running model guidance in separate processes, avoiding reloading of the model.
#####################
def setup_guidance_worker(args, model_name):
    global device, model_wrapper  # pylint: disable=global-statement

    # Might want to introduce a random delay here to avoid all workers starting at the same time.
    # this would allow the method below to find a free GPU and avoid a race?
    sleep_time = np.random.randint(0, 50)
    logger.info(f"Waiting {sleep_time} seconds before initializing model")
    time.sleep(sleep_time)

    # TODO better argument handling would help here
    # FIXME? UNCLEAR whether this will work given that we may be using Lightning/Fabric (for RF3)
    device, model_wrapper = get_model_and_device(
        device=args.device,
        model_checkpoint_path=args.model_checkpoint,
        model_type=model_name,
        method=args.method if model_name == "boltz2" else None
    )

    logger.info(
        f"Model {model_name} initialized with has {model_wrapper.__hash__()} on device: {device}"
    )
    return None


# "guidance_type" is also called "scaler" in many places
def run_guidance(args, guidance_type):
    global model_wrapper

    reward_function, structure = get_reward_function_and_structure(
        args.density,  # actually a str/path to a map file.
        args.device,
        args.em,
        args.loss_order,
        args.resolution,
        args.structure  # also actually a path or string
    )

    if guidance_type == PURE_GUIDANCE:
        logger.info("Initializing pure guidance")
        guidance = PureGuidance(model_wrapper=model_wrapper, reward_function=reward_function)

        logger.info("Running pure guidance using model with hash {model_wrapper.__hash__()}")
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

    elif guidance_type == FK_STEERING:
        logger.info("Initializing Feynman-Kac steering")
        guidance = FKSteering(model_wrapper=model_wrapper, reward_function=reward_function)

        logger.info("Running FK steering")
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
            alignment_reverse_diffusion=True,  # Boltz was trained with this
        )


    save_everything(
        args.output_dir,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        guidance_type
    )

