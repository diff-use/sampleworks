from __future__ import annotations

import argparse
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from atomworks import parse
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import save_structure
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import RewardFunction, setup_scattering_params
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.utils.guidance_constants import BOLTZ_1, BOLTZ_2, FK_STEERING, PURE_GUIDANCE
from sampleworks.utils.guidance_script_arguments import GuidanceConfig, JobResult


# The following imports aren't compatible with each other and are supported in separate
# hatch/pixi envs
try:
    from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper
except ImportError:
    Boltz1Wrapper = None
    logger.warning("Failed to import Boltz, hopefully you're running a different model")
try:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
except ImportError:
    ProtenixWrapper = None
    logger.warning("Failed to import Protenix, hopefully you're running a different model")
try:
    from sampleworks.models.rf3.wrapper import RF3Wrapper
except ImportError:
    RF3Wrapper = None
    logger.warning("Failed to import RF3, hopefully you're running a different model")
from sampleworks.utils.torch_utils import try_gpu


def save_trajectory(
    scaler_type: str,
    trajectory,
    atom_array,
    output_dir,
    reward_param_mask,
    subdir_name,
    save_every=10,
):
    if scaler_type == "pure_guidance":
        _save_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
        )
    elif scaler_type == "fk_steering":
        _save_fk_steering_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
        )
    else:  # we shouldn't ever get here, since we can't have run guidance w/o this!
        raise ValueError(f"Invalid scaler type: {scaler_type}")


def _save_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
):
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    if not isinstance(atom_array, AtomArray):
        raise TypeError(
            "Can only save a trajectory of type AtomArray or "
            f"AtomArrayStack, was given {type(atom_array)}"
        )

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[0]
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        array_copy.coord[:, reward_param_mask] = coords.detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def _save_fk_steering_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
):
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array[0]
    if not isinstance(atom_array, AtomArray):
        raise TypeError(
            "Can only save a trajectory of type AtomArray or "
            f"AtomArrayStack, was given {type(atom_array)}"
        )

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[1]  # first dim is the particle dim
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        # we save only the first ensemble out of n_particles, since saving
        # each particle at every step would clog trajectory saving
        array_copy.coord[:, reward_param_mask] = coords[0].detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


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
    device_str: str,
    model_checkpoint_path: str,
    model_type: str,
    method: str | None = None,
    model: Any = None,
) -> tuple[torch.device, Any]:
    device = torch.device(device_str) if device_str else try_gpu()
    logger.debug(f"Using device: {device}")
    if model_type == "protenix":
        logger.debug(f"Loading Protenix model from {model_checkpoint_path}")
        model_wrapper = ProtenixWrapper(  # pyright: ignore
            checkpoint_path=model_checkpoint_path, device=device, model=model
        )
    elif model_type == "boltz1":
        logger.debug(f"Loading Boltz1 model from {model_checkpoint_path}")
        model_wrapper = Boltz1Wrapper(  # pyright: ignore
            checkpoint_path=model_checkpoint_path,
            use_msa_server=True,
            device=device,
            model=model,
        )
    elif model_type == "boltz2":
        if method is None:
            # TODO: make a useful error msg that includes options for method
            raise ValueError("Method must be specified for Boltz2")
        logger.debug(f"Loading Boltz2 model from {model_checkpoint_path}")
        model_wrapper = Boltz2Wrapper(  #  pyright: ignore
            checkpoint_path=model_checkpoint_path,
            use_msa_server=True,
            device=device,
            method=method.upper(),
            model=model,
        )
    elif model_type == "rf3":
        if RF3Wrapper is None:
            raise ImportError("RF3 dependencies not installed")
        model_wrapper = RF3Wrapper(
            checkpoint_path=model_checkpoint_path,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # RF3 currently manages its own device; prefer that when available.
    device = getattr(model_wrapper, "device", device)

    # (pyright doesn't think Boltz1Wrapper etc are "Any")
    return (device, model_wrapper)  # pyright: ignore


# TODO: further atomize for easier testing.
def get_reward_function_and_structure(
    density: str | Path,
    device: torch.device,
    em,
    loss_order,
    resolution,
    structure_path: str | Path,
) -> tuple[RewardFunction, dict[str, Any]]:
    logger.debug(f"Loading structure from {structure_path}")
    structure = parse(
        structure_path,  # pyright: ignore  (doesn't like the type being passed)
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )

    logger.debug(f"Loading density map from {density}")
    xmap = XMap.fromfile(density, resolution=resolution)

    logger.debug("Setting up scattering parameters")
    scattering_params = setup_scattering_params(structure, em=em)  # pyright: ignore

    atom_array = structure["asym_unit"]  # pyright: ignore
    selection_mask = atom_array.occupancy > 0
    n_selected = selection_mask.sum()
    logger.info(f"Selected {n_selected} atoms with occupancy > 0")

    logger.info("Creating reward function")
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
    scaler_type: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results")
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
        logger.info(f"\nFinal loss: {valid_losses[-1]:.6f}")
        logger.info(f"Initial loss: {valid_losses[0]:.6f}")
        logger.info(f"Loss reduction: {valid_losses[0] - valid_losses[-1]:.6f}")

    logger.info(f"\nResults saved to {output_dir}/")


#####################
# Methods for running model guidance in separate processes, avoiding reloading of the model.
#####################
# TODO: these args are ultimately defined in run_grid_search.py, which is a terrible place
#  for that. Need to refactor this.
def run_guidance(
    args: GuidanceConfig | argparse.Namespace, guidance_type: str, model_wrapper, device
) -> JobResult:
    """
    Wrapper around the actual _run_guidance function, to redirect logs and generate a JobResult.
    Args:
        args:
        guidance_type:

    Returns:
    """

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # need to finish spawning separate logs for each worker
    handle = logger.add(
        args.log_path, level="INFO", filter=lambda rec: rec["extra"].get("special", False) is True
    )
    started_at = datetime.now()
    try:
        with logger.contextualize(special=True):
            _run_guidance(args, guidance_type, model_wrapper, device)
        logger.info("Guidance run successfully!")
        return get_job_result(args, device, started_at, datetime.now(), 0, "success")
    except Exception as e:
        logger.error(f"Error running guidance: {e}")
        logger.error(traceback.format_exc())
        return get_job_result(args, device, started_at, datetime.now(), 1, "failed")
    finally:
        logger.remove(handle)


# "guidance_type" is also called "scaler" in many places
def _run_guidance(
    args: GuidanceConfig | argparse.Namespace, guidance_type: str, model_wrapper, device
):
    reward_function, structure = get_reward_function_and_structure(
        args.density,  # str/path to a map file.
        device,  # this needs to come from the global context, not the args object.
        args.em,
        args.loss_order,
        args.resolution,
        args.structure,  # path/string to a structure file.
    )

    # Boltz was trained with this, others might not have been.
    use_alignment_for_reverse_diffusion = args.model in (BOLTZ_1, BOLTZ_2)

    if guidance_type == PURE_GUIDANCE:
        logger.info("Initializing pure guidance")
        guidance = PureGuidance(model_wrapper=model_wrapper, reward_function=reward_function)

        logger.info(f"Running pure guidance using model with hash {model_wrapper.model.__hash__()}")
        refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
            structure,
            msa_path=getattr(args, "msa_path", None),
            guidance_start=args.guidance_start,
            step_size=args.step_size,
            gradient_normalization=args.gradient_normalization,
            use_tweedie=args.use_tweedie,
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            partial_diffusion_step=args.partial_diffusion_step,
            out_dir=args.output_dir,
            alignment_reverse_diffusion=use_alignment_for_reverse_diffusion,
            ensemble_size=args.ensemble_size,
        )

    elif guidance_type == FK_STEERING:
        logger.info("Initializing Feynman-Kac steering")
        guidance = FKSteering(model_wrapper=model_wrapper, reward_function=reward_function)

        logger.info("Running FK steering")
        refined_structure, (traj_denoised, traj_next_step), losses = guidance.run_guidance(
            structure,
            msa_path=getattr(args, "msa_path", None),
            num_particles=args.num_particles,
            ensemble_size=args.ensemble_size,
            fk_resampling_interval=args.fk_resampling_interval,
            fk_lambda=args.fk_lambda,
            num_gd_steps=args.num_gd_steps,
            guidance_weight=args.guidance_weight,
            gradient_normalization=args.gradient_normalization,
            guidance_interval=args.guidance_interval,  # pyright: ignore (attr added after init intentionally)
            guidance_start=args.guidance_start,
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            partial_diffusion_step=args.partial_diffusion_step,
            out_dir=args.output_dir,
            alignment_reverse_diffusion=use_alignment_for_reverse_diffusion,
        )
    else:
        logger.error(f"Unknown guidance type: {guidance_type}")
        raise TypeError("Unknown guidance type!")

    save_everything(
        args.output_dir, losses, refined_structure, traj_denoised, traj_next_step, guidance_type
    )


def epoch_seconds(time_to_convert: datetime) -> float:
    return (time_to_convert - datetime(1970, 1, 1)).total_seconds()


def get_job_result(
    args: GuidanceConfig | argparse.Namespace,
    device: torch.device,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int,
    status: str,
) -> JobResult:
    start_time = epoch_seconds(started_at)
    end_time = epoch_seconds(ended_at)
    result = JobResult(
        protein=args.protein,
        model=args.model,
        method=args.method,
        scaler=args.guidance_type,
        ensemble_size=args.ensemble_size,
        gradient_weight=args.guidance_weight if hasattr(args, "guidance_weight") else -1.0,
        gd_steps=args.num_gd_steps if hasattr(args, "num_gd_steps") else -1,
        status=status,
        exit_code=exit_code,
        runtime_seconds=round(end_time - start_time, 2),
        started_at=started_at.isoformat(),
        finished_at=ended_at.isoformat(),
        log_path=args.log_path,
        output_dir=args.output_dir,
    )
    return result


def run_guidance_job_queue(job_queue_path: str) -> list[JobResult]:
    with open(job_queue_path, "rb") as fp:
        job_queue: list[GuidanceConfig] = pickle.load(fp)

    template_job = job_queue[0]
    if template_job.model_checkpoint is None:
        raise ValueError("Running guidance requires that you specify a model checkpoint, not None")

    logger.info(f"Running {len(job_queue)} jobs, using {template_job} as a setup template")
    device, model_wrapper = get_model_and_device(
        str(template_job.device),
        template_job.model_checkpoint,
        template_job.model,
        template_job.method,
    )
    job_results = []
    for i, job in enumerate(job_queue):
        logger.info(f"Running job {i + 1}/{len(job_queue)}: {job}")
        # The model wrapper can persist state across runs, so we need to re-initialize it each run.
        if job.model_checkpoint is None:
            raise ValueError(
                "Running guidance requires that you specify a model checkpoint, not None"
            )
        device, model_wrapper = get_model_and_device(
            str(device), job.model_checkpoint, job.model, job.method, model_wrapper.model
        )

        job_result = run_guidance(job, job.guidance_type, model_wrapper, device)
        job_results.append(job_result)

    return job_results
