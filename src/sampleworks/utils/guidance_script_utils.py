from __future__ import annotations

import argparse
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks import parse
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import save_structure
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import RewardFunction, setup_scattering_params
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.eval.structure_utils import get_asym_unit_from_structure
from sampleworks.utils.guidance_constants import (
    GuidanceType,
    StructurePredictor,
)
from sampleworks.utils.guidance_script_arguments import GuidanceConfig, JobResult
from sampleworks.utils.msa import MSAManager


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
    """
    Save a trajectory to disk using the format appropriate for the provided scaler type.
    
    Parameters:
        scaler_type (GuidanceType): The guidance/scaler type determining the trajectory format and saving behavior.
        trajectory: Sequence or array of trajectory coordinate frames produced by the guidance process.
        atom_array: An AtomArray or AtomArrayStack describing the base structure used to create per-step ensembles.
        output_dir (str | Path): Directory where trajectory CIF files will be written.
        reward_param_mask: Boolean mask or index array selecting atoms whose coordinates are present in each trajectory frame.
        subdir_name (str): Name of the subdirectory under `output_dir/trajectory` where files will be saved.
        save_every (int): Interval of steps between saved frames (save one file every `save_every` steps).
    
    Raises:
        ValueError: If `scaler_type` is not a recognized GuidanceType.
    """
    if scaler_type == GuidanceType.PURE_GUIDANCE:
        _save_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
        )
    elif scaler_type == GuidanceType.FK_STEERING:
        _save_fk_steering_trajectory(
            trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
        )
    else:  # we shouldn't ever get here, since we can't have run guidance w/o this!
        raise ValueError(f"Invalid scaler type: {scaler_type}")


def _assign_coords_to_array(
    array_copy: AtomArrayStack,
    coords: np.ndarray,
    reward_param_mask: np.ndarray,
) -> None:
    """
    Assign trajectory coordinates into an AtomArrayStack, writing either to the full atom set or to a masked subset.
    
    Parameters:
        array_copy (AtomArrayStack): Target atom array stack whose `.coord` will be updated.
        coords (np.ndarray): Coordinates to assign; expected shape ends with (n_atoms, 3) or (ensemble, n_atoms, 3).
        reward_param_mask (np.ndarray): Boolean mask selecting the subset of atoms corresponding to `coords` when `coords` does not cover the full array.
    
    Raises:
        ValueError: If the number of atoms in `coords` matches neither the full atom count of `array_copy` nor the number selected by `reward_param_mask`.
    """
    n_atoms_array = array_copy.coord.shape[-2]  # pyright: ignore[reportOptionalMemberAccess]
    n_atoms_coords = coords.shape[-2]

    if n_atoms_coords == n_atoms_array:
        array_copy.coord = coords
    elif n_atoms_coords == int(reward_param_mask.sum()):
        array_copy.coord[:, reward_param_mask] = coords  # pyright: ignore[reportOptionalSubscript]
    else:
        raise ValueError(
            f"Trajectory coords ({n_atoms_coords} atoms) match neither "
            f"the full atom array ({n_atoms_array}) nor the masked subset "
            f"({int(reward_param_mask.sum())})"
        )


def _save_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
):
    """
    Save selected frames of a trajectory as CIF ensemble files under output_dir/trajectory/<subdir_name>.
    
    Parameters:
        trajectory: Iterable of coordinate arrays/tensors, each with shape (ensemble_size, num_coordinates_per_structure).
        atom_array: An AtomArray or AtomArrayStack used as a template for saved ensembles; if an AtomArrayStack is provided its first element is used.
        output_dir: Path or path-like base directory where the "trajectory/<subdir_name>" folder will be created.
        reward_param_mask: Boolean mask or index array selecting which atoms' coordinates in the template should be replaced from each trajectory frame.
        subdir_name (str): Name of the subdirectory under "trajectory" where CIF files will be written.
        save_every (int): Interval of frames to save (only frames where index % save_every == 0 are written).
    
    Raises:
        TypeError: If atom_array is not an AtomArray or AtomArrayStack.
    """
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
        _assign_coords_to_array(array_copy, coords.detach().numpy(), reward_param_mask)
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def _save_fk_steering_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every
):
    """
    Save FK-steering trajectory frames as CIF files using the first ensemble member for each saved step.
    
    Parameters:
        trajectory: Iterable of coordinate tensors/arrays where each element contains coordinates for all ensemble members; the function uses the first ensemble member (coords[0]) when writing a frame.
        atom_array: An AtomArray or AtomArrayStack providing the reference structure; if an AtomArrayStack is provided the first entry is used and duplicated to match the ensemble size.
        output_dir: Path-like base directory under which files are written to <output_dir>/trajectory/<subdir_name>.
        reward_param_mask: Boolean mask selecting which atom coordinates are updated from the trajectory coordinates.
        subdir_name: Subdirectory name under "trajectory" where CIF files will be saved.
        save_every: Integer step interval; only steps whose index is divisible by this value are written to disk.
    
    Raises:
        TypeError: If atom_array is not an AtomArray or AtomArrayStack.
    """
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
        _assign_coords_to_array(array_copy, coords[0].detach().numpy(), reward_param_mask)
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
    if model_type == StructurePredictor.PROTENIX:
        logger.debug(f"Loading Protenix model from {model_checkpoint_path}")
        model_wrapper = ProtenixWrapper(  # pyright: ignore
            checkpoint_path=model_checkpoint_path, device=device, model=model
        )
    elif model_type == StructurePredictor.BOLTZ_1:
        logger.debug(f"Loading Boltz1 model from {model_checkpoint_path}")
        model_wrapper = Boltz1Wrapper(  # pyright: ignore
            checkpoint_path=model_checkpoint_path,
            use_msa_manager=True,
            device=device,
            model=model,
        )
    elif model_type == StructurePredictor.BOLTZ_2:
        if method is None:
            # TODO: make a useful error msg that includes options for method
            raise ValueError("Method must be specified for Boltz2")
        logger.debug(f"Loading Boltz2 model from {model_checkpoint_path}")
        model_wrapper = Boltz2Wrapper(  #  pyright: ignore
            checkpoint_path=model_checkpoint_path,
            use_msa_manager=True,
            device=device,
            method=method.upper(),
            model=model,
        )
    elif model_type == StructurePredictor.RF3:
        if RF3Wrapper is None:
            raise ImportError("RF3 dependencies not installed")
        model_wrapper = RF3Wrapper(checkpoint_path=model_checkpoint_path, msa_manager=MSAManager())
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # RF3 currently manages its own device; prefer that when available.
    device = getattr(model_wrapper, "device", device)

    # (pyright doesn't think Boltz1Wrapper etc are "Any")
    return device, model_wrapper  # pyright: ignore


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

    atom_array = structure["asym_unit"]  # pyright: ignore
    scattering_params = setup_scattering_params(atom_array=atom_array, em_mode=em, device=device)

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
    """
    Save model outputs to disk: writes the refined structure, two trajectory sets, and training losses.
    
    Parameters:
        output_dir (str | Path): Directory where results will be written; created if missing.
        losses (list[Any]): Sequence of loss values per step; entries may be None and will be recorded as "NA".
        refined_structure (dict): Structure dictionary containing an `"asym_unit"` used to write `refined.cif`.
        traj_denoised (list[Any]): Trajectory frames for the denoised run; saved under output_dir/denoised/.
        traj_next_step (list[Any]): Trajectory frames for the next-step run; saved under output_dir/next_step/.
        scaler_type (str): Scaler type used to select the trajectory save format.
    
    Behavior:
        - Writes the refined structure to output_dir/refined.cif.
        - Builds an occupancy-based mask from the first atom array in the structure; if occupancy is absent, all atoms are included.
        - Saves traj_denoised and traj_next_step into `denoised` and `next_step` subdirectories respectively (trajectory files are written every 10 steps).
        - Writes a losses text file in output_dir recording step and loss ("NA" for None).
        - Logs initial, final, and reduction in loss when available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    final_structure = CIFFile()
    set_structure(final_structure, refined_structure["asym_unit"])
    final_structure.write(str(output_dir / "refined.cif"))

    # Build the occupancy mask from the first array in the stack
    # Model atom arrays will lack occupancy annotation, in which case all atoms are considered valid
    asym_unit = get_asym_unit_from_structure(refined_structure)
    first_array: AtomArray = asym_unit[0] if isinstance(asym_unit, AtomArrayStack) else asym_unit  # pyright: ignore[reportAssignmentType]
    if hasattr(first_array, "occupancy"):
        reward_param_mask = first_array.occupancy > 0  # pyright: ignore[reportOptionalOperand]
    else:
        reward_param_mask = np.ones(len(first_array), dtype=bool)

    # Two calls to save_trajectory, very similar, but saving different trajectories!
    save_trajectory(
        scaler_type,
        traj_denoised,  # <--- the difference is here!
        asym_unit,  # this is just used as a dummy structure
        output_dir,
        reward_param_mask,
        "denoised",
        save_every=10,
    )
    save_trajectory(
        scaler_type,
        traj_next_step,  # <--- and here!
        asym_unit,
        output_dir,
        reward_param_mask,
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
# These args are passed from run_grid_search.py via GuidanceConfig.
def run_guidance(
    args: GuidanceConfig | argparse.Namespace, guidance_type: str, model_wrapper, device
) -> JobResult:
    """
    Wrapper around the actual _run_guidance function to redirect logs and generate a JobResult.
    Args:
        args:
        guidance_type:

    Returns:
    """

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    # separate logs for each guidance run
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
        logger.error(f"Error running guidance: {e} consult logs ({args.log_path}) for real errors.")
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
    use_alignment_for_reverse_diffusion = args.model in (
        StructurePredictor.BOLTZ_1,
        StructurePredictor.BOLTZ_2,
    )

    if guidance_type == GuidanceType.PURE_GUIDANCE:
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

    elif guidance_type == GuidanceType.FK_STEERING:
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
        method=args.method if hasattr(args, "method") else "None",
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
    if template_job.model_checkpoint is None or template_job.model_checkpoint == "":
        raise ValueError("Running guidance requires that you specify a model checkpoint")

    logger.info(f"Running {len(job_queue)} jobs, using {template_job} as a setup template")
    device, model_wrapper = get_model_and_device(
        str(template_job.device),
        template_job.model_checkpoint,
        template_job.model,
        method=template_job.method if hasattr(template_job, "method") else None,
    )
    job_results = []
    for i, job in enumerate(job_queue):
        logger.info(f"Running job {i + 1}/{len(job_queue)}: {job}")
        # TODO: I think it is safe now to re-use the wrapper, it might save us some time.
        # The model wrapper can persist state across runs, so we need to re-initialize it each run.
        # if job.model_checkpoint is None:
        #     raise ValueError(
        #         "Running guidance requires that you specify a model checkpoint, not None"
        #     )
        # device, model_wrapper = get_model_and_device(
        #     str(device), job.model_checkpoint, job.model, job.method, model_wrapper.model
        # )

        job_result = run_guidance(job, job.guidance_type, model_wrapper, device)
        job_results.append(job_result)
        torch.cuda.empty_cache()  # just in case

    if hasattr(model_wrapper, "msa_manager") and model_wrapper.msa_manager is not None:
        # reports the number of API calls and cache hits
        model_wrapper.msa_manager.report_on_usage()
    else:
        logger.warning(
            "No MSA manager found, cannot report on MSA usage. "
            "(why aren't you using an MSAManager?)"
        )

    return job_results