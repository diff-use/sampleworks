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
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArray, AtomArrayStack, stack
from biotite.structure.io import save_structure
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import (
    RealSpaceRewardFunction,
    setup_scattering_params,
)
from sampleworks.core.samplers.edm import AF3EDMSampler
from sampleworks.core.scalers.fk_steering import FKSteering
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import (
    DataSpaceDPSScaler,
    NoiseSpaceDPSScaler,
)
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
    Boltz1Wrapper = None  # ty:ignore[invalid-assignment]
    logger.warning("Failed to import Boltz, hopefully you're running a different model")
try:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
except ImportError:
    ProtenixWrapper = None  # ty:ignore[invalid-assignment]
    logger.warning("Failed to import Protenix, hopefully you're running a different model")
try:
    from sampleworks.models.rf3.wrapper import RF3Wrapper
except ImportError:
    RF3Wrapper = None  # ty:ignore[invalid-assignment]
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


def _write_coords_into_array(
    array_copy: AtomArrayStack,
    coords: np.ndarray,
    reward_param_mask: np.ndarray,
) -> None:
    """**Mutates** ``array_copy.coord`` in-place with trajectory coordinates.

    When the trajectory spans all atoms in the array (model trajectories during
    a mismatch run, where the model's internal atom count differs from the input structure we are
    aligning to), coords are assigned directly to ``.coord``. Otherwise the
    ``reward_param_mask`` indexes the correct atom subset.
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
        _write_coords_into_array(array_copy, coords.detach().numpy(), reward_param_mask)
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
        _write_coords_into_array(array_copy, coords[0].detach().numpy(), reward_param_mask)
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
        if ProtenixWrapper is None:
            raise ImportError("Protenix dependencies not installed")
        logger.debug(f"Loading Protenix model from {model_checkpoint_path}")
        model_wrapper = ProtenixWrapper(
            checkpoint_path=model_checkpoint_path, device=device, model=model
        )
    elif model_type == StructurePredictor.BOLTZ_1:
        if Boltz1Wrapper is None:
            raise ImportError("Boltz dependencies not installed")
        logger.debug(f"Loading Boltz1 model from {model_checkpoint_path}")
        model_wrapper = Boltz1Wrapper(
            checkpoint_path=model_checkpoint_path,
            use_msa_manager=True,
            device=device,
            model=model,
        )
    elif model_type == StructurePredictor.BOLTZ_2:
        if Boltz2Wrapper is None:
            raise ImportError("Boltz dependencies not installed")
        if method is None:
            # TODO: make a useful error msg that includes options for method
            raise ValueError("Method must be specified for Boltz2")
        logger.debug(f"Loading Boltz2 model from {model_checkpoint_path}")
        model_wrapper = Boltz2Wrapper(
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
    return device, model_wrapper


# TODO: further atomize for easier testing.
def get_reward_function_and_structure(
    density: str | Path,
    device: torch.device,
    em,
    loss_order,
    resolution,
    structure_path: str | Path,
) -> tuple[RealSpaceRewardFunction, dict[str, Any]]:
    logger.debug(f"Loading structure from {structure_path}")
    structure = parse(
        Path(structure_path),
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )

    logger.debug(f"Loading density map from {density}")
    xmap = XMap.fromfile(density, resolution=resolution)

    logger.debug("Setting up scattering parameters")

    atom_array = structure["asym_unit"]
    scattering_params = setup_scattering_params(atom_array=atom_array, em_mode=em, device=device)

    selection_mask = atom_array.occupancy > 0
    n_selected = selection_mask.sum()
    logger.info(f"Selected {n_selected} atoms with occupancy > 0")

    logger.info("Creating reward function")
    reward_function = RealSpaceRewardFunction(
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
    final_state: torch.Tensor | None = None,
    model_atom_array: AtomArray | None = None,
) -> None:
    """Save everything: refined structure/ensemble CIF, trajectories, and losses.

    When `final_state` is provided, its coordinates are written into the
    `refined_structure` atom array (respecting the occupancy/NaN validity mask) before
    saving.  Both the denoised and next-step trajectories are saved as
    multi-model CIF files (subsampled every 10 steps via ``save_trajectory``).

    Parameters
    ----------
    output_dir : str | Path
        Directory to write all output files into. Created if it doesn't exist.
    losses : list[Any]
        Per-step loss values (may contain ``None`` entries for unguided steps).
    refined_structure : dict
        Atomworks structure dict whose ``"asym_unit"`` is used as the template
        for saving.
    traj_denoised : list[Any]
        Denoised-prediction trajectory tensors, one per diffusion step.
    traj_next_step : list[Any]
        Next-step (noisy) trajectory tensors, one per diffusion step.
    scaler_type : str
        Scaler/guidance identifier, forwarded to ``save_trajectory`` for
        scaler-specific handling. # TODO: handle more gracefully
    final_state : torch.Tensor | None
        Final coordinates with shape ``(ensemble, atoms, 3)``.  If ``None``,
        the `refined_structure`'s existing coordinates are saved as-is.
    model_atom_array : AtomArray | None
        Optional model-space atom template. When provided (mismatch runs),
        this template is used for final structure and trajectory saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving results")
    from biotite.structure.io.pdbx import CIFFile, set_structure

    base_atom_array = ensure_atom_array_stack(refined_structure["asym_unit"])[0]

    # Use model's internal atom accounting template for mismatch runs when available
    atom_array_for_masking: AtomArray = (
        model_atom_array if model_atom_array is not None else base_atom_array
    )

    # Build occupancy mask properly as model atom arrays may lack occupancy annotation
    if (
        hasattr(atom_array_for_masking, "occupancy")
        and atom_array_for_masking.occupancy is not None
    ):
        occupancy_mask = atom_array_for_masking.occupancy > 0
        occupancy_mask &= ~np.any(np.isnan(atom_array_for_masking.coord), axis=-1)
    else:
        occupancy_mask = np.ones(len(atom_array_for_masking), dtype=bool)

    if final_state is not None:
        ensemble_size = final_state.shape[0]

        ensemble_array = stack([atom_array_for_masking.copy() for _ in range(ensemble_size)])
        _write_coords_into_array(ensemble_array, final_state.detach().cpu().numpy(), occupancy_mask)
        atom_array = ensemble_array
    else:
        atom_array = base_atom_array

    final_structure = CIFFile()
    set_structure(final_structure, atom_array)
    final_structure.write(str(output_dir / "refined.cif"))

    # Two calls to save_trajectory, very similar, but saving different trajectories!
    save_trajectory(
        scaler_type,
        traj_denoised,  # <--- the difference is here!
        atom_array_for_masking,
        output_dir,
        occupancy_mask,
        "denoised",
        save_every=10,
    )
    save_trajectory(
        scaler_type,
        traj_next_step,  # <--- and here!
        atom_array_for_masking,
        output_dir,
        occupancy_mask,
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

    log_path = getattr(args, "log_path", None) or os.path.join(args.output_dir, "run.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # separate logs for each guidance run
    handle = logger.add(
        log_path, level="INFO", filter=lambda rec: rec["extra"].get("special", False) is True
    )
    started_at = datetime.now()
    try:
        with logger.contextualize(special=True):
            _run_guidance(args, guidance_type, model_wrapper, device)
        logger.info("Guidance run successfully!")
        return get_job_result(args, device, started_at, datetime.now(), 0, "success")
    except Exception as e:
        logger.error(f"Error running guidance: {e} consult logs ({log_path}) for real errors.")
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

    # Determine model type from wrapper class name
    wrapper_class_name = model_wrapper.__class__.__name__
    is_boltz = "Boltz" in wrapper_class_name

    # Boltz was trained with this, others might not have been.
    use_alignment_for_reverse_diffusion = is_boltz

    # Create sampler with model-appropriate settings
    sampler = AF3EDMSampler(
        device=str(device),
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        alignment_reverse_diffusion=use_alignment_for_reverse_diffusion,
    )

    # Create step scaler for gradient-based guidance
    use_tweedie = getattr(args, "use_tweedie", False)
    if use_tweedie:
        step_scaler = DataSpaceDPSScaler(
            step_size=args.step_size,
            gradient_normalization=args.gradient_normalization,
        )
    else:
        step_scaler = NoiseSpaceDPSScaler(
            step_size=args.step_size,
            gradient_normalization=args.gradient_normalization,
        )

    # TODO: this should be a config option
    num_steps = 200

    if guidance_type == GuidanceType.PURE_GUIDANCE:
        logger.info("Initializing pure guidance")

        # TODO: these should be fractions in the args directly
        guidance_t_start = args.guidance_start / num_steps if args.guidance_start > 0 else 0.0
        t_start = args.partial_diffusion_step / num_steps if args.partial_diffusion_step else 0.0

        guidance = PureGuidance(
            ensemble_size=args.ensemble_size,
            num_steps=num_steps,
            t_start=t_start,
            guidance_t_start=guidance_t_start,
        )

        logger.info(f"Running pure guidance using model with hash {model_wrapper.model.__hash__()}")
        result = guidance.sample(
            structure=structure,
            model=model_wrapper,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward_function,
        )

        refined_structure = result.structure
        losses = result.losses if result.losses else []
        traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
        traj_next_step = list(result.trajectory) if result.trajectory else []

    elif guidance_type == GuidanceType.FK_STEERING:
        logger.info("Initializing Feynman-Kac steering")

        # TODO: same as above
        gs = args.guidance_start
        guidance_start_fraction = gs / num_steps if gs > 0 else 0.0
        pd = args.partial_diffusion_step
        t_start = pd / num_steps if pd else 0.0

        guidance = FKSteering(
            ensemble_size=args.ensemble_size,
            num_steps=num_steps,
            resampling_interval=args.fk_resampling_interval,
            fk_lambda=args.fk_lambda,
            guidance_t_start=guidance_start_fraction,
            t_start=t_start,
        )

        logger.info("Running FK steering")
        result = guidance.sample(
            structure=structure,
            model=model_wrapper,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward_function,
            num_particles=args.num_particles,
        )

        refined_structure = result.structure
        losses = result.losses if result.losses else []
        traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
        traj_next_step = list(result.trajectory) if result.trajectory else []
    else:
        logger.error(f"Unknown guidance type: {guidance_type}")
        raise TypeError("Unknown guidance type!")

    model_atom_array = result.metadata.get("model_atom_array") if result.metadata else None

    save_everything(
        args.output_dir,
        losses,
        refined_structure,
        traj_denoised,
        traj_next_step,
        guidance_type,
        final_state=torch.as_tensor(result.final_state),
        model_atom_array=model_atom_array,
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
        method=getattr(args, "method", None),
        scaler=args.guidance_type,
        ensemble_size=args.ensemble_size,
        gradient_weight=getattr(args, "guidance_weight", -1.0),
        gd_steps=getattr(args, "num_gd_steps", -1),
        status=status,
        exit_code=exit_code,
        runtime_seconds=round(end_time - start_time, 2),
        started_at=started_at.isoformat(),
        finished_at=ended_at.isoformat(),
        log_path=getattr(args, "log_path", None) or os.path.join(args.output_dir, "run.log"),
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
