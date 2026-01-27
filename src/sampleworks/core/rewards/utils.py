"""Shared utilities for scaler implementations."""

from dataclasses import dataclass
from typing import Any

import einx
import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOMIC_NUM_TO_ELEMENT,
)


@dataclass
class RewardInputs:
    """Extracted inputs for reward function computation.

    Contains all the information needed to call a RewardFunctionProtocol,
    extracted from an atom array. This allows the caller to extract inputs
    once and pass them to scale() methods without redundant extraction.
    """

    elements: Float[torch.Tensor, "*batch n_atoms"]
    b_factors: Float[torch.Tensor, "*batch n_atoms"]
    occupancies: Float[torch.Tensor, "*batch n_atoms"]
    input_coords: Float[torch.Tensor, "*batch n_atoms 3"]
    reward_param_mask: np.ndarray
    mask_like: Float[torch.Tensor, "*batch n_atoms"]


def extract_reward_inputs(
    atom_array: AtomArray | AtomArrayStack,
    ensemble_size: int,
    num_particles: int = 1,
    device: torch.device | str = "cpu",
) -> RewardInputs:
    """Extract reward function inputs from atom array (centralizes duplicated logic).

    Parameters
    ----------
    atom_array
        Biotite AtomArray or AtomArrayStack containing structure data
    ensemble_size
        Number of ensemble members (batch dimension)
    num_particles
        Number of particles for FK steering (default 1 for pure guidance)
    device
        PyTorch device to place tensors on

    Returns
    -------
    RewardInputs
        Dataclass containing all inputs needed for reward function computation
    """
    occupancy_mask = atom_array.occupancy > 0  # pyright: ignore[reportOptionalOperand]
    nan_mask = ~np.any(np.isnan(atom_array.coord), axis=-1)  # pyright: ignore[reportArgumentType, reportCallIssue]
    reward_param_mask = occupancy_mask & nan_mask

    elements_list = [
        ATOMIC_NUM_TO_ELEMENT.index(e.title())
        for e in atom_array.element[reward_param_mask]  # pyright: ignore[reportOptionalSubscript]
    ]

    total_batch = num_particles * ensemble_size if num_particles > 1 else ensemble_size

    if num_particles > 1:
        elements = einx.rearrange(
            "n -> p e n", torch.Tensor(elements_list), p=num_particles, e=ensemble_size
        )
        b_factors = einx.rearrange(
            "n -> p e n",
            torch.Tensor(atom_array.b_factor[reward_param_mask]),  # pyright: ignore[reportOptionalSubscript]
            p=num_particles,
            e=ensemble_size,
        )
        occupancies = torch.ones_like(b_factors) / ensemble_size  # pyright: ignore[reportArgumentType]
        input_coords = einx.rearrange(
            "... -> b ...",
            torch.from_numpy(atom_array.coord).to(dtype=torch.float32),
            b=total_batch,
        )[..., reward_param_mask, :]  # pyright: ignore[reportArgumentType, reportCallIssue]
    else:
        elements = einx.rearrange("n -> b n", torch.Tensor(elements_list), b=ensemble_size)
        b_factors = einx.rearrange(
            "n -> b n",
            torch.Tensor(atom_array.b_factor[reward_param_mask]),  # pyright: ignore[reportOptionalSubscript]
            b=ensemble_size,
        )
        occupancies = torch.ones_like(b_factors) / ensemble_size  # pyright: ignore[reportArgumentType]
        input_coords = einx.rearrange(
            "... -> e ...",
            torch.from_numpy(atom_array.coord).to(dtype=torch.float32),
            e=ensemble_size,
        )[..., reward_param_mask, :]  # pyright: ignore[reportArgumentType, reportCallIssue]

    mask_like = torch.ones_like(input_coords[..., 0])

    if isinstance(device, str):
        device = torch.device(device)

    return RewardInputs(
        elements=elements.to(device),  # pyright: ignore[reportAttributeAccessIssue]
        b_factors=b_factors.to(device),  # pyright: ignore[reportAttributeAccessIssue]
        occupancies=occupancies.to(device),
        input_coords=input_coords.to(device),
        reward_param_mask=reward_param_mask,
        mask_like=mask_like.to(device),
    )


def get_atom_array_from_model_input(
    features: Any, structure: dict, model_class_name: str
) -> AtomArray | AtomArrayStack:
    """Extract atom array handling model-specific differences.

    Parameters
    ----------
    features
        Model features as returned by featurize()
    structure
        Atomworks structure dictionary
    model_class_name
        Name of the model wrapper class (e.g., "ProtenixWrapper", "BoltzWrapper")

    Returns
    -------
    AtomArray | AtomArrayStack
        The atom array to use for reward computation
    """
    if model_class_name == "ProtenixWrapper":
        if hasattr(features, "conditioning") and hasattr(features.conditioning, "feats"):
            return features.conditioning.feats.get("true_atom_array")
        return features.get("true_atom_array")
    return structure["asym_unit"][0]
