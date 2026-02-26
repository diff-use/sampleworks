from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import einx
import numpy as np
import torch
from jaxtyping import Float
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ELEMENT_TO_ATOMIC_NUM,
)


if TYPE_CHECKING:
    from biotite.structure import AtomArray, AtomArrayStack


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

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray | AtomArrayStack,
        ensemble_size: int,
        num_particles: int = 1,
        device: torch.device | str = "cpu",
    ) -> RewardInputs:
        """Construct RewardInputs from a Biotite AtomArray.

        Parameters
        ----------
        atom_array
            Biotite AtomArray or AtomArrayStack containing structure data.
        ensemble_size
            Number of ensemble members (batch dimension).
        num_particles
            Number of particles for FK steering (default 1 for pure guidance).
        device
            PyTorch device to place tensors on.

        Returns
        -------
        RewardInputs
            Dataclass containing all inputs needed for reward function computation.
        """
        occupancy_mask = atom_array.occupancy > 0
        nan_mask = ~np.any(np.isnan(atom_array.coord), axis=-1)
        reward_param_mask = occupancy_mask & nan_mask

        elements_list = [
            ELEMENT_TO_ATOMIC_NUM[e.title()] for e in atom_array.element[reward_param_mask]
        ]

        total_batch_size = num_particles * ensemble_size if num_particles > 1 else ensemble_size

        # If we have multiple particles (e.g. in FK Steering), we need to tile the elements and
        # b_factors across the particle dimension.
        if num_particles > 1:
            elements = einx.rearrange(
                "n -> p e n", torch.Tensor(elements_list), p=num_particles, e=ensemble_size
            )
            b_factors = einx.rearrange(
                "n -> p e n",
                torch.Tensor(atom_array.b_factor[reward_param_mask]),
                p=num_particles,
                e=ensemble_size,
            )
            occupancies = torch.ones_like(b_factors) / ensemble_size
            input_coords = einx.rearrange(
                "... -> b ...",
                torch.from_numpy(atom_array.coord).to(dtype=torch.float32),
                b=total_batch_size,
            )[..., reward_param_mask, :]
        else:
            elements = einx.rearrange("n -> b n", torch.Tensor(elements_list), b=ensemble_size)
            b_factors = einx.rearrange(
                "n -> b n",
                torch.Tensor(atom_array.b_factor[reward_param_mask]),
                b=ensemble_size,
            )
            occupancies = torch.ones_like(b_factors) / ensemble_size
            input_coords = einx.rearrange(
                "... -> e ...",
                torch.from_numpy(atom_array.coord).to(dtype=torch.float32),
                e=ensemble_size,
            )[..., reward_param_mask, :]

        mask_like = torch.ones_like(input_coords[..., 0])

        if isinstance(device, str):
            device = torch.device(device)

        return cls(
            elements=elements.to(device),
            b_factors=b_factors.to(device),
            occupancies=occupancies.to(device),
            input_coords=input_coords.to(device),
            reward_param_mask=reward_param_mask,
            mask_like=mask_like.to(device),
        )


@runtime_checkable
class RewardFunctionProtocol(Protocol):
    """Protocol for reward functions used in guided sampling.

    Any callable that computes a scalar reward from atomic coordinates
    and properties can implement this protocol.
    """

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Float[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Compute reward value from atomic coordinates and properties.

        Parameters
        ----------
        coordinates
            Atomic coordinates, shape [batch, n_atoms, 3]
        elements
            Atomic element indices, shape [batch, n_atoms]
        b_factors
            Per-atom B-factors, shape [batch, n_atoms]
        occupancies
            Per-atom occupancies, shape [batch, n_atoms]

        These next parameters are required for vmap compatibility:
        unique_combinations
            Optional pre-computed unique (element, b_factor) pairs
        inverse_indices
            Optional pre-computed inverse indices for vmap compatibility

        Returns
        -------
        Float[torch.Tensor, ""]
            Scalar reward value
        """
        ...


@runtime_checkable
class PrecomputableRewardFunctionProtocol(RewardFunctionProtocol, Protocol):
    """Protocol for reward functions with precomputation for vmap compatibility.

    Extends RewardFunctionProtocol with a method to precompute unique
    (element, b_factor) combinations, avoiding dynamic shapes in vmap contexts.
    """

    def precompute_unique_combinations(
        self,
        elements: torch.Tensor,
        b_factors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute unique (element, b_factor) combinations.

        Parameters
        ----------
        elements
            Atomic element indices
        b_factors
            Per-atom B-factors

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            unique_combinations: Unique (element, b_factor) pairs
            inverse_indices: Indices to reconstruct original from unique
        """
        ...
