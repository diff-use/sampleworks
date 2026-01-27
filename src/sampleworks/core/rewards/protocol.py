from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float


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
