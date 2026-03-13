from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import einx
import numpy as np
import torch
from jaxtyping import Float, Int
from sampleworks.utils.elements import elements_to_scattering_indices


if TYPE_CHECKING:
    from biotite.structure import AtomArray, AtomArrayStack


@dataclass
class RewardInputs:
    """Extracted inputs for reward function computation.

    Contains all the information needed to call a RewardFunctionProtocol,
    extracted from an atom array. This allows the caller to extract inputs
    once and pass them to scale() methods without redundant extraction.

    The atom array passed to :meth:`from_atom_array` must already be clean:
    all coordinates finite and all occupancies positive.  Wrappers are
    responsible for ensuring this (e.g. replacing NaN coordinates with
    noise and setting occupancy to 1.0 for model-operated atoms).
    """

    elements: Int[torch.Tensor, "*batch n_atoms"]
    b_factors: Float[torch.Tensor, "*batch n_atoms"]
    occupancies: Float[torch.Tensor, "*batch n_atoms"]
    input_coords: Float[torch.Tensor, "*batch n_atoms 3"]

    @classmethod
    def from_atom_array(
        cls,
        atom_array: AtomArray | AtomArrayStack,
        ensemble_size: int,
        num_particles: int = 1,
        device: torch.device | str = "cpu",
    ) -> RewardInputs:
        """Construct RewardInputs from a Biotite AtomArray.

        The atom array must contain only valid atoms (finite coordinates,
        positive occupancy).  Callers are responsible for filtering
        beforehand; no masking is applied here.

        Parameters
        ----------
        atom_array
            Biotite AtomArray or AtomArrayStack containing structure data.
            Must have not NaN coordinates and positive occupancy.
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
        # input validation: ensure atom_array has required annotations and valid values
        if not hasattr(atom_array, "element"):
            raise ValueError("Atom array must have 'element' annotation.")
        if not hasattr(atom_array, "b_factor"):
            raise ValueError("Atom array must have 'b_factor' annotation.")
        if np.any(np.isnan(atom_array.coord)):
            raise ValueError("Atom array contains NaN coordinates.")
        if np.any((atom_array.occupancy <= 0) | (atom_array.occupancy > 1)):
            raise ValueError("Atom array contains invalid occupancy values.")

        elements_list = elements_to_scattering_indices(atom_array.element)

        total_batch_size = num_particles * ensemble_size if num_particles > 1 else ensemble_size

        # ensure contiguous arrays for safe conversion to PyTorch tensors
        coords_np = np.ascontiguousarray(np.asarray(atom_array.coord))
        coords_t = torch.from_numpy(coords_np).to(dtype=torch.float32)

        # If we have multiple particles (e.g. in FK Steering), we need to tile the elements and
        # b_factors across the particle dimension.
        if num_particles > 1:
            elements = einx.rearrange(
                "n -> p e n",
                torch.tensor(elements_list, dtype=torch.long),
                p=num_particles,
                e=ensemble_size,
            )
            b_factors = einx.rearrange(
                "n -> p e n",
                torch.Tensor(atom_array.b_factor),
                p=num_particles,
                e=ensemble_size,
            )
            occupancies = torch.ones_like(b_factors) / ensemble_size
            input_coords = einx.rearrange(
                "... -> b ...",
                coords_t,
                b=total_batch_size,
            )
        else:
            elements = einx.rearrange(
                "n -> b n", torch.tensor(elements_list, dtype=torch.long), b=ensemble_size
            )
            b_factors = einx.rearrange(
                "n -> b n",
                torch.Tensor(atom_array.b_factor),
                b=ensemble_size,
            )
            occupancies = torch.ones_like(b_factors) / ensemble_size
            input_coords = einx.rearrange(
                "... -> e ...",
                coords_t,
                e=ensemble_size,
            )

        if isinstance(device, str):
            device = torch.device(device)

        return cls(
            elements=elements.to(device),
            b_factors=b_factors.to(device),
            occupancies=occupancies.to(device),
            input_coords=input_coords.to(device),
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
        elements: Int[torch.Tensor, "batch n_atoms"],
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
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
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
