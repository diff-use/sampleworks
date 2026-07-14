from __future__ import annotations

import torch
from jaxtyping import Float, Int


class CompositeReward:
    """Weighted sum of reward functions using the RewardFunctionProtocol"""

    def __init__(
        self,
        rewards: list,
        weights: list[float] | None = None,
    ):

        if weights is None:
            weights = [1.0] * len(rewards)
        if len(weights) != len(rewards):
            raise ValueError(f"Got {len(rewards)} rewards but {len(weights)} weights.")
        self.rewards = rewards
        self.weights = weights

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Weighted sum of the sub-rewards; differentiable w.r.t coordinates."""
        total = None
        for reward, weight in zip(self.rewards, self.weights):
            term = weight * reward(
                coordinates,
                elements,
                b_factors,
                occupancies,
                unique_combinations,
                inverse_indices,
            )
            total = term if total is None else total + term
        return total

    def prepare(self, atom_array):
        for rewards in self.rewards:
            if hasattr(rewards, "prepare"):
                rewards.prepare(atom_array)  # prepare from TmolPlausibility
        return self
