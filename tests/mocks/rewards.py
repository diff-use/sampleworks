"""Mock implementations of reward protocols for testing."""

import torch
from torch import Tensor


class MockRewardFunction:
    """Mock RewardFunction that satisfies RewardFunctionProtocol.

    Returns the sum of squared coordinates, which provides a simple
    differentiable loss for testing guidance computation.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._call_count = 0

    def __call__(
        self,
        coordinates: Tensor,
        elements: Tensor,
        b_factors: Tensor,
        occupancies: Tensor,
        unique_combinations: Tensor | None = None,
        inverse_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute a simple loss as sum of squared coordinates."""
        self._call_count += 1
        return self.scale * (coordinates**2).sum()


class MockPrecomputableRewardFunction:
    """Mock RewardFunction that satisfies PrecomputableRewardProtocol.

    Includes precompute_unique_combinations() for FK steering compatibility.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._call_count = 0

    def __call__(
        self,
        coordinates: Tensor,
        elements: Tensor,
        b_factors: Tensor,
        occupancies: Tensor,
        unique_combinations: Tensor | None = None,
        inverse_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute a simple loss as sum of squared coordinates."""
        self._call_count += 1
        return self.scale * (coordinates**2).sum()

    def precompute_unique_combinations(
        self,
        elements: Tensor,
        b_factors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pre-compute unique combinations"""
        elements_flat = elements.reshape(-1)
        b_factors_flat = b_factors.reshape(-1)
        combined = torch.stack([elements_flat, b_factors_flat], dim=1)
        unique_combinations, inverse_indices = torch.unique(combined, dim=0, return_inverse=True)
        return unique_combinations, inverse_indices


class MockGradientRewardFunction:
    """Mock RewardFunction with predictable gradient behavior for testing.

    Returns a loss where the gradient with respect to coordinates is
    simply the coordinates themselves (times a scale factor).
    """

    def __init__(self, gradient_scale: float = 1.0):
        self.gradient_scale = gradient_scale

    def __call__(
        self,
        coordinates: Tensor,
        elements: Tensor | None = None,
        b_factors: Tensor | None = None,
        occupancies: Tensor | None = None,
        unique_combinations: Tensor | None = None,
        inverse_indices: Tensor | None = None,
    ) -> Tensor:
        """Loss = 0.5 * scale * ||coords||^2, so grad = scale * coords."""
        return 0.5 * self.gradient_scale * (coordinates**2).sum()
