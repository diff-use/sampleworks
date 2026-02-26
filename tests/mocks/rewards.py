"""Mock implementations of reward protocols for testing."""

from torch import Tensor


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
