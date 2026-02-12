"""Mock scaler implementations for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.samplers.protocol import StepParams
    from sampleworks.models.protocol import FlowModelWrapper


class MockStepScaler:
    """Mock scaler for testing - returns fixed guidance direction."""

    requires_gradients: bool = False

    def __init__(self, guidance_tensor: Tensor | None = None, step_size: float = 0.1):
        self.guidance_tensor = guidance_tensor
        self.step_size = step_size
        self._last_batch_size: int = 1
        self._last_device: torch.device = torch.device("cpu")

    def scale(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        context: StepParams,
        *,
        model: FlowModelWrapper | None = None,
    ) -> tuple[Float[Tensor, "*batch atoms 3"], Float[Tensor, " batch"]]:
        self._last_batch_size = state.shape[0]
        self._last_device = state.device
        if self.guidance_tensor is not None:
            direction = self.guidance_tensor.to(state.device)
        else:
            direction = torch.randn_like(state) * 0.1
        loss = torch.zeros(state.shape[0], device=state.device)
        return direction, loss

    def guidance_strength(self, context: StepParams) -> Float[Tensor, " batch"]:
        t = context.t_effective
        if isinstance(t, Tensor):
            result = torch.ones_like(t) * self.step_size
            if result.ndim == 0:
                result = result.unsqueeze(0)
            if result.shape[0] != self._last_batch_size:
                result = result.expand(self._last_batch_size)
            return result.to(self._last_device)
        return torch.full((self._last_batch_size,), self.step_size, device=self._last_device)
