"""Mock implementations of sampler protocols for testing."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import einx
import torch
from jaxtyping import Float
from sampleworks.core.samplers.protocol import SamplerSchedule, SamplerStepOutput, StepContext
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.scalers.protocol import StepScalerProtocol
    from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput


class MockSchedule(SamplerSchedule):
    """Simple mock schedule for testing."""

    timesteps: Float[Tensor, " steps"]
    num_steps: int

    def __init__(self, timesteps: Tensor, num_steps: int):
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "num_steps", num_steps)

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:
        return {"timesteps": self.timesteps}


class MockTrajectorySampler:
    """Mock TrajectorySampler that satisfies the protocol.

    Provides simple Euler-style sampling for testing.
    """

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")

    def compute_schedule(self, num_steps: int) -> MockSchedule:
        """Compute a simple linear schedule from 1.0 to 0.0."""
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        return MockSchedule(timesteps=timesteps, num_steps=num_steps)

    def get_context_for_step(
        self, step_index: int, schedule: SamplerSchedule, total_steps: int | None = None
    ) -> StepContext:
        """Build StepContext from schedule for given step."""
        mock_schedule = cast(MockSchedule, schedule)
        if total_steps is None:
            total_steps = mock_schedule.num_steps

        t = mock_schedule.timesteps[step_index]
        t_next = mock_schedule.timesteps[step_index + 1]
        dt = t_next - t

        return StepContext(
            step_index=step_index,
            total_steps=total_steps,
            t=t,
            dt=dt,
            noise_scale=torch.tensor(0.01, device=self.device),
        )

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the schedule is compatible with this sampler."""
        if not hasattr(schedule, "timesteps"):
            raise ValueError("MockTrajectorySampler requires schedule with timesteps")

    def check_context(self, context: StepContext) -> None:
        """Validate StepContext for this sampler."""
        if not context.is_trajectory:
            raise ValueError("MockTrajectorySampler requires trajectory-based StepContext")

    def step(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        model_wrapper: FlowModelWrapper,
        context: StepContext,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput:
        """Take one sampling step.

        Follows the EDM style sampling, but without the complex alignment, etc.

        state + dt * delta where delta = (model_update - state) / t

        For NoiseSpaceDPSScaler compatibility:
        - When scaler has requires_gradients=True, creates noisy_state with requires_grad=True
        - Passes x_t in context.metadata for gradient backprop
        """
        self.check_context(context)

        t = cast(Tensor, context.t)
        dt = cast(Tensor, context.dt)

        allow_gradients = scaler is not None and getattr(scaler, "requires_gradients", False)

        noisy_state = state.detach().requires_grad_(allow_gradients)

        model_update = model_wrapper.step(noisy_state, t, features=features)
        delta = (model_update - noisy_state) / t

        loss = None
        guidance_direction_raw = None
        if scaler is not None:
            scaler_metadata: dict[str, object] = {"x_t": noisy_state}
            if context.metadata:
                scaler_metadata = {**context.metadata, "x_t": noisy_state}

            scaler_context = StepContext(
                step_index=context.step_index,
                total_steps=context.total_steps,
                t=context.t,
                dt=context.dt,
                noise_scale=context.noise_scale,
                learning_rate=context.learning_rate,
                reward=context.reward,
                reward_inputs=context.reward_inputs,
                metadata=scaler_metadata,
            )

            guidance_direction, loss = scaler.scale(
                model_update, scaler_context, model=model_wrapper
            )
            guidance_direction_raw = (
                guidance_direction.clone()
                if isinstance(guidance_direction, Tensor)
                else torch.as_tensor(guidance_direction)
            )
            guidance_weight = scaler.guidance_strength(context)
            if isinstance(guidance_weight, Tensor) and guidance_weight.ndim == 1:
                guidance_weight = guidance_weight.view(-1, 1, 1)
            delta = delta + guidance_weight * guidance_direction

        next_state = noisy_state + dt * delta

        log_proposal_correction = None
        if guidance_direction_raw is not None and context.noise_scale is not None:
            ns = context.noise_scale
            noise_var = (ns.item() if isinstance(ns, Tensor) else float(ns)) ** 2
            if noise_var > 0:
                # Approximate: -shift^2 / (2 * var)
                log_proposal_correction = -(guidance_direction_raw**2) / (2 * noise_var)
                log_proposal_correction = einx.sum("... [b n c]", log_proposal_correction)

        return SamplerStepOutput(
            state=next_state.detach(),
            denoised=model_update.detach() if isinstance(model_update, Tensor) else model_update,
            loss=loss,
            log_proposal_correction=log_proposal_correction,
        )
