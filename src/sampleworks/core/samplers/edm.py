"""EDM-style sampler using AF3 schedule formula."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import einx
import torch
from jaxtyping import Float

from sampleworks.core.samplers.protocol import SamplerSchedule, SamplerStepOutput, StepContext
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput
from sampleworks.utils.frame_transforms import (
    align_to_reference_frame,
    apply_forward_transform,
    create_random_transform,
    transform_coords_and_noise_to_frame,
    weighted_rigid_align_differentiable,
)


if TYPE_CHECKING:
    from sampleworks.core.scalers.protocol import StepScalerProtocol


@dataclass(frozen=True, slots=True)
class EDMSchedule(SamplerSchedule):
    """EDM noise schedule arrays."""

    sigma_tm: Float[torch.Tensor, " steps"]
    sigma_t: Float[torch.Tensor, " steps"]
    gamma: Float[torch.Tensor, " steps"]

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:
        return {
            "sigma_tm": self.sigma_tm,
            "sigma_t": self.sigma_t,
            "gamma": self.gamma,
        }


@dataclass(frozen=True, slots=True)
class EDMSamplerConfig:
    """Config to initialize an EDM sampler plus common step defaults."""

    # Schedule parameters (AF3 defaults)
    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0
    gamma_min: float = 0.2
    gamma_0: float = 0.8
    noise_scale: float = 1.003
    augmentation: bool = True
    align_to_input: bool = True
    alignment_reverse_diffusion: bool = False
    device: str | torch.device = "cpu"

    # Default per-step scale (AF3 default)
    step_scale: float = 1.5

    def create_sampler(self) -> AF3EDMSampler:
        """Create EDM sampler instance from this config."""
        return AF3EDMSampler(
            sigma_data=self.sigma_data,
            s_max=self.s_max,
            s_min=self.s_min,
            p=self.p,
            gamma_min=self.gamma_min,
            gamma_0=self.gamma_0,
            noise_scale=self.noise_scale,
            step_scale=self.step_scale,
            augmentation=self.augmentation,
            align_to_input=self.align_to_input,
            alignment_reverse_diffusion=self.alignment_reverse_diffusion,
            device=self.device,
        )


@dataclass
class AF3EDMSampler:
    """EDM-style sampler from AF3-like models.

    All constants are configurable via constructor for model-specific values.
    Default values match AF3 parameterization.

    This sampler implements the EDM (Karras et al.) style sampling
    approach as used in AlphaFold3 and related models, which is the Euler
    variant.
    """

    # Schedule parameters (with defaults from AF3 and the like)
    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0
    gamma_min: float = 0.2
    gamma_0: float = 0.8
    noise_scale: float = 1.003
    step_scale: float = 1.5
    augmentation: bool = True
    align_to_input: bool = True
    alignment_reverse_diffusion: bool = False
    device: str | torch.device = "cpu"

    def check_context(self, context: StepContext) -> None:
        """Validate that the provided StepContext is ready for step.

        Raises
        ------
        ValueError
            If the context is incompatible with this sampler.
        """
        if not context.is_trajectory:
            raise ValueError("AF3EDMSampler requires trajectory-based StepContext with time info")
        if (
            context.t is None
            or context.dt is None
            or context.noise_scale is None
            or context.total_steps is None
        ):
            raise ValueError("AF3EDMSampler requires t, dt, and noise_scale in StepContext")
        if context.step_index >= context.total_steps:
            raise ValueError("StepContext step_index exceeds total_steps")

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the provided schedule is compatible with this sampler.

        Raises
        ------
        ValueError
            If the schedule is incompatible with this sampler.
        """
        if not (
            hasattr(schedule, "sigma_tm")
            and hasattr(schedule, "sigma_t")
            and hasattr(schedule, "gamma")
        ):
            raise ValueError(
                "EDMSampler requires SamplerSchedule with sigma_tm, sigma_t, and gamma"
            )

    def compute_schedule(self, num_steps: int) -> EDMSchedule:
        """Compute sigma-based schedule using RF3's EDM formula.

        Uses the formula:
        sigma = sigma_data * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p

        where t goes from 0 to 1 over num_steps.

        Parameters
        ----------
        num_steps: int
            Number of diffusion sampling steps.

        Returns
        -------
        EDMSchedule
            Schedule object with sigma_tm, sigma_t, and gamma arrays.
        """
        t_values = torch.linspace(0, 1, num_steps + 1, device=self.device)

        sigmas = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + t_values * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        gammas = torch.where(
            sigmas > self.gamma_min,
            torch.tensor(self.gamma_0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )

        return EDMSchedule(
            sigma_tm=sigmas[:-1],
            sigma_t=sigmas[1:],
            gamma=gammas[1:],
        )

    def get_context_for_step(
        self, step_index: int, schedule: SamplerSchedule, total_steps: int | None = None
    ) -> StepContext:
        """Build StepContext from schedule for given step.

        Parameters
        ----------
        step_index: int
            Current timestep index (0-indexed).
        schedule: SamplerSchedule
            The schedule returned by compute_schedule() (must be EDMSchedule).
        total_steps: int | None
            Total number of steps (optional, inferred from schedule if not provided).

        Returns
        -------
        StepContext
            Context with t, dt, noise_scale populated for this step.
        """

        self.check_schedule(schedule)

        sigma_tm = schedule.sigma_tm[step_index]  # pyright: ignore[reportAttributeAccessIssue] (this will be accessible due to the check above)
        sigma_t = schedule.sigma_t[step_index]  # pyright: ignore[reportAttributeAccessIssue] (this will be accessible due to the check above)
        gamma = schedule.gamma[step_index]  # pyright: ignore[reportAttributeAccessIssue] (this will be accessible due to the check above)

        t_hat = sigma_tm * (1 + gamma)
        eps_scale = self.noise_scale * torch.sqrt(t_hat**2 - sigma_tm**2)
        dt = sigma_t - t_hat

        if total_steps is None:
            total_steps = len(schedule.sigma_t)  # pyright: ignore[reportAttributeAccessIssue] (this will be accessible due to the check above)

        return StepContext(
            step_index=step_index,
            total_steps=total_steps,
            t=t_hat,
            dt=dt,
            noise_scale=eps_scale,
        )

    def _apply_scaler_guidance(
        self,
        scaler: StepScalerProtocol,
        x_hat_0_working_frame: Float[torch.Tensor, "*batch n 3"],
        noisy_state: Float[torch.Tensor, "*batch n 3"],
        delta: torch.Tensor,
        context: StepContext,
        model_wrapper: FlowModelWrapper,
        align_transform: Mapping[str, torch.Tensor] | None,
        allow_gradients: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply guidance from scaler to the denoising direction."""
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
            x_hat_0_working_frame, scaler_context, model=model_wrapper
        )
        guidance_direction = torch.as_tensor(guidance_direction)
        loss = torch.as_tensor(loss)
        guidance_weight = scaler.guidance_strength(context)

        if align_transform is not None and allow_gradients:
            guidance_direction = apply_forward_transform(
                guidance_direction, align_transform, rotation_only=True
            )

        # Ensure proper broadcasting of guidance weight
        if isinstance(guidance_weight, torch.Tensor) and guidance_weight.ndim == 1:
            guidance_weight = guidance_weight.view(-1, 1, 1)

        result = delta + guidance_weight * guidance_direction
        return torch.as_tensor(result), loss

    def step(
        self,
        state: Float[torch.Tensor, "*batch num_points 3"],
        model_wrapper: FlowModelWrapper,
        context: StepContext,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput:
        """Take EDM diffusion step with optional guidance.

        Based on Supplemental Algorithm 18 from the AlphaFold3 paper.

        Parameters
        ----------
        state
            Current noisy coordinates.
        model_wrapper
            Model wrapper for x̂₀ prediction.
        context
            Step context with t, dt, noise_scale, and optionally reward info.
        scaler
            Optional step scaler for computing guidance from rewards.
        features
            Additional model features/inputs.

        Returns
        -------
        SamplerStepOutput
            Output containing updated state, denoised prediction (x̂₀), and loss.
        """
        self.check_context(context)

        t_hat = context.t_effective
        dt = context.dt
        eps_scale = context.noise_scale
        allow_gradients = True if scaler and getattr(scaler, "requires_gradients", False) else False

        centroid = einx.mean("... [n] c", state)
        state_centered = einx.subtract("... n c, ... c -> ... n c", state, centroid)

        transform = (
            create_random_transform(state_centered, center_before_rotation=False)
            if self.augmentation
            else None
        )

        maybe_augmented_state = (
            apply_forward_transform(state_centered, transform, rotation_only=False)
            if transform is not None
            else state_centered
        )

        # Store eps separately for proper frame transformation
        # eps_scale will be float if check_context passed
        eps = torch.randn_like(maybe_augmented_state) * eps_scale  # pyright: ignore[reportOperatorIssue]
        noisy_state = maybe_augmented_state + eps
        noisy_state = torch.as_tensor(noisy_state).detach().requires_grad_(allow_gradients)

        # t_hat will be float if check_context passed
        x_hat_0 = model_wrapper.step(noisy_state, t_hat, features=features)

        # Default: work in augmented frame
        x_hat_0_working_frame = x_hat_0
        noisy_state_working_frame = noisy_state
        align_transform = None

        if self.align_to_input and features is not None:
            align_mask = (
                torch.as_tensor(context.metadata.get("mask"))
                if context.metadata and context.metadata.get("mask") is not None
                else None
            )
            x_hat_0_working_frame, align_transform = align_to_reference_frame(
                torch.as_tensor(x_hat_0),
                torch.as_tensor(features.x_init),
                mask=align_mask,
                weights=align_mask,
                allow_gradients=allow_gradients,
            )
            _, _, noisy_state_working_frame = transform_coords_and_noise_to_frame(
                torch.as_tensor(maybe_augmented_state), torch.as_tensor(eps), align_transform
            )

        if self.alignment_reverse_diffusion:
            noisy_state_working_frame = weighted_rigid_align_differentiable(
                torch.as_tensor(noisy_state_working_frame),
                torch.as_tensor(x_hat_0_working_frame),
                weights=torch.ones_like(torch.as_tensor(x_hat_0_working_frame)[..., 0]),
                mask=torch.ones_like(torch.as_tensor(x_hat_0_working_frame)[..., 0]),
                allow_gradients=False,
            )

        x_hat_0_working_frame_t = torch.as_tensor(x_hat_0_working_frame)
        noisy_state_working_frame_t = torch.as_tensor(noisy_state_working_frame)
        delta = torch.as_tensor((noisy_state_working_frame_t - x_hat_0_working_frame_t) / t_hat)

        loss = None
        if scaler is not None:
            delta, loss = self._apply_scaler_guidance(
                scaler=scaler,
                x_hat_0_working_frame=x_hat_0_working_frame_t,
                noisy_state=noisy_state,
                delta=torch.as_tensor(delta),
                context=context,
                model_wrapper=model_wrapper,
                align_transform=align_transform,
                allow_gradients=allow_gradients,
            )

        # Euler step: x_{t-1} = x_t + step_scale * dt * delta
        # pyright sees dt as float | None, but it will be float if check_context passed
        next_state = noisy_state_working_frame_t + self.step_scale * dt * delta  # pyright: ignore[reportOperatorIssue]

        return SamplerStepOutput(
            state=next_state,
            denoised=x_hat_0_working_frame_t,
            loss=loss,
        )
