"""Unified sampler protocol for generative models and optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING, TypeVar

import torch
from jaxtyping import Float

from sampleworks.core.rewards.protocol import RewardFunctionProtocol
from sampleworks.models.protocol import (
    EnergyBasedModelWrapper,
    FlowModelWrapper,
    GenerativeModelInput,
    StructureModelWrapper,
)
from sampleworks.utils.framework_utils import Array


if TYPE_CHECKING:
    from sampleworks.core.rewards.protocol import RewardInputs
    from sampleworks.core.scalers.protocol import StepScalerProtocol


@dataclass(frozen=True, slots=True)
class SamplerSchedule:
    """Base schedule object returned by trajectory-based Sampler implementations.

    A schedule is created once via `compute_schedule(num_steps)` and then passed
    to `get_context_for_step(timestep, schedule)` to build StepContext.

    Implementations should:
    - be immutable dataclasses (frozen=True)
    - store any per-step arrays needed to compute step context
    - provide `as_dict()` for compatibility/serialization.
    """

    def as_dict(self) -> dict[str, Float[torch.Tensor, ...]]:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class StepContext:
    """Universal step context passed to all samplers.

    Contains metadata needed to execute a single sampling step.
    Different sampler types use different fields.
    """

    step_index: int
    total_steps: int | None = None

    # Trajectory parameters (diffusion/flow-matching)
    t: Float[Array, " batch"] | None = None  # Current time/noise level
    dt: Float[Array, " batch"] | None = None  # Step size in time
    noise_scale: Float[Array, " batch"] | None = None  # Scale of noise added to inputs (diffusion)

    # Optimization parameters
    learning_rate: Float[Array, " batch"] | None = None

    # Guidance parameters (optional, for guided sampling)
    reward: RewardFunctionProtocol | None = None
    reward_inputs: RewardInputs | None = None

    # Optional other metadata (could include velocity for momentum, etc.)
    metadata: dict[str, Any] | None = None

    @property
    def t_effective(self) -> Float[Array, " batch"]:
        """Effective time/noise level for this step.

        For trajectory samplers, returns t.
        Raises ValueError for optimization samplers.
        """
        if self.t is None:
            raise ValueError("StepContext has no time information (optimization sampler)")
        return self.t

    @property
    def is_trajectory(self) -> bool:
        """Whether this is a trajectory-based step (diffusion/flow)."""
        return self.t is not None

    @property
    def is_optimization(self) -> bool:
        """Whether this is an optimization-based step."""
        return self.learning_rate is not None

    @property
    def is_guided(self) -> bool:
        """Whether this step has reward information for guidance."""
        return self.reward is not None and self.reward_inputs is not None

    @property
    def is_final_step(self) -> bool:
        """Whether this is the final step."""
        if self.total_steps is None:
            return False
        return self.step_index >= self.total_steps - 1

    def with_reward(
        self,
        reward: RewardFunctionProtocol,
        reward_inputs: RewardInputs,
    ) -> StepContext:
        """Return a new context with reward information for guided sampling."""
        return StepContext(
            step_index=self.step_index,
            total_steps=self.total_steps,
            t=self.t,
            dt=self.dt,
            noise_scale=self.noise_scale,
            learning_rate=self.learning_rate,
            reward=reward,
            reward_inputs=reward_inputs,
            metadata=self.metadata,
        )

    def with_metadata(
        self,
        metadata: dict[str, Any],
    ) -> StepContext:
        """Return a new context with updated metadata."""
        return StepContext(
            step_index=self.step_index,
            total_steps=self.total_steps,
            t=self.t,
            dt=self.dt,
            noise_scale=self.noise_scale,
            learning_rate=self.learning_rate,
            reward=self.reward,
            reward_inputs=self.reward_inputs,
            metadata=metadata,
        )


StateT = TypeVar("StateT")
ScheduleT = TypeVar("ScheduleT", bound=SamplerSchedule)
ModelWrapperT = TypeVar(
    "ModelWrapperT",
    FlowModelWrapper,
    EnergyBasedModelWrapper,
    StructureModelWrapper,
    infer_variance=True,
)


@dataclass(frozen=True, slots=True)
class SamplerStepOutput[StateT]:
    """Output from a single sampler step.

    Encapsulates the updated state along with optional intermediate values
    useful for trajectory tracking and debugging.
    """

    state: StateT
    """Updated coordinates after the sampling step."""

    denoised: StateT | None = None
    """Denoised prediction (x̂₀) from this step, if available."""

    loss: Float[Array, " batch"] | None = None
    """Loss/reward value from scaler, if guidance was applied."""

    frame_transforms: Mapping[str, Float[torch.Tensor, ...]] | None = None
    """Alignment transform applied during step (rotation, translation)."""

    guidance_direction: Float[Array, "*batch atoms 3"] | None = None
    """Raw guidance direction from scaler before weighting."""


@runtime_checkable
class Sampler(Protocol[StateT, ModelWrapperT]):
    """Unified sampler protocol for all sampling strategies.

    The sampler is STATELESS - it receives all information via arguments
    and returns updated state.

    This protocol supports:
    - Trajectory-based methods (diffusion, flow-matching)
    - Optimization-based methods (gradient descent, equilibrium sampling)
    """

    def check_context(self, context: StepContext) -> None:
        """Validate that the provided StepContext is ready for step.

        Raises
        ------
        ValueError
            If the context is incompatible with this sampler.
        """
        ...

    def step(
        self,
        state: StateT,
        model_wrapper: ModelWrapperT,
        context: StepContext,
        *,
        scaler: StepScalerProtocol | None = None,
        features: GenerativeModelInput | None = None,
    ) -> SamplerStepOutput[StateT]:
        """Take a single step.

        Parameters
        ----------
        state : StateT
            Current state (coordinates, model inputs, latents, etc.)
        model_wrapper : FlowModelWrapper | EnergyBasedModelWrapper | StructureModelWrapper
            Model wrapper for performing model evaluations.
        context : StepContext
            Step metadata (time, learning rate, reward/reward_inputs, etc.)
        scaler : StepScalerProtocol | None
            Optional step scaler to modify sampler step using reward from context.
        features : GenerativeModelInput | None
            Additional model features/inputs, if any.

        Returns
        -------
        SamplerStepOutput[StateT]
            Output containing updated state, and optionally the denoised prediction, and
            reward/loss.
        """
        ...


@runtime_checkable
class TrajectorySampler(Sampler[StateT, FlowModelWrapper], Protocol):
    """Sampler that integrates a trajectory (diffusion, flow-matching).

    Extends the base Sampler with schedule management for time-based sampling.
    """

    def compute_schedule(self, num_steps: int) -> SamplerSchedule:
        """Compute the full sampling schedule.

        Parameters
        ----------
        num_steps: int
            Number of sampling steps.

        Returns
        -------
        SamplerSchedule
            Schedule object with sampler-specific arrays.
        """
        ...

    def check_schedule(self, schedule: SamplerSchedule) -> None:
        """Validate that the provided schedule is compatible with this sampler.

        Raises
        ------
        ValueError
            If the schedule is incompatible with this sampler.
        """
        ...

    def get_context_for_step(self, step_index: int, schedule: SamplerSchedule) -> StepContext:
        """Build StepContext from schedule for given step.

        Parameters
        ----------
        step_index: int
            Current timestep index (0-indexed).
        schedule: SamplerSchedule
            The schedule returned by compute_schedule().

        Returns
        -------
        StepContext
            Context with t, dt populated for this step.
        """
        ...
