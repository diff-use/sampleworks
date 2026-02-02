"""Mock implementations of trajectory scaler protocols for testing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from jaxtyping import Float
from sampleworks.utils.framework_utils import Array
from torch import Tensor


if TYPE_CHECKING:
    from sampleworks.core.rewards.real_space_density import RealSpaceRewardFunction
    from sampleworks.core.samplers.protocol import TrajectorySampler
    from sampleworks.core.scalers.protocol import StepScalerProtocol
    from sampleworks.models.protocol import FlowModelWrapper


@dataclass(frozen=True)
class MockGuidanceOutput:
    """Simplified GuidanceOutput for testing without import issues."""

    structure: dict
    final_state: Float[Array, "*batch atoms 3"]
    trajectory: Sequence[Float[Array, ...]] | None = None
    losses: list[float | None] | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)


class MockTrajectoryScaler:
    """Mock TrajectoryScaler that satisfies TrajectoryScalerProtocol.

    Provides simple sampling loop for testing without real guidance.

    Parameters
    ----------
    num_steps
        Number of steps in the trajectory. Default is 10.
    ensemble_size
        Number of structures in the ensemble. Default is 1.
    """

    def __init__(self, num_steps: int = 10, ensemble_size: int = 1):
        self.num_steps = num_steps
        self.ensemble_size = ensemble_size
        self._sample_call_count = 0

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol[FlowModelWrapper],
        reward: RealSpaceRewardFunction,
        num_particles: int = 1,
    ) -> MockGuidanceOutput:
        """Run sampling trajectory with optional guidance.

        Parameters
        ----------
        structure
            Input atomworks structure dictionary.
        model
            FlowModelWrapper to use for sampling.
        sampler
            TrajectorySampler for trajectory generation.
        step_scaler
            StepScalerProtocol for per-step guidance.
        reward
            Reward function for guidance.
        num_particles
            Number of particles (ignored in mock).

        Returns
        -------
        MockGuidanceOutput
            Output with final state, trajectory, and losses.
        """
        self._sample_call_count += 1

        features = model.featurize(structure)
        coords: Float[Tensor, "*batch atoms 3"] = model.initialize_from_prior(
            batch_size=self.ensemble_size,
            features=features,
        )

        trajectory: list[Tensor] = []
        losses: list[float | None] = []

        schedule = sampler.compute_schedule(num_steps=self.num_steps)

        for i in range(self.num_steps):
            context = sampler.get_context_for_step(i, schedule)

            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=step_scaler,
                features=features,
            )

            coords = step_output.state
            trajectory.append(coords.clone().cpu())

            if step_output.loss is not None:
                losses.append(step_output.loss.mean().item())
            else:
                losses.append(None)

        return MockGuidanceOutput(
            structure=structure,
            final_state=coords,
            trajectory=trajectory,
            losses=losses,
            metadata={"mock": True},
        )
