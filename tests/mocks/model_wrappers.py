"""Mock implementations of model wrapper protocols for testing."""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import Array
from torch import Tensor


@dataclass(frozen=True)
class MockConditioning:
    """Simple conditioning for tests."""

    sequence_length: int
    num_atoms: int


class MockFlowModelWrapper:
    """Mock FlowModelWrapper that satisfies the protocol without a real model.

    This mock provides deterministic behavior for testing:
    - featurize() returns a GenerativeModelInput with random but reproducible data
    - step() returns a slightly denoised version of the input
    - initialize_from_prior() returns random noise

    Parameters
    ----------
    num_atoms
        Number of atoms for default structure.
    device
        PyTorch device for tensors.
    target
        Optional target tensor for convergence testing. When provided, step()
        outputs this target instead of zero.
    """

    def __init__(
        self,
        num_atoms: int = 100,
        device: torch.device | None = None,
        target: Tensor | None = None,
    ):
        self.num_atoms = num_atoms
        self.device = device or torch.device("cpu")
        self.target = target
        self._featurize_call_count = 0
        self._step_call_count = 0
        self._initialize_call_count = 0

    def featurize(self, structure: dict, **kwargs: Any) -> GenerativeModelInput[MockConditioning]:
        """Featurize a structure dict into model inputs."""
        self._featurize_call_count += 1
        x_init = torch.randn(1, self.num_atoms, 3, device=self.device)
        conditioning = MockConditioning(sequence_length=10, num_atoms=self.num_atoms)
        return GenerativeModelInput(x_init=x_init, conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "*batch atoms 3"],
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput[MockConditioning] | None = None,
    ) -> Float[Tensor, "*batch atoms 3"]:
        """Perform one denoising step.

        Returns the denoised prediction that is equal to target (or zero if no target).
        """
        if features is None:
            raise ValueError("features required for step()")
        self._step_call_count += 1

        if self.target is not None:
            target = self.target.to(x_t.device)
            if target.shape != x_t.shape:
                target = target.expand_as(x_t)
            return target

        return torch.zeros_like(x_t)

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[MockConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Initialize coordinates from prior distribution (random noise)."""
        self._initialize_call_count += 1
        if features is None and shape is None:
            raise ValueError("Either features or shape must be provided")
        if shape is not None:
            return torch.randn(batch_size, *shape, device=self.device)
        assert features is not None
        num_atoms = features.x_init.shape[-2]
        return torch.randn(batch_size, num_atoms, 3, device=self.device)
