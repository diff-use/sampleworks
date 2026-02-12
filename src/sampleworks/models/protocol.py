"""
Dependency injection protocols for generative model wrappers.

Allows different model wrappers to be used interchangeably in sampling pipelines.
"""

from dataclasses import dataclass
from typing import (
    Generic,
    Protocol,
    runtime_checkable,
    TypeVar,
)

from jaxtyping import Float

from sampleworks.utils.framework_utils import Array


C = TypeVar("C")
StructureModelOutputT = TypeVar("StructureModelOutputT", covariant=True)
FlowOrEnergyBasedModelOutputT = TypeVar("FlowOrEnergyBasedModelOutputT")


@dataclass
class GenerativeModelInput(Generic[C]):  # noqa: UP046 (for Python 3.11 compatibility)
    """
    Container for inputs to generative models.

    The x_init tensor is typically sampled from a prior distribution,
    with shape determined by the input data (e.g., sequence length
    determines atom count).

    Attributes:
        x_init: Initial structure coordinates, shape (*batch, atoms, 3).
                This can be a reference structure (e.g. for alignment during sampling)
                or a noisy sample from a prior distribution. This should have the proper
                shape expected for the given ensemble being sampled, e.g. (4, atoms, 3) for
                ensemble size 4.
        conditioning: Model-specific conditioning features, or None.
    """

    # TODO: make x_init more general (not just Float),
    # relate this to StateT in Sampler protocol?
    x_init: Float[Array, "*batch atoms 3"]
    conditioning: C | None


@runtime_checkable
class StructureModelWrapper(Protocol[C, StructureModelOutputT]):
    """
    Direct structure prediction from features.

    Maps input features to atomic coordinates in a single forward pass.
    Internal iteration (e.g., AlphaFold2 recycling) is encapsulated.

    Examples:
        - AlphaFold2 (JAX)
        - ESMFold (PyTorch)
        - OpenFold (PyTorch/JAX)
    """

    def featurize(self, structure: dict) -> GenerativeModelInput[C]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)

        Returns
        -------
        GenerativeModelInput
            Model features.
        """
        ...

    def step(self, features: GenerativeModelInput[C]) -> StructureModelOutputT:
        """
        Perform a single pass through the model to obtain output, which can then be
        passed into a scaler for optimizing fit with observables.

        Parameters
        ----------
        features: GenerativeModelInput
            Model features as returned by `featurize`.

        Returns
        -------
        StructureModelOutputT
            Model outputs.
        """
        ...


@runtime_checkable
class FlowModelWrapper(Protocol[C, FlowOrEnergyBasedModelOutputT]):
    """
    Flow-matching and diffusion model wrapper.

    Iteratively denoises samples from a prior distribution to generate
    structures. Each step conditions on a timestep t ∈ [0, 1] representing
    the noise level.

    Examples:
        - Boltz-1/Boltz-2 (PyTorch)
        - AlphaFold3 / Protenix (JAX/PyTorch)
        - Chai-1 (PyTorch)
    """

    def featurize(self, structure: dict) -> GenerativeModelInput[C]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        **kwargs: dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        GenerativeModelInput
            Model features.
        """
        ...

    def step(
        self,
        x_t: FlowOrEnergyBasedModelOutputT,
        t: Float[Array, "*batch"],
        *,
        features: GenerativeModelInput[C] | None = None,
    ) -> FlowOrEnergyBasedModelOutputT:
        r"""
        Perform denoising at given timestep/noise level.
        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t: FlowOrEnergyBasedModelOutputT
            Noisy structure at timestep :math:`t`.
        t: Float[Array, "*batch"]
            Current timestep/noise level.
        features: GenerativeModelInput | None, optional
            Model features as returned by `featurize`.

        Returns
        -------
        FlowOrEnergyBasedModelOutputT
            Predicted clean sample or predicted noise.
        """
        ...

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[C] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> FlowOrEnergyBasedModelOutputT:
        """Create a noisy version of a state at given noise level.

        Parameters
        ----------
        batch_size: int
            Number of samples to generate.
        features: GenerativeModelInput | None, optional
            Model features as returned by `featurize`. Useful for determining shape, etc. for
            the state.
        shape: tuple[int, ...] | None, optional
            Explicit shape of the generated state, if features is None or does not
            provide shape info.

        Returns
        -------
        FlowOrEnergyBasedModelOutputT
            Noisy output.
        """
        ...


@runtime_checkable
class EnergyBasedModelWrapper(Protocol[C, FlowOrEnergyBasedModelOutputT]):
    """
    Energy-based model wrapper.

    Generates structures by iteratively minimizing an implicit energy function.
    Unlike flow models, steps are not conditioned on an explicit timestep.
    Sampling typically uses Langevin dynamics or similar MCMC methods.

    Examples:
        - Distributional Graphormer (PyTorch)
        - Equilibrium Matching
    """

    def featurize(self, structure: dict) -> GenerativeModelInput[C]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)

        Returns
        -------
        GenerativeModelInput
            Model features.
        """
        ...

    def step(
        self,
        x: FlowOrEnergyBasedModelOutputT,
        *,
        features: GenerativeModelInput[C] | None = None,
    ) -> FlowOrEnergyBasedModelOutputT:
        """
        Perform one step of energy-based sampling.

        Parameters
        ----------
        x: FlowOrEnergyBasedModelOutputT
            Current structure state.
        features: GenerativeModelInput | None, optional
            Model features as returned by `featurize`.

        Returns
        -------
        FlowOrEnergyBasedModelOutputT
            Updated structure after one sampling step.
        """
        ...
