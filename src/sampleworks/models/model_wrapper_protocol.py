"""
Dependency injection protocols for generative model wrappers.

Allows different model wrappers to be used interchangeably in sampling pipelines.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from jaxtyping import ArrayLike, Float
from torch import Tensor


class ModelWrapper(Protocol):
    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        **kwargs: dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Model features.
        """
        ...

    def step(self, features: dict[str, Any], grad_needed: bool = False, **kwargs) -> dict[str, Any]:
        """
        Perform a single pass through the model to obtain output, which can then be
        passed into a scaler for optimizing fit with observables.

        NOTE: For a diffusion model, this will be done in denoising step, and this
        function should be used for any conditioning or embedding (e.g. Pairformer for
        AF3 and its clones).

        Parameters
        ----------
        features: dict[str, Any]
            Model features as returned by `featurize`.
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Model outputs.
        """
        ...


class DiffusionModelWrapper(ModelWrapper, Protocol):
    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """
        Return the full noise schedule with semantic keys.

        Examples:
        - {"sigma": [...], "timesteps": [...]}
        - {"alpha": [...], "sigma": [...], "betas": [...]}
        - Model-specific keys depending on parameterization.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays.
        """
        ...

    def get_timestep_scaling(self, timestep: float | int) -> dict[str, float]:
        """
        Return scaling constants.

        For v-parameterization: returns {c_skip, c_out, c_in, c_noise}
        For epsilon-parameterization: returns {alpha, sigma}
        For other parameterizations: return model-specific scalings.

        Parameters
        ----------
        timestep: float | int
            Current timestep/noise level.

        Returns
        -------
        dict[str, float]
            Scaling constants.
        """
        ...

    def denoise_step(
        self,
        features: dict[str, Any],
        noisy_coords: Float[ArrayLike | Tensor, "..."],
        timestep: float | int,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform denoising at given timestep/noise level.
        Returns predicted clean sample or predicted noise depending on
        model parameterization.

        Parameters
        ----------
        features: dict[str, Any]
            Model features as returned by `featurize`.
        noisy_coords: Float[ArrayLike | Tensor, "..."]
            Noisy atom coordinates at current timestep.
        timestep: float | int
            Current timestep/noise level.
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Predicted clean sample or predicted noise.
        """
        ...

    def initialize_from_noise(
        self,
        structure: dict,
        noise_level: float | int,
        ensemble_size: int = 1,
        **kwargs,
    ) -> Float[ArrayLike | Tensor, "*batch _num_atoms 3"]:
        """Create a noisy version of structure's coordinates at given noise level.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        noise_level: float | int
            Desired noise level/timestep to initialize at.
        ensemble_size: int, optional
            Number of noisy samples to generate per input structure (default 1).
        **kwargs: dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        Float[ArrayLike | Tensor, "*batch _num_atoms 3"]
            Noisy structure coordinates.
        """
        ...
