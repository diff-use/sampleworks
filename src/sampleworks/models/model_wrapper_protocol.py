from typing import Any, Protocol

from jaxtyping import Array, Float


class ModelWrapper(Protocol):
    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Model features.
        """
        ...

    def step(
        self, features: dict[str, Any], grad_needed: bool = False, **kwargs
    ) -> dict[str, Any]:
        """
        Perform a single pass through the model to obtain output, which can then be
        passed into a scaler for optimizing fit with observables.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by `featurize`.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Model outputs.
        """
        ...


class DiffusionModelWrapper(ModelWrapper):
    def get_noise_schedule(self) -> dict[str, Float[Array, "..."]]:
        """
        Return the full noise schedule with semantic keys.

        Examples:
        - {"sigma": [...], "timesteps": [...]}
        - {"alpha": [...], "sigma": [...], "betas": [...]}
        - Model-specific keys depending on parameterization.

        Returns
        -------
        dict[str, Float[Array, "..."]]
            Noise schedule arrays.
        """
        ...

    def get_timestep_scaling(self, timestep: float) -> dict[str, float]:
        """
        Return scaling constants.

        For v-parameterization: returns {c_skip, c_out, c_in, c_noise}
        For epsilon-parameterization: returns {alpha, sigma}
        For other parameterizations: return model-specific scalings.

        Parameters
        ----------
        timestep : float
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
        timestep: float,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform one denoising step at given timestep/noise level.
        Returns predicted clean sample or predicted noise depending on
        model parameterization.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by `featurize`.
        timestep : float
            Current timestep/noise level.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Predicted clean sample or predicted noise.
        """
        ...

    def initialize_from_noise(
        self, structure: dict, noise_level: float, **kwargs
    ) -> dict[str, Any]:
        """Create a noisy version of structure at given noise level.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        noise_level : float
            Desired noise level/timestep to initialize at.
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        dict[str, Any]
            Noisy structure.
        """
        ...
