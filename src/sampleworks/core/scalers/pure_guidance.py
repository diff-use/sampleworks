"""
Pure diffusion guidance, as described in [DriftLite](http://arxiv.org/abs/2509.21655)

TODO: Make this more generalizable, a reasonable protocol to implement
Currently this only works with the Boltz1Wrapper or Boltz2Wrapper
"""

from typing import Any, cast

import torch
from tqdm import tqdm

from sampleworks.core.rewards.real_space_density import RewardFunction
from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper


class PureGuidance:
    """
    Pure guidance scaler.
    """

    def __init__(
        self,
        model_wrapper: Boltz1Wrapper | Boltz2Wrapper,
        reward_function: RewardFunction,
        **kwargs,
    ):
        """Creates a PureGuidance Scaler for guiding a diffusion model with a
        RewardFunction.

        Parameters
        ----------
        model_wrapper : DiffusionModelWrapper
            Diffusion model wrapper instance
        reward_function : RewardFunction
            Reward function to guide the diffusion process
        """
        self.model_wrapper = model_wrapper
        self.reward_function = reward_function

    def run_guidance(self, structure: dict, **kwargs: dict[str, Any]):
        features = self.model_wrapper.featurize(
            structure, out_dir=kwargs.get("out_dir", "boltz_test")
        )

        # Get coordinates from timestep 0
        noisy_coords = self.model_wrapper.initialize_from_noise(
            structure, noise_level=0
        )

        atom_coords_next = noisy_coords.clone()

        # TODO: this is not generalizable currently, figure this out
        atom_array = structure["asym_unit"]
        reward_param_mask = atom_array.occupancy > 0
        elements = atom_array.element[:, reward_param_mask]
        b_factors = atom_array.b_factor[:, reward_param_mask]
        occupancies = atom_array.occupancy[:, reward_param_mask]

        def step_size(i: int):
            return 0.1

        for i in tqdm(range(cast(int, kwargs.get("n_steps", 200)))):
            timestep_scaling = self.model_wrapper.get_timestep_scaling(i)

            denoised = self.model_wrapper.denoise_step(
                features,
                noisy_coords,
                timestep=i,
                # TODO: figure out how to handle kwargs in these guidance classes
            )["atom_coords_denoised"]

            denoised_over_sigma = (noisy_coords - denoised) / timestep_scaling["t_hat"]

            if i > cast(int, kwargs.get("guidance_start", -1)):
                with torch.set_grad_enabled(True):
                    loss = self.reward_function(
                        coordinates=denoised,
                        elements=elements,
                        b_factors=b_factors,
                        occupancies=occupancies,
                    )
                    loss.backward()

                    coords_grad = cast(torch.Tensor, denoised.grad).clone()

                # TODO: Add gradient normalization
                delta = denoised_over_sigma + step_size(i) * coords_grad

                atom_coords_next = (
                    noisy_coords
                    + self.model_wrapper.model.structure_module.step_scale
                    * delta
                    * (timestep_scaling["sigma_t"] - timestep_scaling["t_hat"])
                )
            else:
                atom_coords_next = (
                    noisy_coords
                    - self.model_wrapper.model.structure_module.step_scale
                    * denoised_over_sigma
                    * (timestep_scaling["sigma_t"] - timestep_scaling["t_hat"])
                )
                noisy_coords = atom_coords_next + timestep_scaling[
                    "eps_scale"
                ] * torch.randn_like(noisy_coords)

        structure["asym_unit"].coord = (
            atom_coords_next.detach().cpu().numpy()[:, reward_param_mask]
        )

        return structure
