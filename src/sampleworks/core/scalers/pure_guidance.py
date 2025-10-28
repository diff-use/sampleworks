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
        model_wrapper : Boltz1Wrapper | Boltz2Wrapper
            Diffusion model wrapper instance
        reward_function : RewardFunction
            Reward function to guide the diffusion process
        """
        self.model_wrapper = model_wrapper
        self.reward_function = reward_function

    def run_guidance(self, structure: dict, **kwargs: dict[str, Any]):
        """Run pure guidance (Diffusion Posterior Sampling) using the provided
        ModelWrapper

        Parameters
        ----------
        structure : dict
            Atomworks parsed structure.
        **kwargs : dict
            step_size : float, optional
                Gradient step size for guidance (default: 0.1)
            gradient_normalization : bool, optional
                Whether to normalize gradients (default: False)
            use_tweedie : bool, optional
                If True, use Tweedie's formula (gradient on x̂_0 only).
                Enables augmentation and alignment. If False, use full
                backprop through model (default: False)
            augmentation : bool, optional
                Enable data augmentation in denoise step (default: True
                for Tweedie mode, False for full backprop)
            align_to_input : bool, optional
                Enable alignment to input in denoise step (default: True
                for Tweedie mode, False for full backprop)

        Returns
        -------
        tuple[dict[str, Any], list[ArrayLike | torch.Tensor], list[float | None]]
            Structure dict with updated coordinates, trajectory of denoised
            coordinates, list of losses at each step
        """

        step_size = kwargs.get("step_size", 0.1)
        gradient_normalization = kwargs.get("gradient_normalization", False)
        use_tweedie = kwargs.get("use_tweedie", False)
        augmentation = kwargs.get("augmentation", True)
        align_to_input = kwargs.get("align_to_input", use_tweedie)
        allow_alignment_gradients = True

        features = self.model_wrapper.featurize(
            structure, out_dir=kwargs.get("out_dir", "boltz_test")
        )

        # Get coordinates from timestep 0
        noisy_coords = self.model_wrapper.initialize_from_noise(
            structure, noise_level=0
        )

        # TODO: this is not generalizable currently, figure this out
        atom_array = structure["asym_unit"]
        reward_param_mask = atom_array.occupancy > 0
        elements = atom_array.element[reward_param_mask]
        b_factors = atom_array.b_factor[reward_param_mask]
        occupancies = atom_array.occupancy[reward_param_mask]

        trajectory = []
        losses = []

        n_steps = cast(int, kwargs.get("n_steps", 200))
        guidance_start = cast(int, kwargs.get("guidance_start", -1))

        if use_tweedie:
            allow_alignment_gradients = False

        for i in tqdm(range(n_steps)):
            apply_guidance = i > guidance_start

            if not use_tweedie and apply_guidance:
                # Technically training free guidance requires grad on noisy_coords, not
                # on denoised, but DPS uses Tweedie's formula which has its limitations.
                # Maddipatla et al. 2025 use full backprop through model with grad on
                # noisy_coords.
                noisy_coords.requires_grad_(True)

            denoised = self.model_wrapper.denoise_step(
                features,
                noisy_coords,
                timestep=i,
                grad_needed=(apply_guidance and not use_tweedie),
                augmentation=augmentation,
                align_to_input=align_to_input,
                allow_alignment_gradients=allow_alignment_gradients,
            )["atom_coords_denoised"]

            guidance_direction = None

            if apply_guidance:
                if use_tweedie:
                    # Using Tweedie's formula like DPS: gradient on denoised (x̂_0) only
                    denoised_for_grad = denoised.detach().requires_grad_(True)
                    loss = self.reward_function(
                        coordinates=denoised_for_grad,
                        elements=elements,
                        b_factors=b_factors,
                        occupancies=occupancies,
                    )
                    loss.backward()

                    with torch.no_grad():
                        grad = cast(torch.Tensor, denoised_for_grad.grad)
                        guidance_direction = grad.clone()
                else:
                    # Like Maddipatla et al. 2025: gradient through model
                    loss = self.reward_function(
                        coordinates=denoised,
                        elements=elements,
                        b_factors=b_factors,
                        occupancies=occupancies,
                    )
                    loss.backward()

                    with torch.no_grad():
                        grad = cast(torch.Tensor, noisy_coords.grad)
                        guidance_direction = grad.clone()
                        noisy_coords.grad = None

                losses.append(loss.item())
            else:
                losses.append(None)

            with torch.no_grad():
                timestep_scaling = self.model_wrapper.get_timestep_scaling(i)
                t_hat = timestep_scaling["t_hat"]
                sigma_t = timestep_scaling["sigma_t"]
                dt = sigma_t - t_hat

                delta = (noisy_coords - denoised) / t_hat

                if guidance_direction is not None:
                    if gradient_normalization:
                        grad_norm = guidance_direction.norm(dim=(1, 2), keepdim=True)
                        delta_norm = delta.norm(dim=(1, 2), keepdim=True)
                        guidance_direction = (
                            guidance_direction * delta_norm / (grad_norm + 1e-8)
                        )
                    delta = delta + step_size * guidance_direction

                noisy_coords = (
                    noisy_coords
                    + self.model_wrapper.model.structure_module.step_scale * dt * delta
                )

                if i < n_steps - 1:
                    eps = timestep_scaling["eps_scale"] * torch.randn_like(noisy_coords)
                    noisy_coords = noisy_coords + eps

                noisy_coords = noisy_coords.detach().clone()

            trajectory.append(denoised.detach().cpu().clone())

        structure["asym_unit"].coord[reward_param_mask] = (
            noisy_coords.detach().cpu().numpy()[0, reward_param_mask]
        )

        return structure, trajectory, losses
