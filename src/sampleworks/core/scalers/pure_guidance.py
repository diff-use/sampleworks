"""
Pure diffusion guidance, as described in [DriftLite](http://arxiv.org/abs/2509.21655)

TODO: Make this more generalizable, a reasonable protocol to implement
Currently this only works with the Boltz1Wrapper or Boltz2Wrapper
"""

from typing import Any, cast

import einx
import torch
from tqdm import tqdm

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOMIC_NUM_TO_ELEMENT,
)
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
            partial_diffusion_step : int, optional
                If provided, start diffusion from this timestep instead of 0.
                (default: None). Will use the provided coordinates in structure
                to initialize the noise at this timestep.

        Returns
        -------
        tuple[dict[str, Any], list[ArrayLike | torch.Tensor], list[float | None]]
            Structure dict with updated coordinates, trajectory of denoised
            coordinates, list of losses at each step
        """

        step_size = cast(float, kwargs.get("step_size", 0.1))
        gradient_normalization = kwargs.get("gradient_normalization", False)
        use_tweedie = kwargs.get("use_tweedie", False)
        augmentation = kwargs.get("augmentation", True)
        align_to_input = kwargs.get("align_to_input", use_tweedie)
        allow_alignment_gradients = True

        features = self.model_wrapper.featurize(
            structure, out_dir=kwargs.get("out_dir", "boltz_test")
        )

        # Get coordinates from timestep
        noisy_coords = self.model_wrapper.initialize_from_noise(
            structure, noise_level=cast(int, kwargs["partial_diffusion_step"])
        )
        ensemble_size = noisy_coords.shape[0]

        # TODO: this is not generalizable currently, figure this out
        atom_array = structure["asym_unit"]
        reward_param_mask = atom_array.occupancy > 0

        # TODO: jank way to get atomic numbers, fix this in real space density
        elements = [
            ATOMIC_NUM_TO_ELEMENT.index(
                e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()
            )
            for e in atom_array.element[reward_param_mask]
        ]
        elements = einx.rearrange("n -> b n", torch.Tensor(elements), b=ensemble_size)
        b_factors = einx.rearrange(
            "n -> b n",
            torch.Tensor(atom_array.b_factor[reward_param_mask]),
            b=ensemble_size,
        )
        occupancies = einx.rearrange(
            "n -> b n",
            torch.Tensor(atom_array.occupancy[reward_param_mask]),
            b=ensemble_size,
        )

        trajectory = []
        losses = []

        n_steps = self.model_wrapper.model.structure_module.num_sampling_steps
        guidance_start = cast(int, kwargs.get("guidance_start", -1))

        if use_tweedie:
            allow_alignment_gradients = False

        for i in tqdm(
            range(cast(int, kwargs.get("partial_diffusion_step", 0)), n_steps)
        ):
            apply_guidance = i > guidance_start

            if not use_tweedie and apply_guidance:
                # Technically training free guidance requires grad on noisy_coords, not
                # on denoised, but DPS uses Tweedie's formula which has its limitations.
                # Maddipatla et al. 2025 use full backprop through model with grad on
                # noisy_coords.
                noisy_coords.requires_grad_(True)

            timestep_scaling = self.model_wrapper.get_timestep_scaling(i)
            t_hat = timestep_scaling["t_hat"]
            sigma_t = timestep_scaling["sigma_t"]
            eps = timestep_scaling["eps_scale"] * torch.randn_like(noisy_coords)

            denoised = self.model_wrapper.denoise_step(
                features,
                noisy_coords,
                timestep=i,
                grad_needed=(apply_guidance and not use_tweedie),
                augmentation=augmentation,
                align_to_input=align_to_input,
                allow_alignment_gradients=allow_alignment_gradients,
                # Provide precomputed t_hat and eps to allow us to calculate the
                # denoising direction properly
                t_hat=t_hat,
                eps=eps,
            )["atom_coords_denoised"]

            guidance_direction = None

            if apply_guidance:
                if use_tweedie:
                    # Using Tweedie's formula like DPS: gradient on denoised (x̂_0) only
                    denoised_for_grad = denoised.detach().requires_grad_(True)
                    loss = self.reward_function(
                        coordinates=denoised_for_grad,
                        elements=cast(torch.Tensor, elements),
                        b_factors=cast(torch.Tensor, b_factors),
                        occupancies=cast(torch.Tensor, occupancies),
                    )
                    loss.backward()

                    with torch.no_grad():
                        grad = cast(torch.Tensor, denoised_for_grad.grad)
                        guidance_direction = grad.clone()
                else:
                    # Like Maddipatla et al. 2025: gradient through model
                    loss = self.reward_function(
                        coordinates=denoised,
                        elements=cast(torch.Tensor, elements),
                        b_factors=cast(torch.Tensor, b_factors),
                        occupancies=cast(torch.Tensor, occupancies),
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
                # Use the same eps as in the denoising step to properly compute
                # the denoising direction
                noisy_coords = noisy_coords + eps

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

                noisy_coords = noisy_coords.detach().clone()

            trajectory.append(denoised.detach().cpu().clone())

        # TODO: Handle ensemble here
        structure["asym_unit"].coord[:, reward_param_mask] = (
            noisy_coords.detach().cpu().numpy()
        )

        return structure, trajectory, losses
