"""
Feynman-Kaç Steering scaler implementation.

TODO: Make this more generalizable, a reasonable protocol to implement
Currently this only works with the implemented wrappers and isn't extensible
"""

from functools import partial
from typing import Any, cast

import einx
import numpy as np
import torch
import torch.nn.functional as F
from biotite.structure import stack
from tqdm import tqdm

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOMIC_NUM_TO_ELEMENT,
)
from sampleworks.core.rewards.real_space_density import RewardFunction
from sampleworks.models.model_wrapper_protocol import DiffusionModelWrapper
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    create_random_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.imports import check_any_model_available


check_any_model_available()


class FKSteering:
    """
    Feynman-Kac Steering scaler.
    """

    def __init__(
        self,
        model_wrapper: DiffusionModelWrapper,
        reward_function: RewardFunction,
    ):
        """Creates a Feynman-Kac Steering Scaler for guiding a diffusion model with a
        RewardFunction.

        Parameters
        ----------
        model_wrapper: DiffusionModelWrapper
            Diffusion model wrapper instance
        reward_function: RewardFunction
            Reward function to guide the diffusion process
        """
        self.model_wrapper = model_wrapper
        self.reward_function = reward_function

    def run_guidance(
        self, structure: dict, **kwargs: Any
    ) -> tuple[dict, tuple[list[Any], list[Any]], list[Any]]:
        """Run Feynman-Kac steering guidance using the provided ModelWrapper.

        NOTE: Because we are interested in sampling ensembles, not individual
        structures, from the guidance, our "particles" are ensembles of ensemble_size
        structures. Of course, if ensemble_size=1, then each particle is a single
        structure.

        Parameters
        ----------
        structure: dict
            Atomworks parsed structure.
        **kwargs: dict
            Additional keyword arguments for FK steering:

            - num_particles: int, optional
                Number of particles (replicas) for FK steering (default: 10)

            - fk_resampling_interval: int, optional
                How often to apply resampling (default: 1, every step)

            - fk_lambda: float, optional
                Weighting factor for resampling (default: 1.0)

            - num_gd_steps: int, optional
                Number of gradient descent steps on x0 (default: 0)

            - guidance_weight: float, optional
                Weight for gradient descent guidance (default: 0.0)

            - gradient_normalization: bool, optional
                Whether to normalize/clip gradients during guidance (default: False)
                NOTE: This is done BEFORE applying the guidance weight.

            - guidance_interval: int, optional
                How often to apply guidance (default: 1, every step)

            - guidance_start: int, optional
                Diffusion step to start applying guidance (default: -1, meaning
                    guidance is applied from the beginning)

            Inference arguments:

            - step_scale: float, optional
                Scale for the model's step size (default: 1.5)

            - ensemble_size: int, optional
                Size of ensemble to generate (default: 1)

            - augmentation: bool, optional
                Enable data augmentation in denoise step (default: True)

            - align_to_input: bool, optional
                Enable alignment to input in denoise step (default: False)

            - partial_diffusion_step: int, optional
                If provided, start diffusion from this timestep instead of 0.
                    (default: None). Will use the provided coordinates in structure
                    to initialize the noise at this timestep.

            - msa_path: dict | str | Path | None, optional
                MSA specification to be passed to model wrapper for featurization.
                Currently only used by RF3. # TODO: use kwargs better!!

            - out_dir: str, optional
                Output directory for any featurization intermediate files
                (default: "test")

            - alignment_reverse_diffusion: bool, optional
                Whether to perform alignment of noisy coords to denoised coords
                during reverse diffusion steps. This is relevant for doing Boltz-2-like
                alignment during diffusion. (default: False)

        Returns
        -------
        tuple[dict[str, Any], tuple[list[torch.Tensor], list[torch.Tensor]],
        list[float | None]]
            Structure dict with updated coordinates (ensemble), trajectories, losses
        """
        # FK Parameters
        num_particles = kwargs.get("num_particles", 10)
        fk_resampling_interval = kwargs.get("fk_resampling_interval", 1)
        fk_lambda = kwargs.get("fk_lambda", 1.0)

        # Guidance Parameters
        num_gd_steps = kwargs.get("num_gd_steps", 1)
        guidance_weight = kwargs.get("guidance_weight", 0.01)
        gradient_normalization = kwargs.get("gradient_normalization", False)
        guidance_interval = kwargs.get("guidance_interval", 1)
        guidance_start = cast(int, kwargs.get("guidance_start", -1))

        # Inference Parameters
        step_scale = kwargs.get("step_scale", 1.5)
        ensemble_size = kwargs.get("ensemble_size", 1)
        augmentation = kwargs.get("augmentation", True)
        align_to_input = kwargs.get("align_to_input", False)
        partial_diffusion_step = kwargs.get("partial_diffusion_step", 0)
        msa_path = kwargs.get("msa_path", None)
        out_dir = kwargs.get("out_dir", "test")
        alignment_reverse_diffusion = kwargs.get("alignment_reverse_diffusion", False)

        features = self.model_wrapper.featurize(structure, msa_path=msa_path, out_dir=out_dir)

        # Get coordinates from timestep
        # coords shape: (ensemble_size, N_atoms, 3)
        coords = cast(
            torch.Tensor,
            self.model_wrapper.initialize_from_noise(
                structure,
                noise_level=partial_diffusion_step,
                ensemble_size=ensemble_size,
            ),
        )

        # (num_particles, ensemble_size, N_atoms, 3)
        coords = cast(torch.Tensor, einx.rearrange("e n c -> p e n c", coords, p=num_particles))

        # TODO: this is not generalizable currently, figure this out
        if self.model_wrapper.__class__.__name__ == "ProtenixWrapper":
            atom_array = features["true_atom_array"]
        else:
            atom_array = structure["asym_unit"][0]
        occupancy_mask = atom_array.occupancy > 0
        nan_mask = ~np.any(np.isnan(atom_array.coord), axis=-1)
        reward_param_mask = occupancy_mask & nan_mask

        # TODO: jank way to get atomic numbers, fix this in real space density
        elements = [
            ATOMIC_NUM_TO_ELEMENT.index(e.title()) for e in atom_array.element[reward_param_mask]
        ]
        elements = cast(
            torch.Tensor,
            einx.rearrange("n -> p e n", torch.Tensor(elements), p=num_particles, e=ensemble_size),
        )
        b_factors = cast(
            torch.Tensor,
            einx.rearrange(
                "n -> p e n",
                torch.Tensor(atom_array.b_factor[reward_param_mask]),
                p=num_particles,
                e=ensemble_size,
            ),
        )
        # TODO: properly handle occupancy values in structure processing
        # occupancies = cast(
        #     torch.Tensor,
        #     einx.rearrange(
        #         "n -> p e n",
        #         torch.Tensor(atom_array.occupancy[reward_param_mask]),
        #         p=num_particles,
        #         e=ensemble_size,
        #     ),
        # )
        occupancies = torch.ones_like(cast(torch.Tensor, b_factors)) / ensemble_size

        # Pre-compute unique combinations for vmap compatibility
        # since torch.unique returns dynamic shape
        # NOTE: For now, this blocks grad w.r.t. B-factor
        unique_combinations, inverse_indices = self.reward_function.precompute_unique_combinations(
            elements[0].detach(),  # shape: (batch, n_atoms)
            b_factors[0].detach(),  # shape: (batch, n_atoms)
        )

        partial_reward_function = partial(
            self.reward_function,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        # (num_particles * ensemble_size, N_atoms, 3)
        input_coords = cast(
            torch.Tensor,
            einx.rearrange(
                "... -> b ...",
                torch.from_numpy(atom_array.coord).to(dtype=coords.dtype, device=coords.device),
                b=num_particles * ensemble_size,
            ),
        )[..., reward_param_mask, :]

        # TODO: account for missing residues in mask
        mask_like = torch.ones_like(input_coords[..., 0])

        # FK State variables
        energy_traj = torch.empty((num_particles, 0), device=coords.device)
        scaled_guidance_update = torch.zeros_like(coords)

        trajectory_denoised = []
        trajectory_next_step = []
        losses = []

        n_steps = len(cast(torch.Tensor, self.model_wrapper.get_noise_schedule()["sigma_t"]))

        pbar = tqdm(range(partial_diffusion_step, n_steps))
        for i in pbar:
            # (num_particles * ensemble_size, N_atoms, 3) for denoising,
            # will reshape for FK steering usage
            coords = cast(torch.Tensor, einx.rearrange("... n c -> (...) n c", coords))

            centroid = einx.mean("... [n] c", coords)
            coords = einx.subtract("... n c, ... c -> ... n c", coords, centroid)

            timestep_scaling = self.model_wrapper.get_timestep_scaling(i)
            t_hat = timestep_scaling["t_hat"]
            sigma_t = timestep_scaling["sigma_t"]
            eps_scale = timestep_scaling["eps_scale"]

            eps = eps_scale * torch.randn_like(coords)

            transform = (
                create_random_transform(coords, center_before_rotation=False)
                if augmentation
                else None
            )
            maybe_augmented_coords = cast(
                torch.Tensor,
                apply_forward_transform(coords, transform, rotation_only=False)
                if transform is not None
                else coords,
            )

            if num_gd_steps > 0:
                scaled_guidance_update = (
                    apply_forward_transform(
                        scaled_guidance_update.reshape(num_particles * ensemble_size, -1, 3),
                        transform,
                        rotation_only=True,
                    ).reshape(num_particles, ensemble_size, -1, 3)
                    if transform is not None
                    else scaled_guidance_update
                )

            denoised = cast(
                torch.Tensor,
                self.model_wrapper.denoise_step(
                    features,
                    maybe_augmented_coords,
                    timestep=i,
                    grad_needed=False,
                    # Provide precomputed t_hat and eps to allow us to calculate the
                    # denoising direction properly
                    t_hat=t_hat,
                    eps=eps,
                )["atom_coords_denoised"],
            )

            # do alignment before reshaping
            # align_transform will have shape (num_particles * ensemble_size, ...)
            align_transform = None
            denoised_working_frame = denoised
            if align_to_input:
                denoised_working_frame, align_transform = weighted_rigid_align_differentiable(
                    denoised,
                    input_coords,
                    weights=mask_like,
                    mask=mask_like,
                    return_transforms=True,
                    allow_gradients=False,
                )

            # we need coords and eps and scaled_guidance_update in working frame
            coords_in_working_frame = (
                apply_forward_transform(
                    maybe_augmented_coords, align_transform, rotation_only=False
                )
                if align_transform is not None
                else maybe_augmented_coords
            )
            eps_in_working_frame = (
                apply_forward_transform(eps, align_transform, rotation_only=True)
                if align_transform is not None
                else eps
            )
            if num_gd_steps > 0:
                scaled_guidance_update = (
                    apply_forward_transform(
                        scaled_guidance_update.reshape(num_particles * ensemble_size, -1, 3),
                        align_transform,
                        rotation_only=True,
                    ).reshape(num_particles, ensemble_size, -1, 3)
                    if align_transform is not None
                    else scaled_guidance_update
                )

            # reshape to be (num_particles, ensemble_size, N_atoms, 3)
            denoised_working_frame = denoised_working_frame.reshape(
                num_particles, ensemble_size, -1, 3
            )
            coords_in_working_frame = coords_in_working_frame.reshape(
                num_particles, ensemble_size, -1, 3
            )
            eps_in_working_frame = eps_in_working_frame.reshape(num_particles, ensemble_size, -1, 3)

            ### FK Resampling
            noise_var = eps_scale**2
            should_resample = (i % fk_resampling_interval == 0 and noise_var > 0) or (
                i == n_steps - 1
            )

            if should_resample:
                with torch.no_grad():
                    # energy shape: (num_particles,)
                    energy = cast(
                        torch.Tensor,
                        einx.vmap(
                            "p [e n c], p [e n], p [e n], p [e n] -> p",
                            denoised_working_frame,
                            elements,
                            b_factors,
                            occupancies,
                            op=partial_reward_function,
                        ),
                    )

                # energy_traj shape: (num_particles, steps)
                energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                if i == partial_diffusion_step:
                    log_G = -1 * energy
                else:
                    log_G = energy_traj[:, -2] - energy_traj[:, -1]

                ll_difference = torch.zeros_like(energy)
                if num_gd_steps > 0 and noise_var > 0:
                    # eps is the noise added. scaled_guidance_update is the shift.
                    # ll_diff = (eps^2 - (eps + shift)^2) / 2var
                    # Sum over dimensions to get total log likelihood difference
                    diff = (
                        eps_in_working_frame**2
                        - (eps_in_working_frame + scaled_guidance_update) ** 2
                    )

                    # shape (num_particles,)
                    ll_difference = diff.sum(dim=(-1, -2, -3)) / (2 * noise_var)

                # Resampling weights
                log_weights = ll_difference + fk_lambda * log_G
                resample_weights = F.softmax(log_weights, dim=0)

                # Resample indices
                # multinomial returns indices in [0, N-1]
                indices = torch.multinomial(
                    resample_weights, num_particles, replacement=True
                )  # shape: (num_particles,)

                coords_in_working_frame = coords_in_working_frame[indices]
                denoised_working_frame = denoised_working_frame[indices]
                eps_in_working_frame = eps_in_working_frame[indices]
                energy_traj = energy_traj[indices]
                scaled_guidance_update = scaled_guidance_update[indices]

            ### Guidance on x̂_0
            if num_gd_steps > 0 and i < n_steps - 1 and i >= guidance_start:
                guidance_update = torch.zeros_like(denoised_working_frame)
                delta_norm = guidance_update.clone()
                if gradient_normalization:
                    original_delta = (
                        coords_in_working_frame + eps_in_working_frame - denoised_working_frame
                    ) / t_hat
                    delta_norm = torch.linalg.norm(original_delta, dim=(-1, -2), keepdim=True)

                current_x0 = denoised_working_frame.detach().clone()

                # this is different from Boltz-*x, where they always do num_gd_steps of
                # guidance at every step, and the interval affects how many of those
                # steps each potential is applied. Here, we only do guidance every
                # guidance_interval steps.
                if i % guidance_interval == 0:
                    for _ in range(num_gd_steps):
                        current_x0.requires_grad_(True)
                        loss = cast(
                            torch.Tensor,
                            einx.vmap(
                                "p [e n c], p [e n], p [e n], p [e n] -> p",
                                current_x0,
                                elements,
                                b_factors,
                                occupancies,
                                op=partial_reward_function,
                            ),
                        ).mean()

                        (grad,) = torch.autograd.grad(loss, current_x0)

                        if gradient_normalization:
                            grad_norm = torch.linalg.norm(grad, dim=(-1, -2), keepdim=True)
                            grad = grad * (delta_norm / (grad_norm + 1e-8))

                        current_x0 = current_x0.detach() - guidance_weight * grad

                guidance_update = current_x0 - denoised_working_frame
                denoised_working_frame = current_x0

                dt = sigma_t - t_hat
                scaled_guidance_update = guidance_update * -1 * step_scale * dt / t_hat

            trajectory_denoised.append(denoised_working_frame.clone().cpu())
            losses.append(energy_traj[:, -1].mean().item() if energy_traj.shape[1] > 0 else 0.0)
            pbar.set_postfix({"loss": losses[-1]})

            with torch.no_grad():
                noisy_coords = coords_in_working_frame + eps_in_working_frame

                if alignment_reverse_diffusion:
                    # Boltz aligns the noisy coords to the denoised coords at each step
                    # to improve stability.

                    # TODO: need all this reshaping since
                    # weighted_rigid_align_differentiable only supports 1 batch dim
                    noisy_coords = weighted_rigid_align_differentiable(
                        noisy_coords.reshape(num_particles * ensemble_size, -1, 3),
                        denoised_working_frame.reshape(num_particles * ensemble_size, -1, 3),
                        weights=mask_like,
                        mask=mask_like,
                        allow_gradients=False,
                    ).reshape(num_particles, ensemble_size, -1, 3)

                dt = sigma_t - t_hat
                denoised_over_sigma = (noisy_coords - denoised_working_frame) / t_hat

                coords_next = noisy_coords + step_scale * dt * denoised_over_sigma

                coords = coords_next.detach().clone()

            trajectory_next_step.append(coords.clone().cpu())

        # Stack atom array to match ensemble size
        final_atom_array = stack([atom_array] * ensemble_size)

        # Get lowest energy particle
        min_energy_index = torch.argmin(energy_traj[:, -1])
        final_atom_array.coord[..., reward_param_mask, :] = (  # type: ignore[reportOptionalSubscript] coords will be subscriptable
            coords[min_energy_index].cpu().numpy()
        )

        structure["asym_unit"] = final_atom_array

        return structure, (trajectory_denoised, trajectory_next_step), losses
