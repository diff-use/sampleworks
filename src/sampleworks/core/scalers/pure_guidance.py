"""
Pure diffusion guidance, as described in [DriftLite](http://arxiv.org/abs/2509.21655)

TODO: Make this more generalizable, a reasonable protocol to implement
Currently this only works with the implemented wrappers and isn't extensible
"""

from typing import Any, cast

import einx
import numpy as np
import torch
from biotite.structure import stack
from loguru import logger
from tqdm import tqdm

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOMIC_NUM_TO_ELEMENT,
)
from sampleworks.core.rewards.real_space_density import RewardFunction
from sampleworks.models.model_wrapper_protocol import DiffusionModelWrapper
from sampleworks.utils.atom_array_utils import filter_to_common_atoms
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    create_random_transform,
    weighted_rigid_align_differentiable,
)
from sampleworks.utils.imports import check_any_model_available


check_any_model_available()


class PureGuidance:
    """
    Pure guidance scaler.
    """

    def __init__(
        self,
        model_wrapper: DiffusionModelWrapper,
        reward_function: RewardFunction,
    ):
        """Creates a PureGuidance Scaler for guiding a diffusion model with a
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
        """Run pure guidance (Training-free guidance or Diffusion Posterior Sampling)
        using the provided ModelWrapper

        Parameters
        ----------
        structure: dict
            Atomworks parsed structure.
        **kwargs: dict
            Additional keyword arguments for pure guidance.

            - step_scale: float, optional
                Scale for the model's step size (default: 1.5)

            - ensemble_size: int, optional
                Size of ensemble to generate (default: 1)

            - step_size: float, optional
                Gradient step size for guidance (default: 0.1)

            - gradient_normalization: bool, optional
                Whether to normalize/clip gradients (default: False)

            - use_tweedie: bool, optional
                If True, use Tweedie's formula (gradient on x̂_0 only).
                    Enables augmentation and alignment. If False, use full
                    backprop through model (default: False)

            - augmentation: bool, optional
                Enable data augmentation in denoise step (default: False)

            - align_to_input: bool, optional
                Enable alignment to input in denoise step (default: True
                    for Tweedie mode, False for full backprop)

            - partial_diffusion_step: int, optional
                If provided, start diffusion from this timestep instead of 0.
                    (default: None). Will use the provided coordinates in structure
                    to initialize the noise at this timestep.

            - guidance_start: int, optional
                Diffusion step to start applying guidance (default: -1, meaning
                    guidance is applied from the beginning)

            - out_dir: str, optional
                Output directory for any featurization intermediate files
                (default: "test")

            - msa_path: dict | str | Path | None, optional
                MSA specification to be passed to model wrapper for featurization.
                Currently only used by RF3. # TODO: use kwargs better!!

            - alignment_reverse_diffusion: bool, optional
                Whether to perform alignment of noisy coords to denoised coords
                during reverse diffusion steps. This is relevant for doing Boltz-2-like
                alignment during diffusion. (default: False)

        Returns
        -------
        tuple[dict[str, Any], tuple[list[torch.Tensor], list[torch.Tensor]],
        list[float | None]]
            Structure dict with updated coordinates, tuple of
            (trajectory_denoised, trajectory_next_step) containing coordinate
            tensors at each step, list of losses at each step
        """
        # TODO: having defaults this deep can be a problem, remove
        step_scale = cast(float, kwargs.get("step_scale", 1.5))
        ensemble_size = cast(int, kwargs.get("ensemble_size", 1))
        step_size = cast(float, kwargs.get("step_size", 0.1))
        gradient_normalization = kwargs.get("gradient_normalization", False)
        use_tweedie = kwargs.get("use_tweedie", False)
        augmentation = kwargs.get("augmentation", False)
        align_to_input = kwargs.get("align_to_input", use_tweedie)
        partial_diffusion_step = cast(int, kwargs.get("partial_diffusion_step", 0))
        guidance_start = cast(int, kwargs.get("guidance_start", -1))
        out_dir = kwargs.get("out_dir", "test")
        msa_path = kwargs.get("msa_path", None)
        alignment_reverse_diffusion = kwargs.get("alignment_reverse_diffusion", not align_to_input)
        allow_alignment_gradients = not use_tweedie

        features = self.model_wrapper.featurize(structure, msa_path=msa_path, out_dir=out_dir)

        # Get coordinates from timestep
        coords = cast(
            torch.Tensor,
            self.model_wrapper.initialize_from_noise(
                structure,
                noise_level=partial_diffusion_step,
                ensemble_size=ensemble_size,
            ),
        )

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
        elements = einx.rearrange("n -> b n", torch.Tensor(elements), b=ensemble_size)
        b_factors = einx.rearrange(
            "n -> b n",
            torch.Tensor(atom_array.b_factor[reward_param_mask]),
            b=ensemble_size,
        )
        # TODO: properly handle occupancy values in structure processing
        # occupancies = einx.rearrange(
        #     "n -> b n",
        #     torch.Tensor(atom_array.occupancy[reward_param_mask]),
        #     b=ensemble_size,
        # )
        occupancies = torch.ones_like(cast(torch.Tensor, b_factors)) / ensemble_size

        input_coords = cast(
            torch.Tensor,
            einx.rearrange(
                "... -> e ...",
                torch.from_numpy(atom_array.coord).to(dtype=coords.dtype, device=coords.device),
                e=ensemble_size,
            ),
        )[..., reward_param_mask, :]

        # TODO: account for missing residues in mask
        mask_like = torch.ones_like(input_coords[..., 0])

        if input_coords.shape != coords.shape:
            raise ValueError(
                f"Input coordinates shape {input_coords.shape} does not match"
                f" initialized coordinates {coords.shape} shape."
            )

        # Detect model to structure atom count mismatch
        # s_ refers to structure indices, m_ refers to model indices.
        model_atom_array = features.get("model_atom_array")
        m_idx_t: torch.Tensor | None = None
        s_idx_t: torch.Tensor | None = None
        common_weights: torch.Tensor | None = None
        if model_atom_array is not None:
            struct_masked = atom_array[reward_param_mask]
            if len(model_atom_array) != len(struct_masked):
                (_, _), (m_idx, s_idx) = filter_to_common_atoms(
                    model_atom_array,
                    struct_masked,
                    normalize_ids=True,
                    return_indices=True,
                )
                m_idx_t = torch.from_numpy(m_idx).to(coords.device)
                s_idx_t = torch.from_numpy(s_idx).to(coords.device)
                common_weights = torch.ones(ensemble_size, len(m_idx), device=coords.device)
                logger.info(
                    f"Atom count mismatch: model={len(model_atom_array)}, "
                    f"structure={len(struct_masked)}, common={len(m_idx)}"
                )

        # Pre-allocate structure buffer for mismatch mapping
        denoised_struct_buf = input_coords.clone() if m_idx_t is not None else None
        last_denoised_model: torch.Tensor | None = None

        trajectory_denoised = []
        trajectory_next_step = []
        losses = []

        n_steps = len(cast(torch.Tensor, self.model_wrapper.get_noise_schedule()["sigma_t"]))

        for i in tqdm(range(partial_diffusion_step, n_steps)):
            apply_guidance = i > guidance_start

            centroid = einx.mean("... [n] c", coords)
            coords = einx.subtract("... n c, ... c -> ... n c", coords, centroid)

            timestep_scaling = self.model_wrapper.get_timestep_scaling(i)
            t_hat = timestep_scaling["t_hat"]
            sigma_t = timestep_scaling["sigma_t"]
            eps = timestep_scaling["eps_scale"] * torch.randn_like(coords)

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

            if not use_tweedie and apply_guidance:
                # Technically training free guidance requires grad on noisy_coords, not
                # on denoised, but DPS uses Tweedie's formula which has its limitations.
                # Maddipatla et al. 2025 use full backprop through model with grad on
                # noisy_coords.
                maybe_augmented_coords.detach_().requires_grad_(True)

            denoised_raw = self.model_wrapper.denoise_step(
                features,
                maybe_augmented_coords,
                timestep=i,
                grad_needed=(apply_guidance and not use_tweedie),
                # Provide precomputed t_hat and eps to allow us to calculate the
                # denoising direction properly
                t_hat=t_hat,
                eps=eps,
            )["atom_coords_denoised"]

            align_transform = None
            if m_idx_t is not None:
                # s_idx_t, common_weights, denoised_struct_buf always set with m_idx_t
                _s_idx = cast(torch.Tensor, s_idx_t)
                _cw = cast(torch.Tensor, common_weights)
                if align_to_input:
                    _, align_transform = weighted_rigid_align_differentiable(
                        denoised_raw[:, m_idx_t],
                        input_coords[:, _s_idx],
                        weights=_cw,
                        mask=_cw,
                        return_transforms=True,
                        allow_gradients=allow_alignment_gradients,
                    )
                    denoised_model = apply_forward_transform(
                        denoised_raw,
                        align_transform,
                        rotation_only=False,
                    )
                else:
                    denoised_model = denoised_raw
                last_denoised_model = denoised_model
                denoised_working_frame = cast(torch.Tensor, denoised_struct_buf).clone()
                denoised_working_frame[:, _s_idx] = denoised_model[:, m_idx_t]
            elif align_to_input:
                denoised_working_frame, align_transform = weighted_rigid_align_differentiable(
                    denoised_raw,
                    input_coords,
                    weights=mask_like,
                    mask=mask_like,
                    return_transforms=True,
                    allow_gradients=allow_alignment_gradients,
                )
            else:
                denoised_working_frame = denoised_raw

            # To align with FK steering, store denoised in working frame (aligned with the
            # input coords if align_to_input is True)
            trajectory_denoised.append(denoised_working_frame.clone().cpu())

            guidance_direction = None
            if apply_guidance:
                if use_tweedie:
                    # Using Tweedie's formula like DPS: gradient on denoised (x̂_0) only
                    denoised_for_grad = denoised_working_frame.detach().requires_grad_(True)
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
                    # Like Maddipatla et al. 2025 and training free guidance: gradient
                    # through model
                    loss = self.reward_function(
                        coordinates=denoised_working_frame,
                        elements=cast(torch.Tensor, elements),
                        b_factors=cast(torch.Tensor, b_factors),
                        occupancies=cast(torch.Tensor, occupancies),
                    )
                    loss.backward()

                    with torch.no_grad():
                        grad = maybe_augmented_coords.grad
                        assert grad is not None
                        guidance_direction = grad.clone()
                        maybe_augmented_coords.grad = None

                losses.append(loss.item())
            else:
                losses.append(None)

            with torch.no_grad():
                # Use the same eps as in the denoising step to properly compute
                # the denoising direction, and put in the denoised working frame
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
                noisy_coords = coords_in_working_frame + eps_in_working_frame

                if alignment_reverse_diffusion:
                    # Boltz aligns the noisy coords to the denoised coords at each step
                    # to improve stability.
                    noisy_coords = weighted_rigid_align_differentiable(
                        noisy_coords,
                        denoised_working_frame,
                        weights=mask_like,
                        mask=mask_like,
                        allow_gradients=False,
                    )

                dt = sigma_t - t_hat

                delta = (noisy_coords - denoised_working_frame) / t_hat

                if guidance_direction is not None:
                    # Make sure guidance direction is in working frame, since denoised
                    # may have been aligned. If using Tweedie/DPS, the grad is already
                    # in working frame because denoised_for_grad was aligned.
                    if not use_tweedie:
                        guidance_direction = (
                            apply_forward_transform(
                                guidance_direction, align_transform, rotation_only=True
                            )
                            if align_transform is not None
                            else guidance_direction
                        )
                    if gradient_normalization:
                        grad_norm = guidance_direction.norm(dim=(1, 2), keepdim=True)
                        delta_norm = delta.norm(dim=(1, 2), keepdim=True)
                        guidance_direction = guidance_direction * delta_norm / (grad_norm + 1e-8)
                    delta = delta + step_size * guidance_direction

                coords = noisy_coords + step_scale * dt * delta

                coords = coords.detach().clone()

            trajectory_next_step.append(coords.clone().cpu())

        # Save final structure
        if m_idx_t is not None and last_denoised_model is not None:
            # Concatenate atom array to match ensemble size
            final_aa = stack([model_atom_array] * ensemble_size)
            final_aa.coord = last_denoised_model.detach().cpu().numpy()  # type: ignore
            structure["asym_unit"] = final_aa
        else:
            # Concatenate atom array to match ensemble size
            atom_array = stack([atom_array] * ensemble_size)
            atom_array.coord[..., reward_param_mask, :] = coords.cpu().numpy()  # type: ignore (coord is NDArray)
            structure["asym_unit"] = atom_array

        return structure, (trajectory_denoised, trajectory_next_step), losses

    def __del__(self):
        logger.debug("Resetting model wrapper cached representations before deleting scaler.")
        if hasattr(self.model_wrapper, "cached_representations"):
            self.model_wrapper.cached_representations = {}  # pyright: ignore[reportAttributeAccessIssue] # see boltz.wrapper L201
        # if there are other reset tasks, they can be added here
        logger.debug("PureGuidance scaler deleted.")
