from typing import cast

import torch
from jaxtyping import Float
from torch.linalg import vector_norm


def sample_af3_step(
    noisy_coords: Float[torch.Tensor, "*batch num_points 3"],
    denoised_coords: Float[torch.Tensor, "*batch num_points 3"],
    t_hat: float,
    sigma_t: float,
    step_scale: float,
    step_size: float,
    gradient_normalization: bool,
    guidance_direction: Float[torch.Tensor, "*batch num_points 3"] | None = None,
) -> Float[torch.Tensor, "*batch num_points 3"]:
    """Take a step with the AF3 predictor-corrector sampler, optionally with guidance.
    (based on Supplemental Algorithm 18)

    TODO: JAX compatibility?

    Parameters
    ----------
    noisy_coords: Float[torch.Tensor, "*batch num_points 3"]
        Noisy coordinates.
    denoised_coords: Float[torch.Tensor, "*batch num_points 3"]
        Denoised coordinates predicted by the model.
    t_hat: float
        Noise level.
    sigma_t: float
        Next noise level.
    step_scale: float
        Scale for the Euler step.
    step_size: float
        Size of the guidance step to take.
    gradient_normalization: bool
        Whether to normalize the guidance direction to have the same norm as the
        model-predicted update.
    guidance_direction: Float[torch.Tensor, "*batch num_points 3"] | None
        Guidance direction for training-free guidance. Must be in the same frame as
        the noisy and denoised coordinates.
    """
    dt = sigma_t - t_hat

    delta = (noisy_coords - denoised_coords) / t_hat

    if guidance_direction is not None:
        if gradient_normalization:
            grad_norm = vector_norm(guidance_direction, dim=(1, 2), keepdim=True)
            delta_norm = vector_norm(delta, dim=(1, 2), keepdim=True)
            guidance_direction = guidance_direction * delta_norm / (grad_norm + 1e-8)

        delta = delta + step_size * cast(torch.Tensor, guidance_direction)

    return noisy_coords + step_scale * dt * delta
