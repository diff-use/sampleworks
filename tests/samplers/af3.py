"""Tests for the AF3 sampler."""

import pytest
import torch
from sampleworks.core.samplers.af3 import sample_af3_step


class TestAF3Sampler:
    """Test the AF3 predictor-corrector sampler."""

    @pytest.fixture
    def basic_coords(self):
        """Create basic coordinate tensors for testing."""
        batch_size = 2
        num_points = 10
        noisy_coords = torch.randn(batch_size, num_points, 3)
        denoised_coords = torch.randn(batch_size, num_points, 3)
        return noisy_coords, denoised_coords

    @pytest.fixture
    def basic_params(self):
        """Create basic parameters for sampling."""
        return {
            "t_hat": 1.0,
            "sigma_t": 0.5,
            "step_scale": 1.0,
            "step_size": 0.1,
            "gradient_normalization": False,
        }

    def test_sample_exact_calculation(self, basic_params):
        """Test the exact calculation of the AF3 sampling step."""
        noisy_coords = torch.tensor([[[0.0, 0.0, 0.0]]])
        denoised_coords = torch.tensor([[[1.0, 1.0, 1.0]]])

        result = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=None,
            **basic_params,
        )

        expected = (
            noisy_coords
            + basic_params["step_scale"]
            * ((denoised_coords - noisy_coords) / basic_params["sigma_t"])
            * basic_params["step_size"]
        )

        assert torch.allclose(result, expected)

    def test_sample_without_guidance(self, basic_coords, basic_params):
        """Test sampling without guidance direction."""
        noisy_coords, denoised_coords = basic_coords

        result = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=None,
            **basic_params,
        )

        # Result should be closer to denoised_coords than noisy_coords is
        dist_noisy = torch.linalg.vector_norm(noisy_coords - denoised_coords)
        dist_result = torch.linalg.vector_norm(result - denoised_coords)
        assert dist_result < dist_noisy

    def test_sample_with_guidance(self, basic_coords, basic_params):
        """Test sampling with guidance direction."""
        noisy_coords, denoised_coords = basic_coords
        guidance_direction = torch.randn_like(noisy_coords)

        result = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=guidance_direction,
            **basic_params,
        )

        # Result should be different from unguided result
        unguided_result = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=None,
            **basic_params,
        )
        assert not torch.allclose(result, unguided_result)

    def test_sample_with_gradient_normalization(self, basic_coords, basic_params):
        """Test sampling with gradient normalization enabled."""
        noisy_coords, denoised_coords = basic_coords
        guidance_direction = torch.randn_like(noisy_coords)

        params_with_norm = basic_params.copy()
        params_with_norm["gradient_normalization"] = True

        result_normalized = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=guidance_direction,
            **params_with_norm,
        )

        params_without_norm = basic_params.copy()
        params_without_norm["gradient_normalization"] = False

        result_unnormalized = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=guidance_direction,
            **params_without_norm,
        )

        # Results should be different when normalization is applied
        assert not torch.allclose(result_normalized, result_unnormalized)

    def test_sample_step_scale_effect(self, basic_coords, basic_params):
        """Test that step_scale parameter affects the result correctly."""
        noisy_coords, denoised_coords = basic_coords

        params_scale_1 = basic_params.copy()
        params_scale_1["step_scale"] = 1.0

        result_scale_1 = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=None,
            **params_scale_1,
        )

        params_scale_2 = basic_params.copy()
        params_scale_2["step_scale"] = 2.0

        result_scale_2 = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=None,
            **params_scale_2,
        )

        # Larger step scale should go further towards denoised_coords
        diff_1 = torch.norm(result_scale_1 - denoised_coords)
        diff_2 = torch.norm(result_scale_2 - denoised_coords)
        assert diff_2 < diff_1

    def test_sample_step_size_effect(self, basic_coords, basic_params):
        """Test that step_size parameter affects guidance strength."""
        noisy_coords, denoised_coords = basic_coords
        guidance_direction = torch.randn_like(noisy_coords)

        params_size_small = basic_params.copy()
        params_size_small["step_size"] = 0.1

        result_small = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=guidance_direction,
            **params_size_small,
        )

        params_size_large = basic_params.copy()
        params_size_large["step_size"] = 0.5

        result_large = sample_af3_step(
            noisy_coords=noisy_coords,
            denoised_coords=denoised_coords,
            guidance_direction=guidance_direction,
            **params_size_large,
        )

        # Results should be different with different step sizes
        assert not torch.allclose(result_small, result_large)
