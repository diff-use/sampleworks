"""Tests for frame transforms from sampleworks.utils.frame_transforms."""

import einx
import jax
import pytest
import torch
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    apply_inverse_transform,
    create_random_transform,
)


@pytest.mark.parametrize("shape", [(5, 5, 5, 10, 3), (5, 6, 10, 3), (5, 10, 3)])
class TestFrameTransforms:
    """Test frame transform utilities used by PureGuidance."""

    def test_transform_invertibility_torch(self, shape: tuple[int, ...]):
        """Test that forward→inverse transform is identity."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = apply_inverse_transform(augmented, transform)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_invertibility_jax(self, shape: tuple[int, ...]):
        """Test that forward→inverse transform is identity."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = jax.numpy.asarray(apply_inverse_transform(augmented, transform))

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_transform_preserves_shape_torch(self, shape: tuple[int, ...]):
        """Test that transforms preserve coordinate shapes."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        assert augmented.shape == coords.shape

        recovered = apply_inverse_transform(augmented, transform)
        assert recovered.shape == coords.shape

    def test_transform_preserves_shape_jax(self, shape: tuple[int, ...]):
        """Test that transforms preserve coordinate shapes."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        assert augmented.shape == coords.shape

        recovered = apply_inverse_transform(augmented, transform)
        assert recovered.shape == coords.shape

    def test_transform_rotation_matrix_orthogonal_torch(self, shape: tuple[int, ...]):
        """Test that rotation matrix is orthogonal."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        R = torch.tensor(transform["rotation"])
        batch_shape = shape[:-2]
        I = torch.eye(3, device=R.device, dtype=R.dtype)
        if batch_shape:
            I = I.expand(*batch_shape, 3, 3)

        RT = einx.rearrange("... i j -> ... j i", R)
        RRT = einx.dot("... i j, ... j k -> ... i k", R, RT)

        torch.testing.assert_close(RRT, I, rtol=1e-5, atol=1e-5)

    def test_transform_rotation_matrix_orthogonal_jax(self, shape: tuple[int, ...]):
        """Test that rotation matrix is orthogonal."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords)

        R = jax.numpy.asarray(transform["rotation"])
        batch_shape = shape[:-2]
        I = jax.numpy.eye(3, dtype=R.dtype)
        if batch_shape:
            I = jax.numpy.broadcast_to(I, (*batch_shape, 3, 3))

        RT = einx.rearrange("... i j -> ... j i", R)
        RRT = jax.numpy.asarray(einx.dot("... i j, ... j k -> ... i k", R, RT))

        assert jax.numpy.allclose(RRT, I, rtol=1e-5, atol=1e-5)

    def test_transform_rotation_matrix_determinant_torch(self, shape: tuple[int, ...]):
        """Test that rotation matrix has determinant 1."""
        coords = torch.randn(*shape)
        transform = create_random_transform(coords)

        R = torch.tensor(transform["rotation"])
        det = torch.det(R)

        torch.testing.assert_close(det, torch.ones(*shape[:-2], dtype=det.dtype))

    def test_transform_rotation_matrix_determinant_jax(self, shape: tuple[int, ...]):
        """Test that rotation matrix has determinant 1."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords)

        R = jax.numpy.asarray(transform["rotation"])
        det = jax.numpy.linalg.det(R)

        assert jax.numpy.allclose(det, 1.0, rtol=1e-5, atol=1e-5)

    def test_create_random_transform_with_centering_torch(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=True creates centered transform."""
        coords = torch.randn(*shape) + torch.tensor([10.0, 20.0, 30.0])
        transform = create_random_transform(coords, center_before_rotation=True)

        centroid = coords.mean(dim=-2)
        expected_t = -centroid

        torch.testing.assert_close(
            transform["translation"], expected_t, rtol=1e-5, atol=1e-5
        )

    def test_create_random_transform_with_centering_jax(self, shape: tuple[int, ...]):
        """Test that center_before_rotation=True creates centered transform."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape) + jax.numpy.array(
            [10.0, 20.0, 30.0]
        )
        transform = create_random_transform(coords, center_before_rotation=True)

        centroid = jax.numpy.mean(coords, axis=-2)
        expected_t = -centroid

        assert jax.numpy.allclose(
            jax.numpy.asarray(transform["translation"]),
            expected_t,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_create_random_transform_without_centering_torch(
        self, shape: tuple[int, ...]
    ):
        """Test that center_before_rotation=False creates random translation."""
        coords = torch.randn(shape)
        transform = create_random_transform(coords, center_before_rotation=False)

        assert transform["translation"].shape == (*shape[:-2], 3)
        assert transform["rotation"].shape == (*shape[:-2], 3, 3)

    def test_create_random_transform_without_centering_jax(
        self, shape: tuple[int, ...]
    ):
        """Test that center_before_rotation=False creates random translation."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape)
        transform = create_random_transform(coords, center_before_rotation=False)

        assert transform["translation"].shape == (*shape[:-2], 3)
        assert transform["rotation"].shape == (*shape[:-2], 3, 3)

    def test_takes_in_different_backends(self, shape: tuple[int, ...]):
        """Test that function works with both torch and jax arrays."""
        # Torch
        coords_torch = torch.randn(*shape)
        transform_torch = create_random_transform(coords_torch)
        assert isinstance(transform_torch["rotation"], torch.Tensor)
        assert isinstance(transform_torch["translation"], torch.Tensor)

        # JAX
        coords_jax = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
        transform_jax = create_random_transform(coords_jax)
        assert isinstance(transform_jax["rotation"], jax.numpy.ndarray)
        assert isinstance(transform_jax["translation"], jax.numpy.ndarray)
