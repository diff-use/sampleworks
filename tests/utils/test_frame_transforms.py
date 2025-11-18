"""Tests for frame transforms from sampleworks.utils.frame_transforms."""

import einx
import jax
import pytest
import torch
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    apply_inverse_transform,
    create_random_transform,
    weighted_rigid_align_differentiable,
)


try:
    from boltz.model.loss.diffusion import (
        weighted_rigid_align as boltz_weighted_rigid_align,
    )
except ImportError:
    pytest.skip("boltz not available", allow_module_level=True)


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

        R = transform["rotation"]
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

        R = transform["rotation"]
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

    def test_undo_transform_far_from_origin_torch(self, shape: tuple[int, ...]):
        """Test that transform and its inverse work far from the origin."""
        coords = torch.randn(*shape) + torch.tensor([1e6, -1e6, 1e6])
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = apply_inverse_transform(augmented, transform)

        torch.testing.assert_close(coords, recovered, rtol=1e-5, atol=1e-5)

    def test_undo_transform_far_from_origin_jax(self, shape: tuple[int, ...]):
        """Test that transform and its inverse work far from the origin."""
        coords = jax.random.normal(jax.random.PRNGKey(0), shape) + jax.numpy.array(
            [1e6, -1e6, 1e6]
        )
        transform = create_random_transform(coords)

        augmented = apply_forward_transform(coords, transform)
        recovered = jax.numpy.asarray(apply_inverse_transform(augmented, transform))

        assert jax.numpy.allclose(coords, recovered, rtol=1e-5, atol=1e-5)

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


@pytest.mark.parametrize("shape", [(1, 10, 3), (2, 50, 3), (15, 100, 3)])
class TestWeightedRigidAlign:
    """Test weighted rigid alignment function."""

    def test_align_rotated_line_torch(self, shape: tuple[int, ...]):
        """Test that a rotated line can be aligned back to the original."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points along x-axis
        t = torch.linspace(0, 1, num_points, dtype=torch.float32)
        original_line = torch.stack(
            [t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1
        )
        # Add batch dimension
        original_line = original_line.unsqueeze(0).expand(batch_size, -1, -1)

        # Create a randomly rotated and translated version
        transform = create_random_transform(original_line, center_before_rotation=False)
        rotated_line = apply_forward_transform(original_line, transform)

        # All points have equal weight
        weights = torch.ones(batch_size, num_points, dtype=torch.float32)
        mask = torch.ones(batch_size, num_points, dtype=torch.float32)

        # Align rotated_line back to original_line
        aligned_line = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            allow_gradients=True,
        )

        # The aligned line should match the original
        torch.testing.assert_close(aligned_line, original_line, rtol=5e-2, atol=1e-3)

    def test_align_rotated_line_jax(self, shape: tuple[int, ...]):
        """Test that a rotated line can be aligned back to the original."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points along x-axis
        t = jax.numpy.linspace(0, 1, num_points, dtype=jax.numpy.float32)
        original_line = jax.numpy.stack(
            [t, jax.numpy.zeros_like(t), jax.numpy.zeros_like(t)], axis=-1
        )
        # Add batch dimension
        original_line = jax.numpy.expand_dims(original_line, 0)
        original_line = jax.numpy.broadcast_to(
            original_line, (batch_size, num_points, 3)
        )

        # Create a randomly rotated and translated version
        transform = create_random_transform(
            original_line, center_before_rotation=False, key=1
        )
        rotated_line = apply_forward_transform(original_line, transform)

        # All points have equal weight
        weights = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)
        mask = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)

        # Align rotated_line back to original_line
        aligned_line = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            allow_gradients=True,
        )

        # The aligned line should match the original
        assert jax.numpy.allclose(aligned_line, original_line, rtol=5e-2, atol=1e-3)

    def test_alignment_with_transforms_torch(self, shape: tuple[int, ...]):
        """Test that return_transforms=True returns the alignment transform."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points
        t = torch.linspace(0, 1, num_points, dtype=torch.float32)
        original_line = torch.stack(
            [t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1
        )
        original_line = original_line.unsqueeze(0).expand(batch_size, -1, -1)

        # Rotate and translate
        transform = create_random_transform(original_line, center_before_rotation=False)
        rotated_line = apply_forward_transform(original_line, transform)

        weights = torch.ones(batch_size, num_points, dtype=torch.float32)
        mask = torch.ones(batch_size, num_points, dtype=torch.float32)

        # Get alignment with transforms
        aligned_line, align_transforms = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=True,
        )

        # Verify aligned line matches original
        torch.testing.assert_close(aligned_line, original_line, rtol=5e-2, atol=1e-3)

        # Verify transform is returned and has correct keys
        assert "rotation" in align_transforms
        assert "translation" in align_transforms
        assert align_transforms["rotation"].shape == (batch_size, 3, 3)
        assert align_transforms["translation"].shape == (batch_size, 3)

    def test_alignment_with_transforms_jax(self, shape: tuple[int, ...]):
        """Test that return_transforms=True returns the alignment transform."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points
        t = jax.numpy.linspace(0, 1, num_points, dtype=jax.numpy.float32)
        original_line = jax.numpy.stack(
            [t, jax.numpy.zeros_like(t), jax.numpy.zeros_like(t)], axis=-1
        )
        original_line = jax.numpy.expand_dims(original_line, 0)
        original_line = jax.numpy.broadcast_to(
            original_line, (batch_size, num_points, 3)
        )

        # Rotate and translate
        transform = create_random_transform(
            original_line, center_before_rotation=False, key=1
        )
        rotated_line = apply_forward_transform(original_line, transform)

        weights = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)
        mask = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)

        # Get alignment with transforms
        aligned_line, align_transforms = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=True,
        )

        # Verify aligned line matches original
        assert jax.numpy.allclose(aligned_line, original_line, rtol=5e-2, atol=1e-3)

        # Verify transform is returned and has correct keys
        assert "rotation" in align_transforms
        assert "translation" in align_transforms
        assert align_transforms["rotation"].shape == (batch_size, 3, 3)
        assert align_transforms["translation"].shape == (batch_size, 3)

    def test_alignment_transform_is_inverse_torch(self, shape: tuple[int, ...]):
        """Test that alignment transform is the inverse of the applied transform."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points
        t = torch.linspace(0, 1, num_points, dtype=torch.float32)
        original_line = torch.stack(
            [t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1
        )
        original_line = original_line.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply a known transform
        transform = create_random_transform(original_line, center_before_rotation=False)
        rotated_line = apply_forward_transform(original_line, transform)

        weights = torch.ones(batch_size, num_points, dtype=torch.float32)
        mask = torch.ones(batch_size, num_points, dtype=torch.float32)

        # Get alignment transform
        _, align_transforms = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=True,
        )

        # Apply alignment transform to rotated_line should give original_line
        recovered_line = apply_forward_transform(rotated_line, align_transforms)
        torch.testing.assert_close(recovered_line, original_line, rtol=5e-2, atol=1e-3)

    def test_alignment_transform_is_inverse_jax(self, shape: tuple[int, ...]):
        """Test that alignment transform is the inverse of the applied transform."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create a line of points
        t = jax.numpy.linspace(0, 1, num_points, dtype=jax.numpy.float32)
        original_line = jax.numpy.stack(
            [t, jax.numpy.zeros_like(t), jax.numpy.zeros_like(t)], axis=-1
        )
        original_line = jax.numpy.expand_dims(original_line, 0)
        original_line = jax.numpy.broadcast_to(
            original_line, (batch_size, num_points, 3)
        )

        # Apply a known transform
        transform = create_random_transform(
            original_line, center_before_rotation=False, key=1
        )
        rotated_line = apply_forward_transform(original_line, transform)

        weights = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)
        mask = jax.numpy.ones((batch_size, num_points), dtype=jax.numpy.float32)

        # Get alignment transform
        _, align_transforms = weighted_rigid_align_differentiable(
            rotated_line,
            original_line,
            weights=weights,
            mask=mask,
            return_transforms=True,
            allow_gradients=True,
        )

        # Apply alignment transform to rotated_line should give original_line
        recovered_line = apply_forward_transform(rotated_line, align_transforms)
        assert jax.numpy.allclose(recovered_line, original_line, rtol=2e-1, atol=1e-2)

    def test_consistency_with_boltz(self, shape: tuple[int, ...]):
        """Test that PyTorch, JAX, and Boltz implementations are consistent."""
        batch_size = shape[0]
        num_points = shape[1]

        # Create test data in PyTorch
        torch.manual_seed(1)
        true_coords_torch = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
        pred_coords_torch = torch.randn(batch_size, num_points, 3, dtype=torch.float32)
        weights_torch = torch.ones(batch_size, num_points, dtype=torch.float32)
        mask_torch = torch.ones(batch_size, num_points, dtype=torch.float32)

        # Run Boltz implementation (original)
        aligned_boltz = boltz_weighted_rigid_align(
            true_coords_torch, pred_coords_torch, weights_torch, mask_torch
        )

        # Run our PyTorch implementation
        aligned_torch = weighted_rigid_align_differentiable(
            true_coords_torch,
            pred_coords_torch,
            weights_torch,
            mask_torch,
            allow_gradients=False,
        )

        # Convert to JAX and run JAX implementation
        true_coords_jax = jax.numpy.array(true_coords_torch.numpy())
        pred_coords_jax = jax.numpy.array(pred_coords_torch.numpy())
        weights_jax = jax.numpy.array(weights_torch.numpy())
        mask_jax = jax.numpy.array(mask_torch.numpy())

        aligned_jax = weighted_rigid_align_differentiable(
            true_coords_jax,
            pred_coords_jax,
            weights_jax,
            mask_jax,
            allow_gradients=False,
        )

        # Verify all three implementations match within tolerance
        torch.testing.assert_close(
            aligned_torch, aligned_boltz, rtol=5e-2, atol=1e-3, msg="PyTorch vs Boltz"
        )

        # Convert JAX array to numpy then to torch for comparison
        import numpy as np

        aligned_jax_np = np.asarray(aligned_jax)
        torch.testing.assert_close(
            torch.from_numpy(aligned_jax_np).clone(),
            aligned_boltz,
            rtol=5e-2,
            atol=1e-3,
            msg="JAX vs Boltz",
        )
