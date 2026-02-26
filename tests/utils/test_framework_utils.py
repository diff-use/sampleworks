"""Tests for framework conversion utilities from sampleworks.utils.framework_utils."""

import numpy as np
import pytest
from sampleworks.utils.framework_utils import (
    ensure_jax,
    ensure_torch,
    is_jax_array,
    is_torch_tensor,
    jax_to_torch,
    match_batch,
    torch_to_jax,
)


jax = pytest.importorskip("jax")
import jax.numpy as jnp


torch = pytest.importorskip("torch")


class TestTypeChecking:
    """Test type checking functions."""

    def test_is_jax_array_with_jax_array(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert is_jax_array(x) is True

    def test_is_jax_array_with_torch_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert is_jax_array(x) is False

    def test_is_jax_array_with_numpy(self):
        x = np.array([1.0, 2.0, 3.0])
        assert is_jax_array(x) is False

    def test_is_jax_array_with_list(self):
        x = [1.0, 2.0, 3.0]
        assert is_jax_array(x) is False

    def test_is_jax_array_with_none(self):
        assert is_jax_array(None) is False

    def test_is_torch_tensor_with_torch_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert is_torch_tensor(x) is True

    def test_is_torch_tensor_with_jax_array(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert is_torch_tensor(x) is False

    def test_is_torch_tensor_with_numpy(self):
        x = np.array([1.0, 2.0, 3.0])
        assert is_torch_tensor(x) is False

    def test_is_torch_tensor_with_list(self):
        x = [1.0, 2.0, 3.0]
        assert is_torch_tensor(x) is False

    def test_is_torch_tensor_with_none(self):
        assert is_torch_tensor(None) is False


class TestConversionFunctions:
    """Test direct conversion functions."""

    @pytest.mark.parametrize(
        "shape",
        [(3,), (2, 3), (2, 3, 4), (5, 10, 3)],
    )
    def test_jax_to_torch_preserves_values(self, shape: tuple[int, ...]):
        x_jax = jax.random.normal(jax.random.PRNGKey(0), shape)
        x_torch = jax_to_torch(x_jax)

        assert isinstance(x_torch, torch.Tensor)
        assert x_torch.shape == x_jax.shape
        np.testing.assert_allclose(x_torch.numpy(), np.asarray(x_jax), rtol=1e-6)

    @pytest.mark.parametrize(
        "shape",
        [(3,), (2, 3), (2, 3, 4), (5, 10, 3)],
    )
    def test_torch_to_jax_preserves_values(self, shape: tuple[int, ...]):
        x_torch = torch.randn(*shape)
        x_jax = torch_to_jax(x_torch)

        assert isinstance(x_jax, jax.Array)
        assert x_jax.shape == x_torch.shape
        np.testing.assert_allclose(np.asarray(x_jax), x_torch.numpy(), rtol=1e-6)

    def test_jax_to_torch_with_device(self):
        x_jax = jnp.array([1.0, 2.0, 3.0])
        x_torch = jax_to_torch(x_jax, device=torch.device("cpu"))

        assert isinstance(x_torch, torch.Tensor)
        assert x_torch.device == torch.device("cpu")

    def test_roundtrip_jax_torch_jax(self):
        x_original = jax.random.normal(jax.random.PRNGKey(42), (5, 10, 3))
        x_torch = jax_to_torch(x_original)
        x_recovered = torch_to_jax(x_torch)

        np.testing.assert_allclose(np.asarray(x_recovered), np.asarray(x_original), rtol=1e-6)

    def test_roundtrip_torch_jax_torch(self):
        x_original = torch.randn(5, 10, 3)
        x_jax = torch_to_jax(x_original)
        x_recovered = jax_to_torch(x_jax)

        torch.testing.assert_close(x_recovered, x_original, rtol=1e-6, atol=1e-6)

    def test_jax_to_torch_preserves_dtype_float32(self):
        x_jax = jnp.array([1.0, 2.0], dtype=jnp.float32)
        x_torch = jax_to_torch(x_jax)
        assert x_torch.dtype == torch.float32

    @pytest.mark.skipif(
        not jax.config.x64_enabled,
        reason="JAX x64 not enabled",
    )
    def test_jax_to_torch_preserves_dtype_float64(self):
        x_jax = jnp.array([1.0, 2.0], dtype=jnp.float64)
        x_torch = jax_to_torch(x_jax)
        assert x_torch.dtype == torch.float64

    def test_jax_to_torch_preserves_dtype_int32(self):
        x_jax = jnp.array([1, 2], dtype=jnp.int32)
        x_torch = jax_to_torch(x_jax)
        assert x_torch.dtype == torch.int32

    def test_torch_to_jax_preserves_dtype_float32(self):
        x_torch = torch.tensor([1.0, 2.0], dtype=torch.float32)
        x_jax = torch_to_jax(x_torch)
        assert x_jax.dtype == jnp.float32

    @pytest.mark.skipif(
        not jax.config.x64_enabled,
        reason="JAX x64 not enabled",
    )
    def test_torch_to_jax_preserves_dtype_float64(self):
        x_torch = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x_jax = torch_to_jax(x_torch)
        assert x_jax.dtype == jnp.float64

    def test_torch_to_jax_detaches_gradient(self):
        x_torch = torch.tensor([1.0, 2.0], requires_grad=True)
        x_jax = torch_to_jax(x_torch)
        assert isinstance(x_jax, jax.Array)


class TestEnsureTorchDecorator:
    """Test the @ensure_torch decorator."""

    def test_converts_jax_positional_arg(self):
        @ensure_torch("x")
        def fn(x):
            return x

        x_jax = jnp.array([1.0, 2.0, 3.0])
        result = fn(x_jax)

        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), np.asarray(x_jax), rtol=1e-6)

    def test_converts_jax_keyword_arg(self):
        @ensure_torch("x")
        def fn(x):
            return x

        x_jax = jnp.array([1.0, 2.0, 3.0])
        result = fn(x=x_jax)

        assert isinstance(result, torch.Tensor)
        np.testing.assert_allclose(result.numpy(), np.asarray(x_jax), rtol=1e-6)

    def test_passes_through_torch_tensor(self):
        @ensure_torch("x")
        def fn(x):
            return x

        x_torch = torch.tensor([1.0, 2.0, 3.0])
        result = fn(x_torch)

        assert result is x_torch

    def test_converts_multiple_args(self):
        @ensure_torch("x", "y")
        def fn(x, y):
            return x, y

        x_jax = jnp.array([1.0, 2.0])
        y_jax = jnp.array([3.0, 4.0])
        x_result, y_result = fn(x_jax, y_jax)

        assert isinstance(x_result, torch.Tensor)
        assert isinstance(y_result, torch.Tensor)

    def test_converts_only_specified_args(self):
        @ensure_torch("x")
        def fn(x, y):
            return x, y

        x_jax = jnp.array([1.0, 2.0])
        y_jax = jnp.array([3.0, 4.0])
        x_result, y_result = fn(x_jax, y_jax)

        assert isinstance(x_result, torch.Tensor)
        assert isinstance(y_result, jax.Array)

    def test_ignores_nonexistent_arg_name(self):
        @ensure_torch("nonexistent")
        def fn(x):
            return x

        x_jax = jnp.array([1.0, 2.0])
        result = fn(x_jax)

        assert isinstance(result, jax.Array)

    def test_handles_default_args(self):
        @ensure_torch("x", "y")
        def fn(x, y=None):
            return x, y

        x_jax = jnp.array([1.0, 2.0])
        x_result, y_result = fn(x_jax)

        assert isinstance(x_result, torch.Tensor)
        assert y_result is None

    def test_handles_mixed_positional_and_keyword(self):
        @ensure_torch("x", "z")
        def fn(x, y, z=None):
            return x, y, z

        x_jax = jnp.array([1.0])
        y_jax = jnp.array([2.0])
        z_jax = jnp.array([3.0])
        x_result, y_result, z_result = fn(x_jax, y_jax, z=z_jax)

        assert isinstance(x_result, torch.Tensor)
        assert isinstance(y_result, jax.Array)
        assert isinstance(z_result, torch.Tensor)

    def test_preserves_function_metadata(self):
        @ensure_torch("x")
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_works_with_method(self):
        class MyClass:
            @ensure_torch("x")
            def method(self, x):
                return x

        obj = MyClass()
        x_jax = jnp.array([1.0, 2.0])
        result = obj.method(x_jax)

        assert isinstance(result, torch.Tensor)

    def test_with_device_parameter(self):
        @ensure_torch("x", device=torch.device("cpu"))
        def fn(x):
            return x

        x_jax = jnp.array([1.0, 2.0])
        result = fn(x_jax)

        assert isinstance(result, torch.Tensor)
        assert result.device == torch.device("cpu")


class TestEnsureJaxDecorator:
    """Test the @ensure_jax decorator."""

    def test_converts_torch_positional_arg(self):
        @ensure_jax("x")
        def fn(x):
            return x

        x_torch = torch.tensor([1.0, 2.0, 3.0])
        result = fn(x_torch)

        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), x_torch.numpy(), rtol=1e-6)

    def test_converts_torch_keyword_arg(self):
        @ensure_jax("x")
        def fn(x):
            return x

        x_torch = torch.tensor([1.0, 2.0, 3.0])
        result = fn(x=x_torch)

        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), x_torch.numpy(), rtol=1e-6)

    def test_passes_through_jax_array(self):
        @ensure_jax("x")
        def fn(x):
            return x

        x_jax = jnp.array([1.0, 2.0, 3.0])
        result = fn(x_jax)

        assert result is x_jax

    def test_converts_multiple_args(self):
        @ensure_jax("x", "y")
        def fn(x, y):
            return x, y

        x_torch = torch.tensor([1.0, 2.0])
        y_torch = torch.tensor([3.0, 4.0])
        x_result, y_result = fn(x_torch, y_torch)

        assert isinstance(x_result, jax.Array)
        assert isinstance(y_result, jax.Array)

    def test_converts_only_specified_args(self):
        @ensure_jax("x")
        def fn(x, y):
            return x, y

        x_torch = torch.tensor([1.0, 2.0])
        y_torch = torch.tensor([3.0, 4.0])
        x_result, y_result = fn(x_torch, y_torch)

        assert isinstance(x_result, jax.Array)
        assert isinstance(y_result, torch.Tensor)

    def test_ignores_nonexistent_arg_name(self):
        @ensure_jax("nonexistent")
        def fn(x):
            return x

        x_torch = torch.tensor([1.0, 2.0])
        result = fn(x_torch)

        assert isinstance(result, torch.Tensor)

    def test_handles_default_args(self):
        @ensure_jax("x", "y")
        def fn(x, y=None):
            return x, y

        x_torch = torch.tensor([1.0, 2.0])
        x_result, y_result = fn(x_torch)

        assert isinstance(x_result, jax.Array)
        assert y_result is None

    def test_handles_mixed_positional_and_keyword(self):
        @ensure_jax("x", "z")
        def fn(x, y, z=None):
            return x, y, z

        x_torch = torch.tensor([1.0])
        y_torch = torch.tensor([2.0])
        z_torch = torch.tensor([3.0])
        x_result, y_result, z_result = fn(x_torch, y_torch, z=z_torch)

        assert isinstance(x_result, jax.Array)
        assert isinstance(y_result, torch.Tensor)
        assert isinstance(z_result, jax.Array)

    def test_preserves_function_metadata(self):
        @ensure_jax("x")
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_works_with_method(self):
        class MyClass:
            @ensure_jax("x")
            def method(self, x):
                return x

        obj = MyClass()
        x_torch = torch.tensor([1.0, 2.0])
        result = obj.method(x_torch)

        assert isinstance(result, jax.Array)

    def test_handles_tensor_with_grad(self):
        @ensure_jax("x")
        def fn(x):
            return x

        x_torch = torch.tensor([1.0, 2.0], requires_grad=True)
        result = fn(x_torch)

        assert isinstance(result, jax.Array)


class TestDecoratorIntegration:
    """Integration tests for decorator usage in realistic scenarios."""

    def test_ensure_torch_in_torch_function(self):
        @ensure_torch("coords", "weights")
        def compute_weighted_mean(coords, weights):
            return (coords * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()

        coords_jax = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
        weights_jax = jnp.ones(10)

        result = compute_weighted_mean(coords_jax, weights_jax)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)

    def test_ensure_jax_in_jax_function(self):
        @ensure_jax("coords", "weights")
        def compute_weighted_mean(coords, weights):
            return jnp.sum(coords * weights[:, None], axis=0) / jnp.sum(weights)

        coords_torch = torch.randn(10, 3)
        weights_torch = torch.ones(10)

        result = compute_weighted_mean(coords_torch, weights_torch)

        assert isinstance(result, jax.Array)
        assert result.shape == (3,)

    def test_decorator_with_non_array_args(self):
        @ensure_torch("x")
        def fn(x, scale: float, name: str):
            return x * scale, name

        x_jax = jnp.array([1.0, 2.0])
        result, name = fn(x_jax, scale=2.0, name="test")

        assert isinstance(result, torch.Tensor)
        assert name == "test"
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0]))

    def test_nested_decorators(self):
        @ensure_torch("output")
        @ensure_jax("input")
        def process(input, output):
            return input, output

        x_torch = torch.tensor([1.0])
        y_jax = jnp.array([2.0])

        input_result, output_result = process(x_torch, y_jax)

        assert isinstance(input_result, jax.Array)
        assert isinstance(output_result, torch.Tensor)


class TestMatchBatch:
    """Test batch-size matching across frameworks."""

    def test_torch_passthrough_when_batch_matches(self):
        array = torch.randn(4, 10, 3)
        result = match_batch(array, target_batch_size=4)
        assert result is array

    def test_torch_singleton_broadcast(self):
        array = torch.randn(1, 6, 3)
        result = match_batch(array, target_batch_size=3)
        assert result.shape == (3, 6, 3)
        torch.testing.assert_close(result[0], array[0])
        torch.testing.assert_close(result[1], array[0])
        torch.testing.assert_close(result[2], array[0])

    def test_jax_singleton_broadcast(self):
        array = jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3)
        result = match_batch(array, target_batch_size=3)
        assert result.shape == (3, 2, 3)
        np.testing.assert_allclose(np.asarray(result[0]), np.asarray(array[0]))
        np.testing.assert_allclose(np.asarray(result[1]), np.asarray(array[0]))

    def test_raises_on_incompatible_batch_sizes(self):
        array = torch.randn(2, 5, 3)
        with pytest.raises(ValueError, match="not divisible"):
            match_batch(array, target_batch_size=5)

    def test_raises_on_scalar_input(self):
        scalar = torch.tensor(1.0)
        with pytest.raises(ValueError, match="ndim >= 1"):
            match_batch(scalar, target_batch_size=2)
