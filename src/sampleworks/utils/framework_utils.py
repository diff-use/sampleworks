from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TYPE_CHECKING, TypeVar

import numpy as np


TORCH_AVAILABLE = False
JAX_AVAILABLE = False

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    import torch

    Array = torch.Tensor | jax.Array
else:
    try:
        import torch

        TORCH_AVAILABLE = True
    except ImportError:
        torch = None

    try:
        import jax
        import jax.numpy as jnp

        JAX_AVAILABLE = True
    except ImportError:
        jax = None
        jnp = None

    if TORCH_AVAILABLE:
        if JAX_AVAILABLE:
            Array = torch.Tensor | jax.Array
        else:
            Array = torch.Tensor
    elif JAX_AVAILABLE:
        Array = jax.Array
    else:
        raise ImportError("Either PyTorch or JAX must be installed to use this module.")


T = TypeVar("T")
P = ParamSpec("P")


def is_jax_array(x: Any) -> bool:
    return JAX_AVAILABLE and isinstance(x, jax.Array)


def is_torch_tensor(x: Any) -> bool:
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def jax_to_torch(x: jax.Array, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(x))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_to_jax(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.detach().cpu().numpy())


def ensure_torch(*arg_names: str, device: torch.device | None = None):
    """
    Decorator that converts specified JAX array arguments to PyTorch tensors.

    Usage:
        @ensure_torch("x_t", "t")
        def forward(self, x_t, t, conditioning=None):
            ...
    """

    # NOTE: Callable[P, T] -> Callable[P, T] is technically inexact. It
    # accepts jax.Array where fn's signature expects torch.Tensor. ParamSpec can't
    # express this broadened input in current Python typing, so we keep this for
    # IDE autocomplete/return type propagation and accept the input-type imprecision.
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in arg_names:
                if name in bound.arguments:
                    val = bound.arguments[name]
                    if is_jax_array(val):
                        bound.arguments[name] = jax_to_torch(val, device)

            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def ensure_jax(*arg_names: str):
    """
    Decorator that converts specified PyTorch tensor arguments to JAX arrays.
    """

    # NOTE: Same ParamSpec caveat as ensure_torch.
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name in arg_names:
                if name in bound.arguments:
                    val = bound.arguments[name]
                    if is_torch_tensor(val):
                        bound.arguments[name] = torch_to_jax(val)

            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
