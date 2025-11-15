from types import ModuleType
from typing import TYPE_CHECKING

import einx


if TYPE_CHECKING:
    from jaxtyping import Array, Float
    from torch import Tensor
else:
    try:
        from jaxtyping import Array, Float
    except ImportError:
        Array = object
        Float = object
    try:
        from torch import Tensor
    except ImportError:
        Tensor = object


def get_backend(
    coords: Float[Array | Tensor, "..."],
) -> tuple[str, ModuleType]:
    """
    Infer backend from coordinates and return backend name and modules.

    Parameters
    ----------
    coords : Float[Array | Tensor, "..."]
        Coordinates to infer backend from

    Returns
    -------
    tuple[str, Any, Any]
        Backend name ("torch" or "jax"), primary module (torch/jnp),
        and secondary module (torch/jax)

    Raises
    ------
    RuntimeError
        If backend cannot be determined
    """
    try:
        import torch

        if isinstance(coords, torch.Tensor):
            return "torch", torch
    except ImportError:
        pass

    try:
        import jax

        if isinstance(coords, jax.Array):
            return "jax", jax
    except ImportError:
        pass

    msg = "Unsupported backend: coords must be torch.Tensor or jax.Array"
    raise RuntimeError(msg)


def apply_inverse_transform(
    coords: Float[Array | Tensor, "... n 3"],
    transform: dict[str, Float[Array | Tensor, "..."]],
) -> Float[Array | Tensor, "... n 3"]:
    """
    Apply inverse of augmentation transform to coordinates.

    The forward transform is: coords_aug = coords @ R.T + t
    The inverse transform is: coords = (coords_aug - t) @ R

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates in augmented frame
    transform : dict[str, Array | Tensor]
        Transform dictionary with keys "rotation" and "translation"

    Returns
    -------
    Float[Array | Tensor, "... n 3"]
        Coordinates in input frame
    """
    R = transform["rotation"]
    t = transform["translation"]
    coords_centered = einx.subtract("... n d, ... d -> ... n d", coords, t)
    return einx.dot("... n i, ... i j -> ... n j", coords_centered, R)


def apply_forward_transform(
    coords: Float[Array | Tensor, "... n 3"],
    transform: dict[str, Float[Array | Tensor, "..."]],
) -> Float[Array | Tensor, "... n 3"]:
    """
    Apply augmentation transform to coordinates.

    The forward transform is: coords_aug = coords @ R.T + t

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates in input frame
    transform : dict[str, Array | Tensor]
        Transform dictionary with keys "rotation" and "translation"

    Returns
    -------
    Float[Array | Tensor, "... n 3"]
        Coordinates in augmented frame
    """
    R = transform["rotation"]
    t = transform["translation"]
    coords_rotated = einx.dot("... n i, ... j i -> ... n j", coords, R)
    return einx.add("... n d, ... d -> ... n d", coords_rotated, t)


def random_rotation_matrix(
    coords: Float[Array | Tensor, "... n 3"],
    key: int | None = None,
) -> Float[Array | Tensor, "... 3 3"]:
    """
    Generate random 3D rotation matrices using QR decomposition.

    Supports batched inputs - generates independent rotation for each batch.

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates to infer backend and batch shape from
    key : int | None, optional
        Random key for JAX backend, by default None

    Returns
    -------
    Float[Array | Tensor, "... 3 3"]
        Random rotation matrices (orthogonal with det=1)
    """
    backend_name, backend = get_backend(coords)
    batch_shape = coords.shape[:-2]

    if backend_name == "torch":
        shape = (*batch_shape, 3, 3)
        random_matrix = backend.randn(*shape, device=coords.device, dtype=coords.dtype)
        q, r = backend.linalg.qr(random_matrix)
        d = backend.diagonal(r, dim1=-2, dim2=-1)
        d = backend.diag_embed(backend.sign(d))
        rotation = einx.dot("... i j, ... j k -> ... i k", q, d)
        det = backend.linalg.det(rotation)
        det_expanded = det[..., None, None]
        rotation = backend.where(det_expanded > 0, rotation, -rotation)
        return rotation

    elif backend_name == "jax":
        shape = (*batch_shape, 3, 3)
        key_obj = backend.random.PRNGKey(key or 0)
        random_matrix = backend.random.normal(key_obj, shape)
        q, r = backend.numpy.linalg.qr(random_matrix)
        d = backend.numpy.diagonal(r, axis1=-2, axis2=-1)
        d = backend.numpy.expand_dims(d, axis=-1) * backend.numpy.eye(3)
        d = backend.numpy.sign(d)
        rotation = einx.dot("... i j, ... j k -> ... i k", q, d)
        det = backend.numpy.linalg.det(rotation)
        det_expanded = backend.numpy.expand_dims(det, axis=(-2, -1))
        rotation = backend.numpy.where(det_expanded > 0, rotation, -rotation)
        return rotation

    msg = f"Unsupported backend: {backend_name}"
    raise RuntimeError(msg)


def create_random_transform(
    coords: Float[Array | Tensor, "... n 3"],
    center_before_rotation: bool = True,
    scale_translation: float = 1.0,
    key: int | None = None,
) -> dict[str, Float[Array | Tensor, "... 3"]]:
    """
    Create a random rotation and translation transform.

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates to base the transform on (for backend dispatch)
    center_before_rotation : bool, optional
        If True, center the structure at origin before rotating, by default True
    scale_translation : float, optional
        Scale factor for random translation magnitude, by default 1.0
    key : int | None, optional
        Random key for JAX backend, by default None

    Returns
    -------
    dict[str, Array | Tensor]
        Transform dictionary with keys "rotation" and "translation"
    """
    R = random_rotation_matrix(coords, key=key)

    if center_before_rotation:
        centroid = einx.mean("... n d -> ... d", coords)
        t = -centroid
    else:
        backend_name, backend = get_backend(coords)
        batch_shape = coords.shape[:-2]
        shape = (*batch_shape, 3)

        if backend_name == "torch":
            t = backend.randn(*shape, device=coords.device, dtype=coords.dtype)
            t = t * scale_translation
        elif backend_name == "jax":
            key_obj = backend.random.PRNGKey(key or 1)
            t = backend.random.normal(key_obj, shape) * scale_translation
        else:
            msg = f"Unsupported backend: {backend_name}"
            raise RuntimeError(msg)

    return {"rotation": R, "translation": t}
