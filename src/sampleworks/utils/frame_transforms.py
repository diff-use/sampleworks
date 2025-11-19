from collections.abc import Mapping
from types import ModuleType
from typing import Literal, overload, TYPE_CHECKING

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
    tuple[str, ModuleType]
        Backend name ("torch" or "jax") and backend module (torch or jax)

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


@overload
def apply_inverse_transform(
    coords: Float[Tensor, "... n 3"],
    transform: Mapping[str, Float[Tensor, "..."]],
    rotation_only: bool = ...,
) -> Float[Tensor, "... n 3"]: ...
@overload
def apply_inverse_transform(
    coords: Float[Array, "... n 3"],
    transform: Mapping[str, Float[Array, "..."]],
    rotation_only: bool = ...,
) -> Float[Array, "... n 3"]: ...


def apply_inverse_transform(
    coords: Float[Array | Tensor, "... n 3"],
    transform: Mapping[str, Float[Array | Tensor, "..."]],
    rotation_only: bool = False,
) -> Float[Array | Tensor, "... n 3"]:
    """
    Apply inverse of augmentation transform to coordinates.

    The forward transform is: coords_aug = R @ coords + t
    The inverse transform is: coords = R.T @ (coords_aug - t)

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates in augmented frame
    transform : dict[str, Array | Tensor]
        Transform dictionary with keys "rotation" and "translation"
    rotation_only : bool, optional
        If True, only apply rotation without translation, by default False

    Returns
    -------
    Float[Array | Tensor, "... n 3"]
        Coordinates in input frame
    """
    R = transform["rotation"]
    t = transform["translation"]
    coords_centered = coords
    if not rotation_only:
        coords_centered = einx.subtract("... n d, ... d -> ... n d", coords, t)
    return einx.dot("... j i, ... n j -> ... n i", R, coords_centered)


@overload
def apply_forward_transform(
    coords: Float[Tensor, "... n 3"],
    transform: Mapping[str, Float[Tensor, "..."]],
    rotation_only: bool = ...,
) -> Float[Tensor, "... n 3"]: ...
@overload
def apply_forward_transform(
    coords: Float[Array, "... n 3"],
    transform: Mapping[str, Float[Array, "..."]],
    rotation_only: bool = ...,
) -> Float[Array, "... n 3"]: ...


def apply_forward_transform(
    coords: Float[Array | Tensor, "... n 3"],
    transform: Mapping[str, Float[Array | Tensor, "..."]],
    rotation_only: bool = False,
) -> Float[Array | Tensor, "... n 3"]:
    """
    Apply augmentation transform to coordinates.

    The forward transform is: coords_aug = R @ coords + t (standard left multiplication)

    Parameters
    ----------
    coords : Float[Array | Tensor, "... n 3"]
        Coordinates in input frame
    transform : dict[str, Array | Tensor]
        Transform dictionary with keys "rotation" and "translation"
    rotation_only : bool, optional
        If True, only apply rotation without translation, by default False

    Returns
    -------
    Float[Array | Tensor, "... n 3"]
        Coordinates in augmented frame
    """
    R = transform["rotation"]
    t = transform["translation"]
    coords_rotated = einx.dot("... i j, ... n j -> ... n i", R, coords)
    return (
        einx.add("... n d, ... d -> ... n d", coords_rotated, t)
        if not rotation_only
        else coords_rotated
    )


@overload
def random_rotation_matrix(
    coords: Float[Tensor, "... n 3"], key: int | None = ...
) -> Float[Tensor, "... 3 3"]: ...
@overload
def random_rotation_matrix(
    coords: Float[Array, "... n 3"], key: int | None = ...
) -> Float[Array, "... 3 3"]: ...


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


@overload
def create_random_transform(
    coords: Float[Tensor, "... n 3"],
    center_before_rotation: bool = ...,
    scale_translation: float = ...,
    key: int | None = ...,
) -> Mapping[str, Float[Tensor, "..."]]: ...
@overload
def create_random_transform(
    coords: Float[Array, "... n 3"],
    center_before_rotation: bool = ...,
    scale_translation: float = ...,
    key: int | None = ...,
) -> Mapping[str, Float[Array, "..."]]: ...


def create_random_transform(
    coords: Float[Array | Tensor, "... n 3"],
    center_before_rotation: bool = True,
    scale_translation: float = 1.0,
    key: int | None = None,
) -> Mapping[str, Float[Array | Tensor, "... 3"]]:
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


@overload
def weighted_rigid_align_differentiable(
    true_coords: Tensor,
    pred_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
    return_transforms: Literal[False] = False,
    allow_gradients: bool = True,
) -> Tensor: ...


@overload
def weighted_rigid_align_differentiable(
    true_coords: Tensor,
    pred_coords: Tensor,
    weights: Tensor,
    mask: Tensor,
    return_transforms: Literal[True],
    allow_gradients: bool = True,
) -> tuple[Tensor, Mapping[str, Tensor]]: ...


@overload
def weighted_rigid_align_differentiable(
    true_coords: Array,
    pred_coords: Array,
    weights: Array,
    mask: Array,
    return_transforms: Literal[False] = False,
    allow_gradients: bool = True,
) -> Array: ...


@overload
def weighted_rigid_align_differentiable(
    true_coords: Array,
    pred_coords: Array,
    weights: Array,
    mask: Array,
    return_transforms: Literal[True],
    allow_gradients: bool = True,
) -> tuple[Array, Mapping[str, Array]]: ...


def weighted_rigid_align_differentiable(
    true_coords: Array | Tensor,
    pred_coords: Array | Tensor,
    weights: Array | Tensor,
    mask: Array | Tensor,
    return_transforms: bool = False,
    allow_gradients: bool = True,
) -> (Array | Tensor) | tuple[Array | Tensor, Mapping[str, Array | Tensor]]:
    """Compute weighted alignment with optional gradient preservation.

    Identical to boltz.model.loss.diffusion.weighted_rigid_align but without
    the detach_() call when allow_gradients=True, enabling gradient flow.

    I preserve the same parameter names as the original function, but note that
    true_coords will be aligned to the pred_coords in both implementations.

    Parameters
    ----------
    true_coords: Array | Tensor
        The ground truth atom coordinates
    pred_coords: Array | Tensor
        The predicted atom coordinates
    weights: Array | Tensor
        The weights for alignment
    mask: Array | Tensor
        The atoms mask
    return_transforms: bool, optional
        If True, also return the computed rotation and translation. Default: False
    allow_gradients: bool, optional
        If True, preserve gradients through alignment. If False, detach
        (matches original Boltz behavior). Default: True

    Returns
    -------
    Array | Tensor | tuple[Array | Tensor, dict[str, Array | Tensor]]
        Aligned coordinates of true -> pred, and optionally the transforms applied
        to align them.

    """
    backend_name, backend = get_backend(true_coords)
    batch_size, num_points, dim = true_coords.shape

    if backend_name == "torch":
        weights_expanded = (mask * weights).unsqueeze(-1)  # type: ignore[union-attr]

        true_centroid = (true_coords * weights_expanded).sum(  # type: ignore[call-arg]
            dim=1,  # type: ignore[call-arg]
            keepdim=True,  # type: ignore[call-arg]
        ) / weights_expanded.sum(dim=1, keepdim=True)  # type: ignore[call-arg]
        pred_centroid = (pred_coords * weights_expanded).sum(  # type: ignore[call-arg]
            dim=1,  # type: ignore[call-arg]
            keepdim=True,  # type: ignore[call-arg]
        ) / weights_expanded.sum(dim=1, keepdim=True)  # type: ignore[call-arg]

        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            print(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        cov_matrix = einx.dot(
            "b [n] i, b [n] j -> b i j",
            weights_expanded * pred_coords_centered,
            true_coords_centered,
        )

        original_dtype = cov_matrix.dtype
        cov_matrix_32 = cov_matrix.to(dtype=backend.float32)  # type: ignore[union-attr]

        U, _, Vh = backend.linalg.svd(cov_matrix_32)

        rotation = backend.matmul(U, Vh)

        det = backend.det(rotation)
        diag = backend.ones(
            batch_size, dim, device=rotation.device, dtype=backend.float32
        )
        diag[:, -1] = det

        rotation = backend.matmul(U * diag.unsqueeze(1), Vh)

        rotation = rotation.to(dtype=original_dtype)  # type: ignore[union-attr]

        # true @ rot.T
        aligned_coords = (
            einx.dot("b n i, b j i -> b n j", true_coords_centered, rotation)
            + pred_centroid
        )

        if not allow_gradients:
            aligned_coords = aligned_coords.detach()

        if return_transforms:
            # Alignment uses: aligned = true @ rotation.T + pred_centroid
            # Transform functions use left multiplication: coords' = R @ coords + t
            # Since we use rotation.T in alignment, return rotation for left-mult
            translation_uncentered = pred_centroid - einx.dot(  # type: ignore[operator]
                "b i j, b n j -> b n i", rotation, true_centroid
            )
            transforms = {
                "rotation": rotation,
                "translation": einx.rearrange("b () d -> b d", translation_uncentered),
            }
            return aligned_coords, transforms

        return aligned_coords

    elif backend_name == "jax":
        weights_expanded = backend.numpy.expand_dims(mask * weights, axis=-1)

        true_centroid = (true_coords * weights_expanded).sum(
            axis=1, keepdims=True
        ) / weights_expanded.sum(axis=1, keepdims=True)
        pred_centroid = (pred_coords * weights_expanded).sum(
            axis=1, keepdims=True
        ) / weights_expanded.sum(axis=1, keepdims=True)

        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            print(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        cov_matrix = einx.dot(
            "b [n] i, b [n] j -> b i j",
            weights_expanded * pred_coords_centered,
            true_coords_centered,
        )

        original_dtype = cov_matrix.dtype
        cov_matrix_32 = cov_matrix.astype(backend.numpy.float32)

        U, _, Vh = backend.numpy.linalg.svd(cov_matrix_32)

        rotation = backend.numpy.matmul(U, Vh)

        det = backend.numpy.linalg.det(rotation)
        diag = backend.numpy.ones((batch_size, dim), dtype=backend.numpy.float32)
        diag = diag.at[:, -1].set(det)

        rotation = backend.numpy.matmul(U * backend.numpy.expand_dims(diag, axis=1), Vh)

        rotation = rotation.astype(original_dtype)

        # Note: Same einsum pattern as Boltz - true @ rot.T
        aligned_coords = (
            einx.dot("b n i, b j i -> b n j", true_coords_centered, rotation)
            + pred_centroid
        )

        if not allow_gradients:
            aligned_coords = backend.lax.stop_gradient(aligned_coords)

        if return_transforms:
            # Same as PyTorch - return rotation for left-multiplication
            translation_uncentered = pred_centroid - einx.dot(
                "b i j, b n j -> b n i", rotation, true_centroid
            )
            transforms = {
                "rotation": rotation,
                "translation": einx.rearrange("b () d -> b d", translation_uncentered),
            }
            return aligned_coords, transforms  # type: ignore[return-value]

        return aligned_coords

    msg = f"Unsupported backend: {backend_name}"
    raise RuntimeError(msg)
