"""Latent-space injection adapter for experimentally-guided sampling.

This module inserts a small, swappable transform on a model's *single
representation* (the post-trunk latent ``s``) at the conditioning boundary
between :meth:`featurize` and :meth:`step`. It is the Step-1 scaffold for an
AlphaSAXS-style latent injection (see project notes): the experimental signal
will eventually drive this transform, but here the transform is a deliberately
trivial affine ``k * s + b`` ("pseudo MLP") so the plumbing and tests can be
validated before any real module is introduced.

Design goals
------------
- **Minimal change.** Everything lives behind :class:`LatentAdaptedWrapper`,
  which itself satisfies the ``FlowModelWrapper`` protocol. The samplers,
  scalers, eval, and their tests consume the protocol, so nothing downstream
  changes. With an identity transform (``k=1, b=0``) the wrapper is
  behaviourally identical to the model it wraps.
- **Heterogeneous across models.** The only model-specific knowledge is *which
  attribute of the conditioning object holds the single representation*. That is
  a single string (``"s"`` for Boltz, ``"s_trunk"`` for Protenix/RF3), supplied
  via :class:`AttrLatentIO`. No model package is imported here, so a 4th or 5th
  model is one more string, not new code.
- **Gradient isolation.** During sampling the injected latent is detached so the
  existing coordinate-only guidance gradient (and all downstream tests) behave
  exactly as before. Training the transform happens in a separate pass (Step 2).
"""

from __future__ import annotations

import dataclasses
from typing import Generic, Protocol, runtime_checkable, TypeVar

import torch
from torch import nn, Tensor

from sampleworks.models.protocol import GenerativeModelInput

C = TypeVar("C")

# Convenience map of the single-representation attribute name per model.
# This is documentation/config only -- the adapter never imports these models.
DEFAULT_SINGLE_REP_ATTR: dict[str, str] = {
    "boltz1": "s",
    "boltz2": "s",
    "protenix": "s_trunk",
    "rf3": "s_trunk",
}


@runtime_checkable
class LatentIO(Protocol[C]):
    """Reads/writes the single representation on a model's conditioning object.

    Implementations encapsulate the *only* model-specific knowledge in this
    module: where the single representation lives on conditioning ``C``.
    """

    def read_single(self, conditioning: C) -> Tensor | None:
        """Return the single representation tensor, or ``None`` if unavailable."""
        ...

    def write_single(self, conditioning: C, single: Tensor) -> C:
        """Return a copy of ``conditioning`` with the single representation replaced."""
        ...


class AttrLatentIO:
    """Generic :class:`LatentIO` that addresses the single rep by attribute name.

    Works for any (dataclass) conditioning whose single representation is stored
    in a single attribute. ``write_single`` uses :func:`dataclasses.replace`, so
    non-tensor sidecar state on the conditioning (e.g. RF3's chiral tracking
    arrays) is preserved untouched.

    Parameters
    ----------
    single_attr
        Name of the attribute holding the single representation, e.g. ``"s"``
        for Boltz or ``"s_trunk"`` for Protenix/RF3.
    """

    def __init__(self, single_attr: str):
        self.single_attr = single_attr

    def read_single(self, conditioning: C) -> Tensor | None:
        if conditioning is None:
            return None
        return getattr(conditioning, self.single_attr, None)

    def write_single(self, conditioning: C, single: Tensor) -> C:
        if dataclasses.is_dataclass(conditioning) and not isinstance(conditioning, type):
            return dataclasses.replace(conditioning, **{self.single_attr: single})  # ty: ignore
        # Fallback for non-dataclass conditioning: mutate a shallow copy.
        import copy

        new_cond = copy.copy(conditioning)
        setattr(new_cond, self.single_attr, single)
        return new_cond


@runtime_checkable
class LatentInjector(Protocol):
    """A transform applied to the single representation. Swappable in Step 2."""

    def __call__(self, single: Tensor) -> Tensor:
        """Map a single representation to a modified single representation."""
        ...


class AffineInjector(nn.Module):
    """Step-1 "pseudo MLP": an elementwise affine transform ``k * s + b``.

    With the defaults (``k=1, b=0``) this is the identity, which makes
    :class:`LatentAdaptedWrapper` behaviourally identical to the wrapped model --
    the basis of the zero-impact guarantee. ``k`` and ``b`` are learnable
    :class:`~torch.nn.Parameter` s so that Step 2 can either optimize them
    directly or replace this module wholesale with an MLP/FiLM head.

    Parameters
    ----------
    k_init, b_init
        Initial scale and shift. Defaults give the identity transform.
    """

    def __init__(self, k_init: float = 1.0, b_init: float = 0.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, single: Tensor) -> Tensor:
        return self.k * single + self.b


class LatentAdaptedWrapper(Generic[C]):  # noqa: UP046 (Python 3.11 compatibility)
    """Decorator over any ``FlowModelWrapper`` that injects into the latent.

    Satisfies the ``FlowModelWrapper`` protocol itself, so it is a drop-in
    replacement for the model it wraps. Only :meth:`featurize` is augmented;
    ``step`` and ``initialize_from_prior`` delegate verbatim.

    Parameters
    ----------
    inner
        The wrapped model wrapper (Boltz/Protenix/RF3 or a mock).
    latent_io
        Accessor for the single representation on ``inner``'s conditioning.
    injector
        Transform applied to the single representation. Defaults to an identity
        :class:`AffineInjector`.
    training_adapter
        When ``False`` (default, i.e. sampling), the injected latent is detached
        so downstream guidance gradients and behaviour are unchanged. Set
        ``True`` only inside a dedicated optimization pass.
    """

    def __init__(
        self,
        inner,
        latent_io: LatentIO[C],
        injector: LatentInjector | None = None,
        *,
        training_adapter: bool = False,
    ):
        self.inner = inner
        self.latent_io = latent_io
        self.injector = injector if injector is not None else AffineInjector()
        self.training_adapter = training_adapter

    def featurize(self, structure: dict) -> GenerativeModelInput[C]:
        """Featurize via ``inner``, then inject into the single representation."""
        feats = self.inner.featurize(structure)
        single = self.latent_io.read_single(feats.conditioning)
        if single is None:
            return feats  # nothing to inject into; pass through unchanged

        injected = self.injector(single)
        if not self.training_adapter:
            # Honour the framework's constant-conditioning contract during
            # sampling: detach so guidance backprop reaches coords only.
            injected = injected.detach()

        new_cond = self.latent_io.write_single(feats.conditioning, injected)
        return GenerativeModelInput(x_init=feats.x_init, conditioning=new_cond)

    def step(self, *args, **kwargs):
        """Delegate denoising to the wrapped model."""
        return self.inner.step(*args, **kwargs)

    def initialize_from_prior(self, *args, **kwargs):
        """Delegate prior initialization to the wrapped model."""
        return self.inner.initialize_from_prior(*args, **kwargs)
