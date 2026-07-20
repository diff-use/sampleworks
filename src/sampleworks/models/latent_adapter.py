"""Latent-space injection adapter for experimentally-guided sampling.

This module inserts small, swappable transforms on a model's post-trunk latents
-- the *single* representation ``s`` and the *pair* representation ``z`` -- at the
conditioning boundary between :meth:`featurize` and :meth:`step`. It is the
plumbing for inference-time latent-space optimization (IT-opt): the free
variables of the optimization are these latents (or a structured perturbation of
them), and the experimental reward is evaluated on the denoised structure. See
``docs/IT_OPTIMIZATION_PLAN.md``.

Design goals
------------
- **Minimal, model-agnostic.** The only model-specific knowledge is *which
  attribute of the conditioning object holds each representation*: a single
  string (``"s"``/``"z"`` for Boltz, ``"s_trunk"``/``"z_trunk"`` for
  Protenix/RF3), supplied via :class:`AttrLatentIO`. No model package is imported
  here, so a 4th or 5th model is one more pair of strings, not new code.
- **Zero-impact by default.** With identity injectors the wrapper is
  behaviourally identical to the model it wraps; a :class:`DeltaInjector` is
  zero-initialized so its first reconstruction is exactly the trunk baseline.
- **Gradient isolation.** During ordinary sampling the injected latents are
  detached so existing coordinate-only guidance behaves exactly as before.
  Training the latents/injectors happens only when ``training_adapter=True``.

.. note::
   The per-diffusion-step IT-opt scaler (``core/scalers/latent_optimization.py``)
   uses :class:`AttrLatentIO` *directly* -- it reads the baseline latents once,
   holds its own optimizable leaves, and re-injects them each step via
   :meth:`AttrLatentIO.write_single`/:meth:`~AttrLatentIO.write_pair`. The
   featurize-time :class:`LatentAdaptedWrapper` path is for the alternative
   "optimize an injector at the featurize boundary" mode.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Generic, Protocol, runtime_checkable, TypeVar

import torch
from torch import nn, Tensor

from sampleworks.models.protocol import GenerativeModelInput

C = TypeVar("C")

# Convenience maps of the representation attribute name per model. Documentation/
# config only -- the adapter never imports these models.
DEFAULT_SINGLE_REP_ATTR: dict[str, str] = {
    "boltz1": "s",
    "boltz2": "s",
    "protenix": "s_trunk",
    "rf3": "s_trunk",
}
DEFAULT_PAIR_REP_ATTR: dict[str, str] = {
    "boltz1": "z",
    "boltz2": "z",
    "protenix": "z_trunk",
    "rf3": "z_trunk",
}


@runtime_checkable
class LatentIO(Protocol[C]):
    """Reads/writes the single and pair representations on a conditioning object.

    Implementations encapsulate the *only* model-specific knowledge in this
    module: where each representation lives on conditioning ``C``. Pair-rep
    methods return ``None`` / pass through unchanged when the implementation is
    single-rep only.
    """

    def read_single(self, conditioning: C) -> Tensor | None:
        """Return the single representation tensor, or ``None`` if unavailable."""
        ...

    def write_single(self, conditioning: C, single: Tensor) -> C:
        """Return a copy of ``conditioning`` with the single representation replaced."""
        ...

    def read_pair(self, conditioning: C) -> Tensor | None:
        """Return the pair representation tensor, or ``None`` if unavailable."""
        ...

    def write_pair(self, conditioning: C, pair: Tensor) -> C:
        """Return a copy of ``conditioning`` with the pair representation replaced."""
        ...


class AttrLatentIO:
    """Generic :class:`LatentIO` addressing each representation by attribute name.

    Works for any (dataclass) conditioning whose representations are stored in
    named attributes. Writes use :func:`dataclasses.replace`, so non-tensor
    sidecar state on the conditioning (e.g. RF3's chiral tracking arrays) is
    preserved untouched. All three model wrappers use
    ``@dataclass(frozen=True, slots=True)`` conditioning, which is exactly this
    case.

    Parameters
    ----------
    single_attr
        Attribute holding the single representation, e.g. ``"s"`` (Boltz) or
        ``"s_trunk"`` (Protenix/RF3).
    pair_attr
        Attribute holding the pair representation, e.g. ``"z"`` / ``"z_trunk"``.
        When ``None`` (default), the pair accessors are no-ops -- single-rep-only
        behaviour, matching the original scaffold.
    """

    def __init__(self, single_attr: str, pair_attr: str | None = None):
        self.single_attr = single_attr
        self.pair_attr = pair_attr

    def read_single(self, conditioning: C) -> Tensor | None:
        return self._read(conditioning, self.single_attr)

    def write_single(self, conditioning: C, single: Tensor) -> C:
        return self._write(conditioning, self.single_attr, single)

    def read_pair(self, conditioning: C) -> Tensor | None:
        if self.pair_attr is None:
            return None
        return self._read(conditioning, self.pair_attr)

    def write_pair(self, conditioning: C, pair: Tensor) -> C:
        if self.pair_attr is None:
            return conditioning
        return self._write(conditioning, self.pair_attr, pair)

    @staticmethod
    def _read(conditioning: C, attr: str) -> Tensor | None:
        if conditioning is None:
            return None
        return getattr(conditioning, attr, None)

    @staticmethod
    def _write(conditioning: C, attr: str, value: Tensor) -> C:
        if dataclasses.is_dataclass(conditioning) and not isinstance(conditioning, type):
            return dataclasses.replace(conditioning, **{attr: value})  # ty: ignore
        # Fallback for non-dataclass conditioning: mutate a shallow copy.
        new_cond = copy.copy(conditioning)
        setattr(new_cond, attr, value)
        return new_cond


@runtime_checkable
class LatentInjector(Protocol):
    """A transform applied to a representation (single or pair). Swappable."""

    def __call__(self, latent: Tensor) -> Tensor:
        """Map a representation tensor to a modified representation tensor."""
        ...


class AffineInjector(nn.Module):
    """Elementwise affine transform ``k * latent + b`` (the original pseudo-MLP).

    With the defaults (``k=1, b=0``) this is the identity, the basis of the
    zero-impact guarantee. ``k`` and ``b`` are scalar learnable parameters. Note
    that two scalars cannot represent a general latent perturbation -- for
    optimizing the latent itself, use :class:`DeltaInjector` (or the scaler's
    direct-leaf path).

    Parameters
    ----------
    k_init, b_init
        Initial scale and shift. Defaults give the identity transform.
    """

    def __init__(self, k_init: float = 1.0, b_init: float = 0.0):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, latent: Tensor) -> Tensor:
        return self.k * latent + self.b


class DeltaInjector(nn.Module):
    """Zero-initialized additive perturbation ``latent + Δ``.

    ``Δ`` is a full free-variable delta the same shape as the latent, lazily
    allocated on the first forward (its shape is only known after featurize) and
    initialized to zero -- so the first reconstruction is exactly the trunk
    baseline. This is the general-purpose injector for latent optimization: the
    anchor prior ``λ·‖latent − latent₀‖²`` becomes simply ``λ·‖Δ‖²``.

    A structured low-rank ``Δ = UᵀV`` variant (lower-variance, drift-resistant)
    is a future swap for the pair representation, whose full delta is O(N²·d).
    """

    def __init__(self):
        super().__init__()
        # Reserve the parameter slot; filled on first forward once the shape is known.
        self.register_parameter("delta", None)

    def forward(self, latent: Tensor) -> Tensor:
        if self.delta is None:
            self.delta = nn.Parameter(torch.zeros_like(latent))
        return latent + self.delta


class LatentAdaptedWrapper(Generic[C]):  # noqa: UP046 (Python 3.11 compatibility)
    """Decorator over any ``FlowModelWrapper`` that injects into the latents.

    Satisfies the ``FlowModelWrapper`` protocol itself, so it is a drop-in
    replacement for the model it wraps. Only :meth:`featurize` is augmented;
    ``step`` and ``initialize_from_prior`` delegate verbatim.

    Parameters
    ----------
    inner
        The wrapped model wrapper (Boltz/Protenix/RF3 or a mock).
    latent_io
        Accessor for the representations on ``inner``'s conditioning.
    single_injector
        Transform applied to the single representation. Defaults to an identity
        :class:`AffineInjector`.
    pair_injector
        Transform applied to the pair representation. ``None`` (default) leaves
        the pair representation untouched.
    training_adapter
        When ``False`` (default, i.e. sampling), injected latents are detached so
        downstream guidance gradients and behaviour are unchanged. Set ``True``
        only inside a dedicated optimization pass.
    """

    def __init__(
        self,
        inner,
        latent_io: LatentIO[C],
        single_injector: LatentInjector | None = None,
        pair_injector: LatentInjector | None = None,
        *,
        training_adapter: bool = False,
    ):
        self.inner = inner
        self.latent_io = latent_io
        self.single_injector = single_injector if single_injector is not None else AffineInjector()
        self.pair_injector = pair_injector
        self.training_adapter = training_adapter

    def featurize(self, structure: dict, **kwargs) -> GenerativeModelInput[C]:
        """Featurize via ``inner``, then inject into the single (and pair) reps."""
        feats = self.inner.featurize(structure, **kwargs)
        conditioning = feats.conditioning

        conditioning = self._inject(
            conditioning, self.latent_io.read_single, self.latent_io.write_single, self.single_injector
        )
        conditioning = self._inject(
            conditioning, self.latent_io.read_pair, self.latent_io.write_pair, self.pair_injector
        )

        if conditioning is feats.conditioning:
            return feats  # nothing injected; pass through unchanged
        return GenerativeModelInput(x_init=feats.x_init, conditioning=conditioning)

    def _inject(self, conditioning: C, read, write, injector: LatentInjector | None) -> C:
        """Read a representation, apply ``injector``, write it back (or pass through)."""
        if injector is None:
            return conditioning
        latent = read(conditioning)
        if latent is None:
            return conditioning
        injected = injector(latent)
        if not self.training_adapter:
            # Honour the constant-conditioning contract during sampling: detach so
            # guidance backprop reaches coords only.
            injected = injected.detach()
        return write(conditioning, injected)

    def step(self, *args, **kwargs):
        """Delegate denoising to the wrapped model."""
        return self.inner.step(*args, **kwargs)

    def initialize_from_prior(self, *args, **kwargs):
        """Delegate prior initialization to the wrapped model."""
        return self.inner.initialize_from_prior(*args, **kwargs)
