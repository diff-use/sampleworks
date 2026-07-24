"""Read and write a model's post-trunk latents (IT-opt plumbing).

The latents are the *single* representation ``s`` and the *pair* representation ``z``. They live on
the model's *conditioning* -- the bundle of inputs ``featurize`` produces and hands to the ``step``
(denoise) call, a (frozen) dataclass on which ``s`` and ``z`` are named attributes. This module lets
inference-time latent optimization (IT-opt) read those attributes, swap them out, and tune them
against an experimental reward. See ``docs/IT_OPTIMIZATION_PLAN.md``.

What each piece is for
----------------------
- :class:`AttrLatentIO` -- reads/writes ``s`` and ``z`` by attribute name. This is the piece the
  IT-opt scaler (``core/scalers/latent_optimization.py``) actually uses: it reads the baseline
  latents once, holds its own optimizable copies, and writes them back each step.
- :class:`AffineInjector` / :class:`DeltaInjector` / :class:`LatentAdaptedWrapper` -- an alternative
  path that instead trains a small transform at the ``featurize`` boundary. Kept for that mode.

Design goals
------------
- **Minimal, model-agnostic.** The only model-specific knowledge is *which attribute of the
  conditioning holds each representation* -- ``"s"``/``"z"`` for Boltz, ``"s_trunk"``/``"z_trunk"``
  for Protenix/RF3. No model package is imported here, so a 4th or 5th model is one more pair of
  strings, not new code.
- **Zero-impact by default.** With identity injectors the wrapper behaves exactly like the model it
  wraps; :class:`DeltaInjector` starts at zero, so its first output is exactly the trunk baseline.
- **Gradient isolation.** During ordinary sampling the injected latents are detached, so existing
  coordinate-only guidance is unchanged; latents are trained only when ``training_adapter=True``.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Protocol, runtime_checkable

import torch
from torch import nn, Tensor

from sampleworks.models.protocol import GenerativeModelInput


# ---- Reading the type hints in this file --------------------------------------
# A ": type" after a name (or "-> type" after a function) is only a HINT -- it CLAIMS what a
# value should be, but nothing enforces it: pass the wrong type and Python still runs the code,
# and deleting every hint changes nothing. Hints are for humans (and optional checkers like ty).
# (In the table, "|" means "or".)
#
#   with the hint                    plain Python               what it claims (useless in runtime)
#   k_init: float = 1.0              k_init = 1.0               should be a float
#   training_adapter: bool = False   training_adapter = False   should be True or False
#   attr: str                        attr                       should be a string
#   single: Tensor                   single                     should be a Tensor (~ a numpy array)
#   pair_attr: str | None = None     pair_attr = None           should be a str, or None
#   d: dict[str, str] = {}           d = {}                     should be a dict, string -> string
#   f(...) -> Tensor | None          f(...)                     f should return a Tensor or None
#   f(...) -> GenerativeModelInput   f(...)                     f should return that object
# -------------------------------------------------------------------------------

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


# "conditioning" throughout this file is the model's conditioning object: a (frozen) dataclass
# carrying the latents ``s``/``z`` (and other cached state) as named attributes. These two helpers
# just get, or replace, one such attribute on it.
def _read_attr(conditioning, attr: str) -> Tensor | None:
    """Return ``conditioning.<attr>`` (a representation tensor), or None if it is absent."""
    if conditioning is None:
        return None
    return getattr(conditioning, attr, None)


def _write_attr(conditioning, attr: str, value: Tensor):
    """Return a copy of ``conditioning`` with ``<attr>`` set to ``value`` (dataclass-aware)."""
    if dataclasses.is_dataclass(conditioning) and not isinstance(conditioning, type):
        return dataclasses.replace(conditioning, **{attr: value})  # ty: ignore
    # Fallback for non-dataclass conditioning: mutate a shallow copy.
    new_cond = copy.copy(conditioning)
    setattr(new_cond, attr, value)
    return new_cond


class AttrLatentIO:
    """A general-purpose :class:`LatentIO` addressing each representation by attribute name.

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

    def read_single(self, conditioning) -> Tensor | None:
        return _read_attr(conditioning, self.single_attr)

    def write_single(self, conditioning, single: Tensor):
        return _write_attr(conditioning, self.single_attr, single)

    def read_pair(self, conditioning) -> Tensor | None:
        if self.pair_attr is None:
            return None
        return _read_attr(conditioning, self.pair_attr)

    def write_pair(self, conditioning, pair: Tensor):
        if self.pair_attr is None:
            return conditioning
        return _write_attr(conditioning, self.pair_attr, pair)


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


class LatentAdaptedWrapper:
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
        latent_io: LatentIO,
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

    def featurize(self, structure: dict, **kwargs) -> GenerativeModelInput:
        """Featurize via ``inner``, then inject into the single (and pair) reps."""
        feats = self.inner.featurize(structure, **kwargs)
        conditioning = feats.conditioning  # the model's bundle of inputs, holding the s/z latents

        # Replace s with single_injector(s), then z with pair_injector(z). When an injector (or
        # that representation) is absent, _inject returns the SAME conditioning object back -- which
        # is what the identity check below reads as "nothing changed, pass the original through".
        conditioning = self._inject(
            conditioning,
            self.latent_io.read_single,
            self.latent_io.write_single,
            self.single_injector,
        )
        conditioning = self._inject(
            conditioning,
            self.latent_io.read_pair,
            self.latent_io.write_pair,
            self.pair_injector,
        )

        if conditioning is feats.conditioning:
            return feats  # nothing was injected; pass the original through unchanged
        return GenerativeModelInput(x_init=feats.x_init, conditioning=conditioning)

    def _inject(self, conditioning, read_rep, write_rep, injector: LatentInjector | None):
        """Run one representation through ``injector`` and write the result back.

        ``read_rep`` and ``write_rep`` are *functions* -- the two ``latent_io`` accessors for the
        representation being injected (e.g. ``read_single`` / ``write_single`` for ``s``). Returns
        ``conditioning`` untouched when there is no injector, or that representation is absent.
        """
        if injector is None:
            return conditioning
        latent = read_rep(conditioning)
        if latent is None:
            return conditioning
        injected = injector(latent)
        if not self.training_adapter:
            # Honour the constant-conditioning contract during sampling: detach so
            # guidance backprop reaches coords only.
            injected = injected.detach()
        return write_rep(conditioning, injected)

    def step(self, *args, **kwargs):
        """Delegate denoising to the wrapped model."""
        return self.inner.step(*args, **kwargs)

    def initialize_from_prior(self, *args, **kwargs):
        """Delegate prior initialization to the wrapped model."""
        return self.inner.initialize_from_prior(*args, **kwargs)


# ============================ protocols (templates / interfaces) ============================
# A Protocol is just a checklist of methods a class must have -- think a C header, or an
# abstract base class, but *structural*: any class that already has these methods counts as a
# LatentIO / LatentInjector without inheriting from anything. These definitions do not run; they
# only document the contract and let the ty checker flag a mismatch. @runtime_checkable also lets
# isinstance(obj, LatentIO) work, which the unit tests use. The real implementations are above:
# AttrLatentIO is the LatentIO; AffineInjector / DeltaInjector are the LatentInjectors.


@runtime_checkable
class LatentIO(Protocol):
    """Reads/writes the single and pair representations on a conditioning object.

    Implementations encapsulate the *only* model-specific knowledge in this
    module: where each representation lives on the conditioning object. Pair-rep
    methods return ``None`` / pass through unchanged when the implementation is
    single-rep only. Each write returns a copy of the conditioning with one
    representation replaced.
    """

    def read_single(self, conditioning) -> Tensor | None:
        """Return the single representation tensor, or ``None`` if unavailable."""
        ...

    def write_single(self, conditioning, single: Tensor):
        """Return a copy of ``conditioning`` with the single representation replaced."""
        ...

    def read_pair(self, conditioning) -> Tensor | None:
        """Return the pair representation tensor, or ``None`` if unavailable."""
        ...

    def write_pair(self, conditioning, pair: Tensor):
        """Return a copy of ``conditioning`` with the pair representation replaced."""
        ...


@runtime_checkable
class LatentInjector(Protocol):
    """A transform applied to a representation (single or pair). Swappable."""

    def __call__(self, latent: Tensor) -> Tensor:
        """Map a representation tensor to a modified representation tensor."""
        ...
