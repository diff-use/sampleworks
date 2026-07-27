# Archived: the injector / adapted-wrapper approach (superseded, not in the build)

This is the **Phase-0 scaffold** for IT-opt latent optimization — an *alternative* to the shipped
`LatentOptimization` scaler ([`core/scalers/latent_optimization.py`](../../src/sampleworks/core/scalers/latent_optimization.py)).
Instead of optimizing the latent tensor directly (the "direct-leaf" approach that shipped), it
trains a small **transform at the `featurize` boundary**:

- `AffineInjector` — `k*s + b`, two scalar parameters (**the "kx+b" one**). Identity at init.
- `DeltaInjector` — `s + Δ`, a zero-initialized full-shape delta (identity at init).
- `LatentAdaptedWrapper` — a decorator over any `FlowModelWrapper` that injects the transformed
  latent at `featurize` and delegates `step` / `initialize_from_prior` verbatim.
- `LatentInjector` — the protocol the injectors satisfy.

## Why it's archived, not deleted

It was **never wired into `_run_guidance`**, and the direct-leaf scaler superseded it:

- `DeltaInjector` (`s + Δ`, optimize `Δ`) is *mathematically the same thing* the scaler already does
  by making the latent a leaf; the scaler's anchor `‖leaf − baseline‖²` is exactly `‖Δ‖²`.
- `AffineInjector` (two scalars) is too weak to represent a real latent perturbation.

So it is dead relative to the build. It is kept because the **structured** version of this idea is
genuinely useful future work: the reference `it_opt` implements a whole `update_mode` family
(`low_rank` = `Δ = UᵀV`, `per_residue_bias`, `channel_scale`, …) that gives regularized,
drift-resistant perturbations of `s`/`z`. This wrapper/injector seam is the natural place to hang
those — an injector whose `forward` applies a low-rank `Δ` slots straight in.

## Not in the build, not linted

This code lives in a doc (a fenced block), so it is not imported by any main-code path and is not
checked by CI. To revive it, drop it back into
[`src/sampleworks/models/latent_adapter.py`](../../src/sampleworks/models/latent_adapter.py) and
restore its imports:

```python
import copy  # noqa: F401  (used by _write_attr in the host module)
import dataclasses  # noqa: F401
import torch
from torch import nn, Tensor
from typing import Protocol, runtime_checkable

from sampleworks.models.protocol import GenerativeModelInput
# LatentIO is defined alongside this code in latent_adapter.py
```

## The code

```python
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


@runtime_checkable
class LatentInjector(Protocol):
    """A transform applied to a representation (single or pair). Swappable."""

    def __call__(self, latent: Tensor) -> Tensor:
        """Map a representation tensor to a modified representation tensor."""
        ...
```
