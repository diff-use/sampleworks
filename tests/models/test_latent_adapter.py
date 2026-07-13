"""Tests for the Step-1 latent injection adapter (pseudo-MLP ``k*s + b``).

These run on CPU with no model checkpoints, using a minimal mock wrapper whose
conditioning carries a single-representation tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor

from sampleworks.models.latent_adapter import (
    AffineInjector,
    AttrLatentIO,
    LatentAdaptedWrapper,
    LatentInjector,
    LatentIO,
)
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput


@dataclass(frozen=True)
class _SingleRepConditioning:
    """Minimal conditioning carrying a single representation ``s`` plus sidecar state."""

    s: Tensor
    sidecar: str = "untouched"  # stands in for non-tensor state (e.g. RF3 chiral arrays)


class _SingleRepWrapper:
    """Mock FlowModelWrapper returning a deterministic single representation."""

    def __init__(self, s: Tensor):
        self._s = s

    def featurize(self, structure: dict, **kwargs: Any) -> GenerativeModelInput[_SingleRepConditioning]:
        return GenerativeModelInput(
            x_init=torch.zeros(1, self._s.shape[-2], 3),
            conditioning=_SingleRepConditioning(s=self._s.clone()),
        )

    def step(self, x_t, t, *, features=None):
        return torch.zeros_like(x_t)

    def initialize_from_prior(self, batch_size, features=None, *, shape=None):
        return torch.randn(batch_size, self._s.shape[-2], 3)


@pytest.fixture
def s_tensor() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 8, 4)  # [batch, tokens, d_s]


# --- AffineInjector: the pseudo-MLP ------------------------------------------


def test_affine_injector_identity_at_init(s_tensor: Tensor):
    """Default k=1, b=0 must be the identity."""
    out = AffineInjector()(s_tensor)
    torch.testing.assert_close(out, s_tensor)


def test_affine_injector_applies_kx_plus_b(s_tensor: Tensor):
    out = AffineInjector(k_init=2.0, b_init=3.0)(s_tensor)
    torch.testing.assert_close(out, 2.0 * s_tensor + 3.0)


def test_affine_injector_satisfies_injector_protocol():
    assert isinstance(AffineInjector(), LatentInjector)


# --- AttrLatentIO: the only model-specific knowledge -------------------------


def test_attr_latent_io_roundtrip(s_tensor: Tensor):
    io = AttrLatentIO(single_attr="s")
    cond = _SingleRepConditioning(s=s_tensor)
    assert io.read_single(cond) is s_tensor
    new = io.write_single(cond, s_tensor * 5)
    torch.testing.assert_close(new.s, s_tensor * 5)


def test_attr_latent_io_preserves_sidecar_state(s_tensor: Tensor):
    """replace() must leave non-tensor state (e.g. RF3 chiral arrays) intact."""
    io = AttrLatentIO(single_attr="s")
    cond = _SingleRepConditioning(s=s_tensor, sidecar="keep-me")
    new = io.write_single(cond, s_tensor + 1)
    assert new.sidecar == "keep-me"


def test_attr_latent_io_satisfies_protocol():
    assert isinstance(AttrLatentIO("s"), LatentIO)


def test_attr_latent_io_missing_attr_returns_none(s_tensor: Tensor):
    io = AttrLatentIO(single_attr="does_not_exist")
    assert io.read_single(_SingleRepConditioning(s=s_tensor)) is None


# --- LatentAdaptedWrapper: the decorator -------------------------------------


def test_adapted_wrapper_satisfies_flow_protocol(s_tensor: Tensor):
    wrapped = LatentAdaptedWrapper(_SingleRepWrapper(s_tensor), AttrLatentIO("s"))
    assert isinstance(wrapped, FlowModelWrapper)


def test_identity_injection_is_zero_impact(s_tensor: Tensor):
    """k=1,b=0 => featurized single rep is unchanged: the zero-impact guarantee."""
    inner = _SingleRepWrapper(s_tensor)
    wrapped = LatentAdaptedWrapper(inner, AttrLatentIO("s"), AffineInjector())
    base = inner.featurize({})
    out = wrapped.featurize({})
    torch.testing.assert_close(out.conditioning.s, base.conditioning.s)


def test_affine_injection_is_applied(s_tensor: Tensor):
    inner = _SingleRepWrapper(s_tensor)
    wrapped = LatentAdaptedWrapper(inner, AttrLatentIO("s"), AffineInjector(2.0, 3.0))
    out = wrapped.featurize({})
    torch.testing.assert_close(out.conditioning.s, 2.0 * inner.featurize({}).conditioning.s + 3.0)


def test_sampling_mode_detaches_latent(s_tensor: Tensor):
    """In sampling mode the injected latent is detached -> guidance hits coords only."""
    wrapped = LatentAdaptedWrapper(
        _SingleRepWrapper(s_tensor), AttrLatentIO("s"), AffineInjector(), training_adapter=False
    )
    out = wrapped.featurize({})
    assert out.conditioning.s.requires_grad is False


def test_training_mode_keeps_graph(s_tensor: Tensor):
    """In training mode the latent stays attached so the injector can be optimized."""
    wrapped = LatentAdaptedWrapper(
        _SingleRepWrapper(s_tensor), AttrLatentIO("s"), AffineInjector(), training_adapter=True
    )
    out = wrapped.featurize({})
    assert out.conditioning.s.requires_grad is True
    # gradient flows back to the injector's k and b
    out.conditioning.s.sum().backward()
    assert wrapped.injector.k.grad is not None
    assert wrapped.injector.b.grad is not None


def test_step_and_prior_delegate(s_tensor: Tensor):
    inner = _SingleRepWrapper(s_tensor)
    wrapped = LatentAdaptedWrapper(inner, AttrLatentIO("s"))
    x = torch.randn(2, 8, 3)
    torch.testing.assert_close(wrapped.step(x, torch.tensor(0.5), features=None), torch.zeros_like(x))
    assert wrapped.initialize_from_prior(2, shape=(8, 3)).shape == (2, 8, 3)
