"""Tests for CompositeReward (weighted sum of reward functions).

Validates that CompositeReward:
1. Returns the weighted sum of its sub-rewards.
2. Defaults to equal weights of 1.0 and rejects mismatched weight counts.
3. Is differentiable w.r.t. coordinates (gradients flow through every sub-reward).
4. Forwards the optional prepare() hook only to sub-rewards that define it.
"""

import pytest
import torch
from sampleworks.core.rewards.composite import CompositeReward


class _SumSquares:
    """Toy reward: sum of squared coordinates."""

    def __call__(self, coordinates, *args, **kwargs):
        return coordinates.pow(2).sum()


class _Sum:
    """Toy reward: sum of coordinates (no prepare() hook)."""

    def __call__(self, coordinates, *args, **kwargs):
        return coordinates.sum()


class _Preparable:
    """Toy reward that records the atom array passed to its prepare() hook."""

    def __init__(self):
        self.prepared_with = None

    def prepare(self, atom_array):
        self.prepared_with = atom_array
        return self

    def __call__(self, coordinates, *args, **kwargs):
        return coordinates.sum()


def _coords():
    return torch.randn(2, 5, 3)


def test_weighted_sum_matches_manual():
    coords = _coords()
    comp = CompositeReward([_SumSquares(), _Sum()], [2.0, 3.0])
    out = comp(coords, None, None, None)
    expected = 2.0 * coords.pow(2).sum() + 3.0 * coords.sum()
    assert torch.allclose(out, expected)


def test_default_weights_are_one():
    coords = _coords()
    comp = CompositeReward([_SumSquares(), _Sum()])  # weights omitted -> all 1.0
    out = comp(coords, None, None, None)
    expected = coords.pow(2).sum() + coords.sum()
    assert torch.allclose(out, expected)


def test_mismatched_weights_raises():
    with pytest.raises(ValueError):
        CompositeReward([_SumSquares(), _Sum()], [1.0])


def test_gradient_flows_to_coordinates():
    coords = _coords().requires_grad_(True)
    comp = CompositeReward([_SumSquares(), _Sum()], [2.0, 3.0])
    comp(coords, None, None, None).backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()
    # d/dx [2*sum(x^2) + 3*sum(x)] = 4x + 3
    assert torch.allclose(coords.grad, 4.0 * coords.detach() + 3.0)


def test_prepare_forwards_only_to_preparable_rewards():
    prep = _Preparable()
    plain = _Sum()  # has no prepare()
    comp = CompositeReward([prep, plain])
    atom_array = object()  # sentinel

    result = comp.prepare(atom_array)

    assert result is comp  # returns self for chaining
    assert prep.prepared_with is atom_array  # forwarded to the preparable sub-reward
    # `plain` has no prepare(); the composite must skip it without error
