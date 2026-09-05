"""Tests for the two-phase reward preparation hook."""

import torch
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
)

from tests.mocks import MockGradientRewardFunction, MockPreparableRewardFunction
from tests.utils.atom_array_builders import build_test_atom_array


def test_preparable_reward_satisfies_both_protocols():
    """A reward with prepare() is still an ordinary reward function."""
    reward = MockPreparableRewardFunction()

    assert isinstance(reward, RewardFunctionProtocol)
    assert isinstance(reward, PreparableRewardFunctionProtocol)


def test_plain_reward_is_not_preparable():
    """Structural typing must not classify every reward as two-phase."""
    assert not isinstance(MockGradientRewardFunction(), PreparableRewardFunctionProtocol)


def test_prepare_hook_forwards_atom_array_and_device():
    """The reward is bound to the atom array and device the caller passed."""
    reward = MockPreparableRewardFunction()

    prepare_reward_if_needed(reward, build_test_atom_array(n_atoms=7), device=torch.device("cpu"))

    assert reward.prepared_with == [(7, "cpu")]


def test_prepare_hook_is_a_no_op_for_rewards_that_do_not_need_it():
    """Callers can prepare unconditionally, so one-phase rewards must be untouched."""
    reward = MockGradientRewardFunction()

    prepare_reward_if_needed(reward, build_test_atom_array(n_atoms=4), device="cpu")

    # 0.5 * ||ones(1, 4, 3)||^2: the reward scores exactly as if the hook had never run.
    assert reward(torch.ones(1, 4, 3)) == 6.0


def test_prepare_is_rerunnable_for_a_new_topology():
    """Preparing again rebinds, so a reward can move between topologies or devices."""
    reward = MockPreparableRewardFunction()

    prepare_reward_if_needed(reward, build_test_atom_array(n_atoms=3), device="cpu")
    prepare_reward_if_needed(reward, build_test_atom_array(n_atoms=5), device="cpu")

    assert reward.prepared_with == [(3, "cpu"), (5, "cpu")]
