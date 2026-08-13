"""Tests for the two-phase reward preparation hook."""

import torch
from biotite.structure import AtomArray
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
)


class PlainReward:
    """Reward that is fully configured at construction time."""

    def __call__(
        self,
        coordinates: torch.Tensor,
        elements: torch.Tensor | None = None,
        b_factors: torch.Tensor | None = None,
        occupancies: torch.Tensor | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (coordinates**2).sum()


class PreparableReward(PlainReward):
    """Reward that binds to the model topology, recording what it was given."""

    def __init__(self):
        self.prepared_with: list[tuple[int, str]] = []
        self.calls_before_prepare = 0

    def prepare(self, atom_array: AtomArray, *, device: torch.device | str = "cpu") -> None:
        self.prepared_with.append((atom_array.array_length(), str(device)))

    def __call__(self, coordinates: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.prepared_with:
            self.calls_before_prepare += 1
        return super().__call__(coordinates)


def make_atom_array(n_atoms: int = 4) -> AtomArray:
    """Build a minimal AtomArray of carbons at the origin."""
    atom_array = AtomArray(n_atoms)
    atom_array.coord = torch.zeros(n_atoms, 3).numpy()
    atom_array.element = ["C"] * n_atoms
    return atom_array


def test_preparable_reward_satisfies_both_protocols():
    reward = PreparableReward()

    assert isinstance(reward, RewardFunctionProtocol)
    assert isinstance(reward, PreparableRewardFunctionProtocol)


def test_plain_reward_is_not_preparable():
    assert not isinstance(PlainReward(), PreparableRewardFunctionProtocol)


def test_prepare_hook_forwards_atom_array_and_device():
    reward = PreparableReward()
    atom_array = make_atom_array(7)

    prepare_reward_if_needed(reward, atom_array, device=torch.device("cpu"))

    assert reward.prepared_with == [(7, "cpu")]


def test_prepare_hook_is_a_no_op_for_rewards_that_do_not_need_it():
    reward = PlainReward()

    prepare_reward_if_needed(reward, make_atom_array(), device="cpu")

    assert reward(torch.ones(1, 4, 3)) == 12.0


def test_prepare_is_rerunnable_for_a_new_topology():
    reward = PreparableReward()

    prepare_reward_if_needed(reward, make_atom_array(3), device="cpu")
    prepare_reward_if_needed(reward, make_atom_array(5), device="cpu")

    assert reward.prepared_with == [(3, "cpu"), (5, "cpu")]
