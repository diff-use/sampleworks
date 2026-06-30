"""Reward-agnostic contract tests, parametrized over every reward function.

These tests exercise behavior that any `RewardFunctionProtocol` implementation must
satisfy (see `sampleworks.core.rewards.protocol`), and run against both
`RealSpaceRewardFunction` and `StructureFactorRewardFunction` via the `reward_case`
fixture. They were extracted from `test_real_space_density_reward.py` (whose header asked
for exactly this generalization).

Only *reward-agnostic* checks live here:
- Absolute loss thresholds ARE shared, but the value is per-reward (see `_LOSS_THRESHOLDS`):
  both losses are now on normalized, interpretable scales — RealSpace is normalized density
  (sigma units) and SFC scores normalized E-values (`normalize_amplitude=True`,
  unit-variance per resolution shell). Correlation is still checked relatively
  (true < perturbed < random, monotonic) on top of the absolute bar.
- The batch=1 occupancy-gradient check IS shared (both rewards consume the single conformer's
  occupancy); see `test_gradients_wrt_occupancies` below. Tests that change atom count /
  topology (single-atom) or rely on `precompute_unique_combinations` /
  `structure_to_reward_input` stay in the reward-specific files, since SFC fixes topology at
  `prepare()` and does not implement those methods. Per-conformer (batch>1) occupancy/B is
  also SF-specific: SFC has no batch occupancy/B axis, so the SF reward rejects non-broadcast
  input (see `test_structure_factor_reward.py`).

Each reward runs against its own self-consistent data bundle (see `reward_case`): the SFC
case uses a committed crystal-frame chain-A 1vme model cif + matching synthetic MTZ
(`1vme_final_crystalframe_0.5occA_0.5occB_1.80A.{cif,mtz}`), while real_space keeps the
recentered carved cif + `.ccp4` map. The SFC case still skips gracefully if its MTZ is
absent (see the `mtz_path_1vme` fixture).
"""

from dataclasses import dataclass

import pytest
import torch
from sampleworks.core.rewards.protocol import RewardFunctionProtocol

from tests.rewards.reward_input_helpers import build_scattering_indices


# Every test exercises CUDA-targeted reward code on the `device` fixture (try_gpu), so the
# whole module is gpu-marked. Deliberately NOT `slow`: measured warm per-test time is <2.5s
# (slowest is the SFC gradient-descent loop at ~2.4s; the rest are sub-second). The ~11s of
# fixed cost is one-time import + session-scoped reward construction, which is paid once per
# pytest invocation and cannot be skipped by `slow`-marking these tests.
pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class RewardCase:
    """A reward function bundled with a standard set of per-atom inputs for one structure."""

    name: str
    reward_function: RewardFunctionProtocol
    coords: torch.Tensor  # [N, 3]
    elements: torch.Tensor  # [N]
    b_factors: torch.Tensor  # [N]
    occupancies: torch.Tensor  # [N]

    def batch(self, n: int = 1, coords: torch.Tensor | None = None) -> dict:
        """Build a [B, ...] kwargs dict by expanding the stored inputs over a batch."""
        c = self.coords if coords is None else coords
        return dict(
            coordinates=c.unsqueeze(0).expand(n, -1, -1),
            elements=self.elements.unsqueeze(0).expand(n, -1),
            b_factors=self.b_factors.unsqueeze(0).expand(n, -1),
            occupancies=self.occupancies.unsqueeze(0).expand(n, -1),
        )


# Per-reward bundles: each param resolves its OWN coordinates/atom_array/reward so the
# inputs are self-consistent with that reward's target. real_space uses the recentered
# carved 1vme (matches its .ccp4 map frame); structure_factor uses the crystal-frame
# chain-A model the synthetic MTZ was generated from (recentering corrupts SF symmetry
# mates). Sharing one structure across both would feed SFC coordinates from the wrong
# frame.
_REWARD_BUNDLES = {
    "real_space": ("test_coordinates_1vme", "reward_function_1vme"),
    "structure_factor": ("test_coordinates_1vme_sf", "reward_function_1vme_sf"),
}

# Absolute loss bar for the TRUE structure, per reward. Both losses are MSE on normalized
# quantities (RealSpace: density in sigma units; SFC: E-values, unit-variance per shell), so
# a wrong/random model scores ~O(1) while the true model scores ~0. Measured for the SFC
# case (synthetic MTZ from the same model): true E-MSE ~2e-14, 0.5A-perturbed ~0.34,
# random ~0.44 — so 0.1 sits an order of magnitude above numerical zero and ~3x below the
# smallest perturbation signal.
_LOSS_THRESHOLDS = {
    "real_space": 1.0,
    "structure_factor": 0.1,
}


@pytest.fixture(params=list(_REWARD_BUNDLES))
def reward_case(request, device: torch.device) -> RewardCase:
    coords_fixture, reward_fixture = _REWARD_BUNDLES[request.param]
    coords, atom_array = request.getfixturevalue(coords_fixture)
    elements = build_scattering_indices(atom_array, device)
    b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
    occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

    reward_function = request.getfixturevalue(reward_fixture)  # SFC skips here if the MTZ is absent
    return RewardCase(request.param, reward_function, coords, elements, b_factors, occupancies)


class TestRewardFunctionInterface:
    """Output type/shape contract any reward must satisfy."""

    def test_reward_function_conforms_to_protocol(self, reward_case):
        assert isinstance(reward_case.reward_function, RewardFunctionProtocol)

    def test_reward_function_call_shapes(self, reward_case):
        """Single [N,3] and batched [B,N,3] inputs both return a scalar."""
        for n in (1, 3):
            loss = reward_case.reward_function(**reward_case.batch(n))
            assert loss.shape == torch.Size([])
            assert loss.ndim == 0

    def test_reward_function_output_is_scalar(self, reward_case):
        loss = reward_case.reward_function(**reward_case.batch(1))
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0.0

    def test_reward_function_deterministic(self, reward_case):
        """Same inputs give the same output."""
        loss1 = reward_case.reward_function(**reward_case.batch(1))
        loss2 = reward_case.reward_function(**reward_case.batch(1))
        torch.testing.assert_close(loss1, loss2)


class TestRewardCorrelation:
    """Loss must rank structures by correctness (absolute bar + relative ordering)."""

    def test_perfect_structure_has_low_loss(self, reward_case):
        """The true structure clears the per-reward absolute bar (normalized scales)."""
        loss = reward_case.reward_function(**reward_case.batch(1)).item()
        assert loss < _LOSS_THRESHOLDS[reward_case.name]

    def test_perturbed_structure_has_higher_loss(self, reward_case):
        loss_true = reward_case.reward_function(**reward_case.batch(1)).item()

        torch.manual_seed(42)
        coords_perturbed = reward_case.coords + torch.randn_like(reward_case.coords) * 0.5
        loss_perturbed = reward_case.reward_function(
            **reward_case.batch(1, coords=coords_perturbed)
        ).item()

        assert loss_perturbed > loss_true

    def test_random_structure_has_higher_loss(self, reward_case):
        """Random coordinates score worse than the true structure.

        The original RealSpace test required random > 2x true; that 2x margin is a
        RealSpace-scale assumption, so the shared contract only asserts the robust
        ordering (random > true). Tighten per-reward if desired once SFC data lands.
        """
        loss_true = reward_case.reward_function(**reward_case.batch(1)).item()

        torch.manual_seed(42)
        coords_random = torch.randn_like(reward_case.coords) * 10.0
        loss_random = reward_case.reward_function(
            **reward_case.batch(1, coords=coords_random)
        ).item()

        assert loss_random > loss_true

    def test_loss_monotonic_with_perturbation(self, reward_case):
        """Loss increases (non-strictly) with per-atom perturbation magnitude."""
        torch.manual_seed(42)
        # Per-atom Gaussian displacement (NOT normalized over the whole tensor): `scale` is the
        # per-component std in Angstrom, so each step is a real, above-noise displacement.
        direction = torch.randn_like(reward_case.coords)

        losses = []
        for scale in [0.0, 0.1, 0.25, 0.5, 1.0]:
            coords_pert = reward_case.coords + direction * scale
            losses.append(
                reward_case.reward_function(**reward_case.batch(1, coords=coords_pert)).item()
            )

        for i in range(len(losses) - 1):
            assert losses[i + 1] >= losses[i] - 1e-6


class TestRewardGradientFlow:
    """Gradients must flow w.r.t. coordinates for guided sampling / optimization."""

    def test_gradients_wrt_coordinates(self, reward_case):
        coords_opt = reward_case.coords.clone().unsqueeze(0).requires_grad_(True)
        loss = reward_case.reward_function(
            coordinates=coords_opt,
            elements=reward_case.elements.unsqueeze(0),
            b_factors=reward_case.b_factors.unsqueeze(0),
            occupancies=reward_case.occupancies.unsqueeze(0),
        )
        loss.backward()

        assert coords_opt.grad is not None
        assert torch.all(torch.isfinite(coords_opt.grad))
        assert torch.any(coords_opt.grad != 0)
        # Magnitude sanity: non-zero but not exploding (NaN/inf already ruled out above).
        assert 0 < coords_opt.grad.norm().item() < 1e6

    def test_gradients_wrt_occupancies(self, reward_case):
        """Occupancy must be differentiable (batch=1).

        At batch=1 every reward consumes the single conformer's occupancy: RealSpace feeds
        the full occupancies tensor, and SFC wires occupancies[0] (the only row) into the
        graph. Per-conformer (batch>1) occupancy is NOT a shared contract — SFC has no batch
        occupancy axis (only occupancies[0] is used); that asymmetry lives in the SF-specific
        tests. Mirrors test_gradients_wrt_coordinates: finiteness only, no nonzero assertion
        (the gradient can vanish near the synthetic-data minimum).
        """
        occupancies_opt = reward_case.occupancies.clone().unsqueeze(0).requires_grad_(True)
        loss = reward_case.reward_function(
            coordinates=reward_case.coords.unsqueeze(0),
            elements=reward_case.elements.unsqueeze(0),
            b_factors=reward_case.b_factors.unsqueeze(0),
            occupancies=occupancies_opt,
        )
        loss.backward()

        assert occupancies_opt.grad is not None
        assert torch.all(torch.isfinite(occupancies_opt.grad))

    def test_gradient_descent_improves_loss(self, reward_case):
        torch.manual_seed(42)
        perturbation = torch.randn_like(reward_case.coords) * 0.5
        coords_opt = (reward_case.coords + perturbation).unsqueeze(0).requires_grad_(True)
        optimizer = torch.optim.Adam([coords_opt], lr=0.01)

        def loss_fn():
            return reward_case.reward_function(
                coordinates=coords_opt,
                elements=reward_case.elements.unsqueeze(0),
                b_factors=reward_case.b_factors.unsqueeze(0),
                occupancies=reward_case.occupancies.unsqueeze(0),
            )

        loss_initial = loss_fn().item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
        loss_final = loss_fn().item()

        assert loss_final < loss_initial


# Batch sizes span a single conformer up to a 20-member ensemble. All run under the
# module-level gpu mark; even batch=20 is sub-second (measured ~0.5s), so none is `slow`.
@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1, id="single"),
        pytest.param(3, id="ensemble-3"),
        pytest.param(5, id="ensemble-5"),
        pytest.param(20, id="ensemble-20"),
    ],
)
class TestRewardBatchHandling:
    """Various batch shapes (incl. a large ensemble) produce a valid finite scalar."""

    def test_batch_shape(self, reward_case, batch_size):
        loss = reward_case.reward_function(**reward_case.batch(batch_size))
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


class TestRewardEdgeCases:
    """Edge cases that keep the (prepared) atom topology intact."""

    def test_numerical_stability(self, reward_case):
        """Extreme coordinate values still produce a finite loss."""
        coords_far = reward_case.coords + torch.randn_like(reward_case.coords) * 1e9
        loss = reward_case.reward_function(**reward_case.batch(1, coords=coords_far))
        assert torch.isfinite(loss)
