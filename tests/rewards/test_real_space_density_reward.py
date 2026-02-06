"""Tests for real-space density reward function.

This module tests the RewardFunction class from
sampleworks.core.rewards.real_space_density, validating that it:
1. Produces outputs with high correlation with underlying electron density
2. Is properly vmappable and usable in guidance scalers (FK steering)
3. Handles gradients correctly for optimization
4. Works with various batch shapes and edge cases

# TODO: Generalize this across reward functions, move to different file
"""

from functools import partial
from typing import cast

import einx
import pytest
import torch
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ELEMENT_TO_ATOMIC_NUM,
)
from sampleworks.core.rewards.real_space_density import (
    RealSpaceRewardFunction,
)


@pytest.mark.slow
class TestRewardFunctionBasics:
    """Test basic functionality of the RewardFunction class."""

    def test_reward_function_initialization(self, reward_function_1vme):
        """Test that RewardFunction can be instantiated."""
        assert reward_function_1vme is not None
        assert isinstance(reward_function_1vme, RealSpaceRewardFunction)

    def test_reward_function_call_shapes(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test output shapes are correct for various input shapes."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        # Test single structure [N_atoms, 3]
        loss = reward_function_1vme(
            coordinates=coords,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

        # Test batch [batch_size, N_atoms, 3]
        batch_size = 3
        loss = reward_function_1vme(
            coordinates=coords.unsqueeze(0).expand(batch_size, -1, -1),
            elements=elements.unsqueeze(0).expand(batch_size, -1),
            b_factors=b_factors.unsqueeze(0).expand(batch_size, -1),
            occupancies=occupancies.unsqueeze(0).expand(batch_size, -1),
        )
        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_reward_function_output_is_scalar(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that single structure returns scalar loss."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        loss = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0.0

    def test_reward_function_deterministic(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that same inputs give same outputs."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        loss1 = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        loss2 = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        torch.testing.assert_close(loss1, loss2)


@pytest.mark.slow
class TestDensityCorrelation:
    """Test that reward function correlates with underlying electron density."""

    def test_perfect_structure_has_low_loss(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that true structure coordinates give low loss."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        loss = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert loss.item() < 1.0

    def test_perturbed_structure_has_higher_loss(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that perturbed coordinates give higher loss than true structure."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        loss_true = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        torch.manual_seed(42)
        perturbation = torch.randn_like(coords) * 0.5
        coords_perturbed = coords + perturbation

        loss_perturbed = reward_function_1vme(
            coordinates=coords_perturbed.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert loss_perturbed.item() > loss_true.item()

    def test_random_structure_has_high_loss(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that random coordinates give very high loss."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        loss_true = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        torch.manual_seed(42)
        coords_random = torch.randn_like(coords) * 10.0

        loss_random = reward_function_1vme(
            coordinates=coords_random.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert loss_random.item() > loss_true.item() * 2.0

    def test_loss_monotonic_with_perturbation(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test loss increases monotonically with perturbation magnitude."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        torch.manual_seed(42)
        perturbation_direction = torch.randn_like(coords)
        perturbation_direction = perturbation_direction / perturbation_direction.norm()

        losses = []
        for magnitude in [0.0, 0.2, 0.5, 1.0, 2.0]:
            coords_pert = coords + perturbation_direction * magnitude
            loss = reward_function_1vme(
                coordinates=coords_pert.unsqueeze(0),
                elements=elements.unsqueeze(0),
                b_factors=b_factors.unsqueeze(0),
                occupancies=occupancies.unsqueeze(0),
            )
            losses.append(loss.item())

        for i in range(len(losses) - 1):
            assert losses[i + 1] >= losses[i]


@pytest.mark.slow
class TestVmapCompatibility:
    """Test vmap functionality for use in FK steering and particle methods."""

    def test_vmap_over_particle_dimension(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test vmap over particle dimension as used in FK steering."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        num_particles = 3
        ensemble_size = 3

        coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=ensemble_size)
        elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=ensemble_size)
        b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=ensemble_size)
        occupancies_batch = einx.rearrange(
            "n -> p e n", occupancies, p=num_particles, e=ensemble_size
        )

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements_batch[0], b_factors_batch[0]
        )

        rf_partial = partial(
            reward_function_1vme,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        result = cast(
            torch.Tensor,
            einx.vmap(
                "p [e n c], p [e n], p [e n], p [e n] -> p",
                coords_batch,
                elements_batch,
                b_factors_batch,
                occupancies_batch,
                op=rf_partial,
            ),
        )

        assert result.shape == torch.Size([num_particles])
        assert torch.all(torch.isfinite(result))

    def test_vmap_with_precomputed_combinations(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test vmap with pre-computed unique combinations."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements, b_factors
        )

        loss_with_precompute = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        loss_without_precompute = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        torch.testing.assert_close(loss_with_precompute, loss_without_precompute)

    def test_vmap_output_shape(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test vmap returns correct shape (num_particles,)."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        for num_particles in [1, 3, 5]:
            coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=1)
            elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=1)
            b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=1)
            occupancies_batch = einx.rearrange("n -> p e n", occupancies, p=num_particles, e=1)

            unique_combinations, inverse_indices = (
                reward_function_1vme.precompute_unique_combinations(
                    elements_batch[0, 0],  # pyright: ignore[reportCallIssue,reportArgumentType]
                    b_factors_batch[0, 0],  # pyright: ignore[reportCallIssue,reportArgumentType]
                )
            )

            rf_partial = partial(
                reward_function_1vme,
                unique_combinations=unique_combinations,
                inverse_indices=inverse_indices,
            )

            result = einx.vmap(
                "p [e n c], p [e n], p [e n], p [e n] -> p",
                coords_batch,
                elements_batch,
                b_factors_batch,
                occupancies_batch,
                op=rf_partial,
            )

            assert result.shape == torch.Size([num_particles])  # pyright: ignore[reportAttributeAccessIssue]

    def test_vmap_consistency(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test vmap results match sequential calls."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        num_particles = 3
        coords_batch = einx.rearrange("n c -> p e n c", coords, p=num_particles, e=1)
        elements_batch = einx.rearrange("n -> p e n", elements, p=num_particles, e=1)
        b_factors_batch = einx.rearrange("n -> p e n", b_factors, p=num_particles, e=1)
        occupancies_batch = einx.rearrange("n -> p e n", occupancies, p=num_particles, e=1)

        unique_combinations, inverse_indices = reward_function_1vme.precompute_unique_combinations(
            elements_batch[0, 0],  # pyright: ignore[reportCallIssue,reportArgumentType]
            b_factors_batch[0, 0],  # pyright: ignore[reportCallIssue,reportArgumentType]
        )

        rf_partial = partial(
            reward_function_1vme,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        )

        result_vmap = einx.vmap(
            "p [e n c], p [e n], p [e n], p [e n] -> p",
            coords_batch,
            elements_batch,
            b_factors_batch,
            occupancies_batch,
            op=rf_partial,
        )

        result_sequential = []
        for i in range(num_particles):
            loss = rf_partial(
                coordinates=coords_batch[i],
                elements=elements_batch[i],
                b_factors=b_factors_batch[i],
                occupancies=occupancies_batch[i],
            )
            result_sequential.append(loss.item())

        result_sequential = torch.tensor(result_sequential, device=result_vmap.device)  # pyright: ignore[reportAttributeAccessIssue]

        torch.testing.assert_close(result_vmap, result_sequential, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
class TestGradientFlow:
    """Test gradient computation for coordinate optimization."""

    def test_gradients_wrt_coordinates(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test gradients flow through coordinates."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        coords_opt = coords.clone().unsqueeze(0).requires_grad_(True)

        loss = reward_function_1vme(
            coordinates=coords_opt,
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        loss.backward()

        assert coords_opt.grad is not None
        assert torch.all(torch.isfinite(coords_opt.grad))
        assert torch.any(coords_opt.grad != 0)

    def test_gradients_wrt_occupancies(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test gradients flow through occupancies when not using pre-computed."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        occupancies_opt = occupancies.clone().unsqueeze(0).requires_grad_(True)

        loss = reward_function_1vme(
            coordinates=coords.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies_opt,
        )

        loss.backward()

        assert occupancies_opt.grad is not None
        assert torch.all(torch.isfinite(occupancies_opt.grad))

    def test_gradient_descent_improves_loss(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test that gradient descent on coordinates reduces loss."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        torch.manual_seed(42)
        perturbation = torch.randn_like(coords) * 0.5
        coords_opt = (coords + perturbation).unsqueeze(0).requires_grad_(True)

        optimizer = torch.optim.Adam([coords_opt], lr=0.01)

        loss_initial = reward_function_1vme(
            coordinates=coords_opt,
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        ).item()

        for _ in range(10):
            optimizer.zero_grad()
            loss = reward_function_1vme(
                coordinates=coords_opt,
                elements=elements.unsqueeze(0),
                b_factors=b_factors.unsqueeze(0),
                occupancies=occupancies.unsqueeze(0),
            )
            loss.backward()
            optimizer.step()

        loss_final = reward_function_1vme(
            coordinates=coords_opt,
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        ).item()

        assert loss_final < loss_initial

    def test_gradient_magnitudes_reasonable(
        self, reward_function_1vme, test_coordinates_1vme, device
    ):
        """Test gradient magnitudes are in reasonable range."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        coords_opt = coords.clone().unsqueeze(0).requires_grad_(True)

        loss = reward_function_1vme(
            coordinates=coords_opt,
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        loss.backward()

        assert coords_opt.grad is not None
        grad_norm = coords_opt.grad.norm().item()

        assert grad_norm > 0
        assert grad_norm < 1e6


@pytest.mark.slow
@pytest.mark.parametrize(
    "shape",
    [
        (1, "N_atoms", 3),
        (3, "N_atoms", 3),
        (5, "N_atoms", 3),
    ],
    ids=["single", "ensemble-3", "ensemble-5"],
)
class TestBatchHandling:
    """Test handling of various batch shapes."""

    def test_batch_shape(self, reward_function_1vme, test_coordinates_1vme, device, shape):
        """Test various batch shapes produce valid outputs."""
        coords, atom_array = test_coordinates_1vme
        batch_size = shape[0]

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
        elements_batch = elements.unsqueeze(0).expand(batch_size, -1)
        b_factors_batch = b_factors.unsqueeze(0).expand(batch_size, -1)
        occupancies_batch = occupancies.unsqueeze(0).expand(batch_size, -1)

        loss = reward_function_1vme(
            coordinates=coords_batch,
            elements=elements_batch,
            b_factors=b_factors_batch,
            occupancies=occupancies_batch,
        )

        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_atom(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test with just one atom."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor([ELEMENT_TO_ATOMIC_NUM["C"]], device=device, dtype=torch.float32)
        b_factors = torch.tensor([20.0], device=device, dtype=torch.float32)
        occupancies = torch.tensor([1.0], device=device, dtype=torch.float32)
        coords_single = coords[:1]

        loss = reward_function_1vme(
            coordinates=coords_single.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert torch.isfinite(loss)

    def test_large_ensemble(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test with large ensemble sizes."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        batch_size = 20
        coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
        elements_batch = elements.unsqueeze(0).expand(batch_size, -1)
        b_factors_batch = b_factors.unsqueeze(0).expand(batch_size, -1)
        occupancies_batch = occupancies.unsqueeze(0).expand(batch_size, -1)

        loss = reward_function_1vme(
            coordinates=coords_batch,
            elements=elements_batch,
            b_factors=b_factors_batch,
            occupancies=occupancies_batch,
        )

        assert torch.isfinite(loss)

    def test_numerical_stability(self, reward_function_1vme, test_coordinates_1vme, device):
        """Test with extreme coordinate values."""
        coords, atom_array = test_coordinates_1vme

        elements = torch.tensor(
            [
                ELEMENT_TO_ATOMIC_NUM[e.upper() if len(e) == 1 else e[0].upper() + e[1:].lower()]
                for e in atom_array.element
            ],
            device=device,
            dtype=torch.float32,
        )
        b_factors = torch.from_numpy(atom_array.b_factor).to(device=device, dtype=torch.float32)
        occupancies = torch.from_numpy(atom_array.occupancy).to(device=device, dtype=torch.float32)

        coords_far = coords + torch.randn_like(coords) * 1e9

        loss = reward_function_1vme(
            coordinates=coords_far.unsqueeze(0),
            elements=elements.unsqueeze(0),
            b_factors=b_factors.unsqueeze(0),
            occupancies=occupancies.unsqueeze(0),
        )

        assert torch.isfinite(loss)

    def test_structure_to_reward_input(self, reward_function_1vme, structure_1vme_density):
        """Test structure_to_reward_input function."""
        inputs = reward_function_1vme.structure_to_reward_input(structure_1vme_density)

        assert "coordinates" in inputs
        assert "elements" in inputs
        assert "b_factors" in inputs
        assert "occupancies" in inputs

        # Check shapes - should be batched [B, N, ...]
        assert inputs["coordinates"].ndim == 3
        assert inputs["coordinates"].shape[0] == 1
        assert inputs["elements"].ndim == 2
        assert inputs["elements"].shape[0] == 1
        assert inputs["b_factors"].ndim == 2
        assert inputs["b_factors"].shape[0] == 1
        assert inputs["occupancies"].ndim == 2
        assert inputs["occupancies"].shape[0] == 1

        loss = reward_function_1vme(**inputs)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0
