"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement ModelWrapper and
DiffusionModelWrapper protocols.
"""

from collections.abc import Mapping

import pytest
import torch


@pytest.mark.parametrize("wrapper_fixture", ["boltz1_wrapper", "boltz2_wrapper"])
class TestModelWrapperProtocol:
    """Test that wrappers implement ModelWrapper protocol correctly."""

    def test_has_featurize_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "featurize")
        assert callable(getattr(wrapper, "featurize"))

    def test_has_step_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "step")
        assert callable(getattr(wrapper, "step"))

    @pytest.mark.slow
    def test_featurize_returns_dict(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0

    @pytest.mark.slow
    def test_step_returns_dict(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = wrapper.step(features, grad_needed=False)
        assert isinstance(output, dict)
        assert len(output) > 0


@pytest.mark.parametrize("wrapper_fixture", ["boltz1_wrapper", "boltz2_wrapper"])
class TestDiffusionModelWrapperProtocol:
    """Test that wrappers implement DiffusionModelWrapper protocol correctly."""

    def test_has_get_noise_schedule_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "get_noise_schedule")
        assert callable(getattr(wrapper, "get_noise_schedule"))

    def test_has_get_timestep_scaling_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "get_timestep_scaling")
        assert callable(getattr(wrapper, "get_timestep_scaling"))

    def test_has_denoise_step_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "denoise_step")
        assert callable(getattr(wrapper, "denoise_step"))

    def test_has_initialize_from_noise_method(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "initialize_from_noise")
        assert callable(getattr(wrapper, "initialize_from_noise"))

    def test_get_noise_schedule_returns_mapping(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()
        assert isinstance(schedule, Mapping)
        assert len(schedule) > 0

    def test_get_timestep_scaling_returns_dict(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0)
        assert isinstance(scaling, dict)
        assert len(scaling) > 0
        for value in scaling.values():
            assert isinstance(value, (int, float))

    @pytest.mark.slow
    def test_initialize_from_noise_returns_tensor(
        self, wrapper_fixture: str, structure_6b8x: dict, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)
        assert torch.is_tensor(noisy_coords)
        assert noisy_coords.shape[-1] == 3

    @pytest.mark.slow
    def test_denoise_step_returns_dict(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)
        output = wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert isinstance(output, dict)
        assert "atom_coords_denoised" in output


@pytest.mark.parametrize("wrapper_fixture", ["boltz1_wrapper", "boltz2_wrapper"])
class TestProtocolMethodSignatures:
    """Test that protocol methods have correct signatures and behavior."""

    @pytest.mark.slow
    def test_featurize_accepts_structure_and_kwargs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(
            structure_6b8x, out_dir=temp_output_dir, num_workers=2
        )
        assert isinstance(features, dict)

    @pytest.mark.slow
    def test_step_accepts_grad_needed_flag(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)

        output_no_grad = wrapper.step(features, grad_needed=False)
        assert isinstance(output_no_grad, dict)

        output_with_grad = wrapper.step(features, grad_needed=True)
        assert isinstance(output_with_grad, dict)

    def test_get_timestep_scaling_accepts_float(self, wrapper_fixture: str, request):
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0.0)
        assert isinstance(scaling, dict)

        scaling = wrapper.get_timestep_scaling(5.0)
        assert isinstance(scaling, dict)

    @pytest.mark.slow
    def test_denoise_step_accepts_kwargs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)

        output = wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            augmentation=False,
            align_to_input=False,
        )
        assert isinstance(output, dict)
