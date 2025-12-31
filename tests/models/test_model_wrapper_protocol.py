"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement ModelWrapper and
DiffusionModelWrapper protocols. These tests automatically run for
all wrapper implementations listed in WRAPPER_FIXTURES.
"""

from collections.abc import Mapping

import numpy as np
import pytest
import torch


WRAPPER_FIXTURES = [
    "boltz1_wrapper",
    "boltz2_wrapper",
    "protenix_wrapper",
    "rf3_wrapper",
]


@pytest.mark.parametrize("wrapper_fixture", WRAPPER_FIXTURES)
class TestModelWrapperProtocol:
    """Test that wrappers implement ModelWrapper protocol correctly.

    The ModelWrapper protocol requires:
    - featurize(structure: dict, **kwargs) -> dict[str, Any]
    - step(features: dict, grad_needed: bool, **kwargs) -> dict[str, Any]
    """

    @pytest.mark.slow
    def test_featurize_return_type(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that featurize returns dict[str, Any]."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict), (
            f"{wrapper_fixture}.featurize must return dict, got {type(features)}"
        )
        assert len(features) > 0, f"{wrapper_fixture}.featurize returned empty dict"

    @pytest.mark.slow
    def test_step_runs(self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request):
        """Test that step executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = wrapper.step(features, grad_needed=False)
        assert output is not None, f"{wrapper_fixture}.step returned None"


# TODO: More thorough tests once we lock down the protocol details


@pytest.mark.parametrize("wrapper_fixture", WRAPPER_FIXTURES)
class TestDiffusionModelWrapperProtocol:
    """Test that wrappers implement DiffusionModelWrapper protocol correctly.

    The DiffusionModelWrapper protocol extends ModelWrapper and requires:
    - get_noise_schedule() -> Mapping[str, Float[ArrayLike | Tensor, "..."]]
    - get_timestep_scaling(timestep: float | int) -> dict[str, float]
    - denoise_step(features, noisy_coords, timestep, grad_needed, **kwargs)
        -> dict[str, Any]
    - initialize_from_noise(structure, noise_level, **kwargs)
        -> Float[Tensor, "*batch _num_atoms 3"]
    """

    def test_get_noise_schedule_runs(self, wrapper_fixture: str, request):
        """Test that get_noise_schedule executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()
        assert schedule is not None, f"{wrapper_fixture}.get_noise_schedule returned None"

    def test_get_noise_schedule_return_type(self, wrapper_fixture: str, request):
        """Test that get_noise_schedule returns Mapping with float arrays."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()
        assert isinstance(schedule, Mapping), (
            f"{wrapper_fixture}.get_noise_schedule must return Mapping, got{type(schedule)}"
        )
        assert len(schedule) > 0, f"{wrapper_fixture}.get_noise_schedule returned empty mapping"

        for key, value in schedule.items():
            assert isinstance(key, str), (
                f"{wrapper_fixture}.get_noise_schedule key must be str, got {type(key)}"
            )
            is_valid_array = (
                torch.is_tensor(value)
                or isinstance(value, np.ndarray)
                or (hasattr(value, "__array__") and hasattr(value, "shape"))
            )
            assert is_valid_array, (
                f"{wrapper_fixture}.get_noise_schedule['{key}'] must be array-like,"
                f"got {type(value)}"
            )

    def test_get_noise_schedule_required_keys(self, wrapper_fixture: str, request):
        """Test that get_noise_schedule returns required keys."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()

        required_keys = {"sigma_tm", "sigma_t", "gamma"}
        missing_keys = required_keys - set(schedule.keys())
        assert not missing_keys, (
            f"{wrapper_fixture}.get_noise_schedule missing required keys:{missing_keys}"
        )

    def test_get_timestep_scaling_runs(self, wrapper_fixture: str, request):
        """Test that get_timestep_scaling executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0)
        assert scaling is not None, f"{wrapper_fixture}.get_timestep_scaling returned None"

    @pytest.mark.slow
    def test_initialize_from_noise_runs(self, wrapper_fixture: str, structure_6b8x: dict, request):
        """Test that initialize_from_noise executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)
        assert noisy_coords is not None, f"{wrapper_fixture}.initialize_from_noise returned None"

    @pytest.mark.slow
    def test_denoise_step_runs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that denoise_step executes without exceptions."""
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
        assert output is not None, f"{wrapper_fixture}.denoise_step returned None"

    @pytest.mark.slow
    def test_denoise_step_return_type(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that denoise_step returns dict with atom_coords_denoised."""
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
        assert isinstance(output, dict), (
            f"{wrapper_fixture}.denoise_step must return dict, got {type(output)}"
        )
        assert "atom_coords_denoised" in output, (
            f"{wrapper_fixture}.denoise_step must return 'atom_coords_denoised' key"
        )


@pytest.mark.parametrize("wrapper_fixture", WRAPPER_FIXTURES)
class TestFeaturizeClearsCache:
    """Test that featurize() clears cached representations.

    This is critical for wrapper reuse - without clearing, stale
    representations from previous proteins could cause dimension
    mismatches or incorrect results.
    """

    def test_featurize_clears_cached_representations(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that calling featurize() clears any cached representations."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        # only test wrappers that actually have cached_representations
        if not hasattr(wrapper, "cached_representations"):
            pytest.skip(f"{wrapper_fixture} does not have cached_representations attribute")

        # dummy data to simulate previous run
        wrapper.cached_representations = {"cached": "representations"}

        wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)

        assert len(wrapper.cached_representations) == 0, (
            f"{wrapper_fixture}.featurize must clear cached_representations, "
            f"but cache still contains: {list(wrapper.cached_representations.keys())}"
        )
