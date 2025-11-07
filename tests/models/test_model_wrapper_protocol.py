"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement ModelWrapper and
DiffusionModelWrapper protocols. These tests automatically run for
all wrapper implementations listed in WRAPPER_FIXTURES.
"""

from collections.abc import Mapping

import numpy as np
import pytest
import torch


WRAPPER_FIXTURES = ["boltz1_wrapper", "boltz2_wrapper", "protenix_wrapper"]


@pytest.mark.parametrize("wrapper_fixture", WRAPPER_FIXTURES)
class TestModelWrapperProtocol:
    """Test that wrappers implement ModelWrapper protocol correctly.

    The ModelWrapper protocol requires:
    - featurize(structure: dict, **kwargs) -> dict[str, Any]
    - step(features: dict, grad_needed: bool, **kwargs) -> dict[str, Any]
    """

    def test_has_featurize_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable featurize method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(
            wrapper, "featurize"
        ), f"{wrapper_fixture} missing featurize method"
        assert callable(
            getattr(wrapper, "featurize")
        ), f"{wrapper_fixture}.featurize is not callable"

    def test_has_step_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable step method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(wrapper, "step"), f"{wrapper_fixture} missing step method"
        assert callable(
            getattr(wrapper, "step")
        ), f"{wrapper_fixture}.step is not callable"

    @pytest.mark.slow
    def test_featurize_runs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that featurize executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert features is not None, f"{wrapper_fixture}.featurize returned None"

    @pytest.mark.slow
    def test_featurize_return_type(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that featurize returns dict[str, Any]."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(
            features, dict
        ), f"{wrapper_fixture}.featurize must return dict, got {type(features)}"
        assert len(features) > 0, f"{wrapper_fixture}.featurize returned empty dict"

    @pytest.mark.slow
    def test_featurize_accepts_kwargs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that featurize accepts optional keyword arguments."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(
            structure_6b8x, out_dir=temp_output_dir, num_workers=2
        )
        assert isinstance(
            features, dict
        ), f"{wrapper_fixture}.featurize failed with kwargs"

    @pytest.mark.slow
    def test_step_runs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that step executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = wrapper.step(features, grad_needed=False)
        assert output is not None, f"{wrapper_fixture}.step returned None"

    @pytest.mark.slow
    def test_step_return_type(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that step returns dict[str, Any]."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = wrapper.step(features, grad_needed=False)
        assert isinstance(
            output, dict
        ), f"{wrapper_fixture}.step must return dict, got {type(output)}"
        assert len(output) > 0, f"{wrapper_fixture}.step returned empty dict"

    @pytest.mark.slow
    def test_step_accepts_grad_needed(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that step accepts grad_needed parameter."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)

        output_no_grad = wrapper.step(features, grad_needed=False)
        assert isinstance(
            output_no_grad, dict
        ), f"{wrapper_fixture}.step failed with grad_needed=False"

        output_with_grad = wrapper.step(features, grad_needed=True)
        assert isinstance(
            output_with_grad, dict
        ), f"{wrapper_fixture}.step failed with grad_needed=True"


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

    def test_has_get_noise_schedule_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable get_noise_schedule method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(
            wrapper, "get_noise_schedule"
        ), f"{wrapper_fixture} missing get_noise_schedule method"
        assert callable(
            getattr(wrapper, "get_noise_schedule")
        ), f"{wrapper_fixture}.get_noise_schedule is not callable"

    def test_has_get_timestep_scaling_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable get_timestep_scaling method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(
            wrapper, "get_timestep_scaling"
        ), f"{wrapper_fixture} missing get_timestep_scaling method"
        assert callable(
            getattr(wrapper, "get_timestep_scaling")
        ), f"{wrapper_fixture}.get_timestep_scaling is not callable"

    def test_has_denoise_step_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable denoise_step method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(
            wrapper, "denoise_step"
        ), f"{wrapper_fixture} missing denoise_step method"
        assert callable(
            getattr(wrapper, "denoise_step")
        ), f"{wrapper_fixture}.denoise_step is not callable"

    def test_has_initialize_from_noise_method(self, wrapper_fixture: str, request):
        """Test that wrapper has a callable initialize_from_noise method."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        assert hasattr(
            wrapper, "initialize_from_noise"
        ), f"{wrapper_fixture} missing initialize_from_noise method"
        assert callable(
            getattr(wrapper, "initialize_from_noise")
        ), f"{wrapper_fixture}.initialize_from_noise is not callable"

    def test_get_noise_schedule_runs(self, wrapper_fixture: str, request):
        """Test that get_noise_schedule executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()
        assert (
            schedule is not None
        ), f"{wrapper_fixture}.get_noise_schedule returned None"

    def test_get_noise_schedule_return_type(self, wrapper_fixture: str, request):
        """Test that get_noise_schedule returns Mapping with float arrays."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        schedule = wrapper.get_noise_schedule()
        assert isinstance(schedule, Mapping), (
            f"{wrapper_fixture}.get_noise_schedule must return Mapping, got"
            f"{type(schedule)}"
        )
        assert (
            len(schedule) > 0
        ), f"{wrapper_fixture}.get_noise_schedule returned empty mapping"

        for key, value in schedule.items():
            assert isinstance(
                key, str
            ), f"{wrapper_fixture}.get_noise_schedule key must be str, got {type(key)}"
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
            f"{wrapper_fixture}.get_noise_schedule missing required keys:"
            f"{missing_keys}"
        )

    def test_get_timestep_scaling_runs(self, wrapper_fixture: str, request):
        """Test that get_timestep_scaling executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0)
        assert (
            scaling is not None
        ), f"{wrapper_fixture}.get_timestep_scaling returned None"

    def test_get_timestep_scaling_return_type(self, wrapper_fixture: str, request):
        """Test that get_timestep_scaling returns dict[str, float]."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0)
        assert isinstance(scaling, dict), (
            f"{wrapper_fixture}.get_timestep_scaling must return dict, got"
            f"{type(scaling)}"
        )
        assert (
            len(scaling) > 0
        ), f"{wrapper_fixture}.get_timestep_scaling returned empty dict"

        for key, value in scaling.items():
            assert isinstance(key, str), (
                f"{wrapper_fixture}.get_timestep_scaling key must be str, got"
                f"{type(key)}"
            )
            assert isinstance(value, int | float), (
                f"{wrapper_fixture}.get_timestep_scaling['{key}'] must be numeric, got"
                f"{type(value)}"
            )

    def test_get_timestep_scaling_has_required_keys(
        self, wrapper_fixture: str, request
    ):
        """Test that get_timestep_scaling returns sigma_t at minimum."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        scaling = wrapper.get_timestep_scaling(0)

        assert (
            "sigma_t" in scaling
        ), f"{wrapper_fixture}.get_timestep_scaling must include 'sigma_t' key"

    def test_get_timestep_scaling_accepts_float(self, wrapper_fixture: str, request):
        """Test that get_timestep_scaling accepts float timestep values."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        scaling_int = wrapper.get_timestep_scaling(0)
        assert isinstance(
            scaling_int, dict
        ), f"{wrapper_fixture}.get_timestep_scaling failed with int timestep"

        scaling_float = wrapper.get_timestep_scaling(5.0)
        assert isinstance(
            scaling_float, dict
        ), f"{wrapper_fixture}.get_timestep_scaling failed with float timestep"

    @pytest.mark.slow
    def test_initialize_from_noise_runs(
        self, wrapper_fixture: str, structure_6b8x: dict, request
    ):
        """Test that initialize_from_noise executes without exceptions."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)
        assert (
            noisy_coords is not None
        ), f"{wrapper_fixture}.initialize_from_noise returned None"

    @pytest.mark.slow
    def test_initialize_from_noise_return_type(
        self, wrapper_fixture: str, structure_6b8x: dict, request
    ):
        """Test that initialize_from_noise returns tensor or array."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)

        is_valid_array = (
            torch.is_tensor(noisy_coords)
            or isinstance(noisy_coords, np.ndarray)
            or (hasattr(noisy_coords, "__array__") and hasattr(noisy_coords, "shape"))
        )
        assert is_valid_array, (
            f"{wrapper_fixture}.initialize_from_noise must return array-like, got"
            f"{type(noisy_coords)}"
        )

    @pytest.mark.slow
    def test_initialize_from_noise_coords_shape(
        self, wrapper_fixture: str, structure_6b8x: dict, request
    ):
        """Test that initialize_from_noise returns coordinates with shape
        [..., N, 3]."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)

        assert hasattr(
            noisy_coords, "shape"
        ), f"{wrapper_fixture}.initialize_from_noise output must have shape attribute"
        assert noisy_coords.shape[-1] == 3, (
            f"{wrapper_fixture}.initialize_from_noise must return coords with last "
            f"dim=3, got {noisy_coords.shape}"
        )

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
        assert isinstance(
            output, dict
        ), f"{wrapper_fixture}.denoise_step must return dict, got {type(output)}"
        assert (
            "atom_coords_denoised" in output
        ), f"{wrapper_fixture}.denoise_step must return 'atom_coords_denoised' key"

    @pytest.mark.slow
    def test_denoise_step_coords_shape(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that denoise_step returns coordinates with shape [..., N, 3]."""
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

        coords = output["atom_coords_denoised"]
        is_valid_array = (
            torch.is_tensor(coords)
            or isinstance(coords, np.ndarray)
            or (hasattr(coords, "__array__") and hasattr(coords, "shape"))
        )
        assert is_valid_array, (
            f"{wrapper_fixture}.denoise_step['atom_coords_denoised'] must be "
            f"array-like, got {type(coords)}"
        )
        assert hasattr(coords, "shape"), (
            f"{wrapper_fixture}.denoise_step['atom_coords_denoised'] must have shape"
            " attribute"
        )
        assert coords.shape[-1] == 3, (
            f"{wrapper_fixture}.denoise_step coords must have last dim=3, "
            f"got {coords.shape}"
        )

    @pytest.mark.slow
    def test_denoise_step_accepts_grad_needed(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that denoise_step accepts grad_needed parameter."""
        wrapper = request.getfixturevalue(wrapper_fixture)
        features = wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = wrapper.initialize_from_noise(structure_6b8x, noise_level=0)

        output_no_grad = wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert isinstance(
            output_no_grad, dict
        ), f"{wrapper_fixture}.denoise_step failed with grad_needed=False"

        output_with_grad = wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=True,
            align_to_input=False,
        )
        assert isinstance(
            output_with_grad, dict
        ), f"{wrapper_fixture}.denoise_step failed with grad_needed=True"

    @pytest.mark.slow
    def test_denoise_step_accepts_kwargs(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test that denoise_step accepts optional keyword arguments."""
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
        assert isinstance(
            output, dict
        ), f"{wrapper_fixture}.denoise_step failed with kwargs"
