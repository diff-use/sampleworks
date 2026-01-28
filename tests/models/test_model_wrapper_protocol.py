"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement FlowModelWrapper protocol.
"""

import pytest
import torch

from tests.conftest import (
    annotate_structure_for_wrapper,
    get_conditioning_type,
    get_slow_wrappers,
    STRUCTURES,
    WrapperInfo,
)


def wrapper_structure_id(val):
    """Generate test ID for wrapper or structure."""
    if isinstance(val, WrapperInfo):
        return val.name
    return val.replace("structure_", "")


@pytest.mark.parametrize("wrapper_info", get_slow_wrappers(), ids=lambda w: w.name)
class TestFlowModelWrapperProtocol:
    """Test that wrappers implement FlowModelWrapper protocol correctly.

    The FlowModelWrapper protocol requires:
    - featurize(structure: dict) -> GenerativeModelInput[C]
    - step(x_t, t, *, features) -> FlowOrEnergyBasedModelOutputT
    - initialize_from_prior(batch_size, features, *, shape) -> FlowOrEnergyBasedModelOutputT
    """

    def test_isinstance_flow_model_wrapper(self, wrapper_info: WrapperInfo, request):
        """Test wrapper implements FlowModelWrapper protocol."""
        from sampleworks.models.protocol import FlowModelWrapper

        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        assert isinstance(wrapper, FlowModelWrapper), (
            f"{wrapper_info.name} does not implement FlowModelWrapper protocol"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_featurize_returns_generative_model_input(
        self, wrapper_info: WrapperInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test featurize returns GenerativeModelInput with x_init and conditioning."""
        from sampleworks.models.protocol import GenerativeModelInput

        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        structure = request.getfixturevalue(structure_fixture)
        conditioning_type = get_conditioning_type(wrapper_info)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        assert isinstance(features, GenerativeModelInput), (
            f"{wrapper_info.name}.featurize must return GenerativeModelInput, got {type(features)}"
        )
        assert features.x_init is not None, (
            f"{wrapper_info.name}.featurize returned None for x_init"
        )
        assert features.conditioning is not None, (
            f"{wrapper_info.name}.featurize returned None for conditioning"
        )
        assert isinstance(features.conditioning, conditioning_type), (
            f"{wrapper_info.name}.featurize conditioning must be {conditioning_type.__name__}, "
            f"got {type(features.conditioning)}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_featurize_x_init_shape(
        self, wrapper_info: WrapperInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test featurize x_init has correct shape (batch, atoms, 3)."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        ensemble_size = 2
        annotated = annotate_structure_for_wrapper(
            wrapper_info, structure, temp_output_dir, ensemble_size=ensemble_size
        )
        features = wrapper.featurize(annotated)

        assert features.x_init.ndim == 3, (
            f"{wrapper_info.name}.featurize x_init should be 3D, got {features.x_init.ndim}D"
        )
        assert features.x_init.shape[0] == ensemble_size, (
            f"{wrapper_info.name}.featurize x_init batch should be {ensemble_size}, "
            f"got {features.x_init.shape[0]}"
        )
        assert features.x_init.shape[2] == 3, (
            f"{wrapper_info.name}.featurize x_init last dim should be 3, "
            f"got {features.x_init.shape[2]}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_step_returns_tensor(
        self, wrapper_info: WrapperInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test step(x_t, t, features) returns coordinates tensor."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        t = torch.tensor([1.0])
        result = wrapper.step(features.x_init, t, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.step must return Tensor, got {type(result)}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_info.name}.step result last dim should be 3, got {result.shape[-1]}"
        )
        assert result.shape == features.x_init.shape, (
            f"{wrapper_info.name}.step output shape {result.shape} != input shape "
            f"{features.x_init.shape}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_step_with_float_t(
        self, wrapper_info: WrapperInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test step works with float t value."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        t = 1.0
        result = wrapper.step(features.x_init, t, features=features)

        assert torch.is_tensor(result), f"{wrapper_info.name}.step must return Tensor with float t"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "structure_fixture", STRUCTURES, ids=lambda s: s.replace("structure_", "")
    )
    def test_initialize_from_prior_with_features(
        self, wrapper_info: WrapperInfo, structure_fixture: str, temp_output_dir, request
    ):
        """Test initialize_from_prior with features from featurize."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_wrapper(wrapper_info, structure, temp_output_dir)
        features = wrapper.featurize(annotated)

        batch_size = 3
        result = wrapper.initialize_from_prior(batch_size, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.initialize_from_prior must return Tensor"
        )
        assert result.shape[0] == batch_size, (
            f"{wrapper_info.name}.initialize_from_prior batch should be {batch_size}, "
            f"got {result.shape[0]}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_info.name}.initialize_from_prior last dim should be 3, got "
            f"{result.shape[-1]}"
        )

    @pytest.mark.slow
    def test_initialize_from_prior_with_shape(self, wrapper_info: WrapperInfo, request):
        """Test initialize_from_prior with explicit shape."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)

        batch_size = 2
        num_atoms = 100
        result = wrapper.initialize_from_prior(batch_size, shape=(num_atoms, 3))

        assert torch.is_tensor(result), (
            f"{wrapper_info.name}.initialize_from_prior must return Tensor"
        )
        assert result.shape == (batch_size, num_atoms, 3), (
            f"{wrapper_info.name}.initialize_from_prior shape should be "
            f"({batch_size}, {num_atoms}, 3), got {result.shape}"
        )

    def test_initialize_from_prior_raises_without_features_or_shape(
        self, wrapper_info: WrapperInfo, request
    ):
        """Test initialize_from_prior raises ValueError without features or shape."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)

        with pytest.raises(ValueError, match="features|shape"):
            wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.parametrize("wrapper_info", get_slow_wrappers(), ids=lambda w: w.name)
class TestStepRequiresFeatures:
    """Test that step() requires features parameter."""

    def test_step_raises_without_features(self, wrapper_info: WrapperInfo, request):
        """Test step raises ValueError when features is None."""
        wrapper = request.getfixturevalue(wrapper_info.fixture_name)

        x_t = torch.randn(1, 100, 3, device=wrapper.device)
        t = torch.tensor([1.0])

        with pytest.raises(ValueError, match="features"):
            wrapper.step(x_t, t, features=None)
