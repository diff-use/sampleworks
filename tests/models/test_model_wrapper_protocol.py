"""Tests for protocol compliance of model wrappers.

Tests that implementations correctly implement FlowModelWrapper protocol.
"""

import pytest
import torch


BOLTZ_WRAPPER_FIXTURES = [
    "boltz1_wrapper",
    "boltz2_wrapper",
]


@pytest.mark.parametrize("wrapper_fixture", BOLTZ_WRAPPER_FIXTURES)
class TestFlowModelWrapperProtocol:
    """Test that Boltz wrappers implement FlowModelWrapper protocol correctly.

    The FlowModelWrapper protocol requires:
    - featurize(structure: dict) -> GenerativeModelInput[C]
    - step(x_t, t, *, features) -> FlowOrEnergyBasedModelOutputT
    - initialize_from_prior(t, *, structure) -> FlowOrEnergyBasedModelOutputT
    """

    def test_isinstance_flow_model_wrapper(self, wrapper_fixture: str, request):
        """Test wrapper implements FlowModelWrapper protocol."""
        from sampleworks.models.protocol import FlowModelWrapper

        wrapper = request.getfixturevalue(wrapper_fixture)
        assert isinstance(wrapper, FlowModelWrapper), (
            f"{wrapper_fixture} does not implement FlowModelWrapper protocol"
        )

    @pytest.mark.slow
    def test_featurize_returns_generative_model_input(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test featurize returns GenerativeModelInput with x_init and conditioning."""
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz, BoltzConditioning
        from sampleworks.models.protocol import GenerativeModelInput

        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = wrapper.featurize(structure)

        assert isinstance(features, GenerativeModelInput), (
            f"{wrapper_fixture}.featurize must return GenerativeModelInput, got {type(features)}"
        )
        assert features.x_init is not None, f"{wrapper_fixture}.featurize returned None for x_init"
        assert features.conditioning is not None, (
            f"{wrapper_fixture}.featurize returned None for conditioning"
        )
        assert isinstance(features.conditioning, BoltzConditioning), (
            f"{wrapper_fixture}.featurize conditioning must be BoltzConditioning, "
            f"got {type(features.conditioning)}"
        )

    @pytest.mark.slow
    def test_featurize_x_init_shape(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test featurize x_init has correct shape (batch, atoms, 3)."""
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz

        wrapper = request.getfixturevalue(wrapper_fixture)
        ensemble_size = 2
        structure = annotate_structure_for_boltz(
            structure_6b8x, out_dir=temp_output_dir, ensemble_size=ensemble_size
        )
        features = wrapper.featurize(structure)

        assert features.x_init.ndim == 3, (
            f"{wrapper_fixture}.featurize x_init should be 3D, got {features.x_init.ndim}D"
        )
        assert features.x_init.shape[0] == ensemble_size, (
            f"{wrapper_fixture}.featurize x_init batch should be {ensemble_size}, "
            f"got {features.x_init.shape[0]}"
        )
        assert features.x_init.shape[2] == 3, (
            f"{wrapper_fixture}.featurize x_init last dim should be 3, "
            f"got {features.x_init.shape[2]}"
        )

    @pytest.mark.slow
    def test_step_returns_tensor(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test step(x_t, t, features) returns coordinates tensor."""
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz

        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = wrapper.featurize(structure)

        t = torch.tensor([1.0])
        result = wrapper.step(features.x_init, t, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_fixture}.step must return Tensor, got {type(result)}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_fixture}.step result last dim should be 3, got {result.shape[-1]}"
        )
        assert result.shape == features.x_init.shape, (
            f"{wrapper_fixture}.step output shape {result.shape} != input shape "
            f"{features.x_init.shape}"
        )

    @pytest.mark.slow
    def test_step_with_float_t(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test step works with float t value."""
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz

        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = wrapper.featurize(structure)

        t = 1.0
        result = wrapper.step(features.x_init, t, features=features)

        assert torch.is_tensor(result), f"{wrapper_fixture}.step must return Tensor with float t"

    @pytest.mark.slow
    def test_initialize_from_prior_with_features(
        self, wrapper_fixture: str, structure_6b8x: dict, temp_output_dir, request
    ):
        """Test initialize_from_prior with features from featurize."""
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz

        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = wrapper.featurize(structure)

        batch_size = 3
        result = wrapper.initialize_from_prior(batch_size, features=features)

        assert torch.is_tensor(result), (
            f"{wrapper_fixture}.initialize_from_prior must return Tensor"
        )
        assert result.shape[0] == batch_size, (
            f"{wrapper_fixture}.initialize_from_prior batch should be {batch_size}, "
            f"got {result.shape[0]}"
        )
        assert result.shape[-1] == 3, (
            f"{wrapper_fixture}.initialize_from_prior last dim should be 3, got {result.shape[-1]}"
        )

    @pytest.mark.slow
    def test_initialize_from_prior_with_shape(self, wrapper_fixture: str, request):
        """Test initialize_from_prior with explicit shape."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        batch_size = 2
        num_atoms = 100
        result = wrapper.initialize_from_prior(batch_size, shape=(num_atoms, 3))

        assert torch.is_tensor(result), (
            f"{wrapper_fixture}.initialize_from_prior must return Tensor"
        )
        assert result.shape == (batch_size, num_atoms, 3), (
            f"{wrapper_fixture}.initialize_from_prior shape should be "
            f"({batch_size}, {num_atoms}, 3), got {result.shape}"
        )

    def test_initialize_from_prior_raises_without_features_or_shape(
        self, wrapper_fixture: str, request
    ):
        """Test initialize_from_prior raises ValueError without features or shape."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        with pytest.raises(ValueError, match="features|shape"):
            wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.parametrize("wrapper_fixture", BOLTZ_WRAPPER_FIXTURES)
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
        from sampleworks.models.boltz.wrapper import annotate_structure_for_boltz

        wrapper = request.getfixturevalue(wrapper_fixture)

        if not hasattr(wrapper, "cached_representations"):
            pytest.skip(f"{wrapper_fixture} does not have cached_representations attribute")

        wrapper.cached_representations = {"cached": "representations"}
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        wrapper.featurize(structure)

        # After featurize, cache should be populated with fresh representations
        # (not the dummy data we set)
        assert "cached" not in wrapper.cached_representations, (
            f"{wrapper_fixture}.featurize must clear old cached_representations"
        )


@pytest.mark.parametrize("wrapper_fixture", BOLTZ_WRAPPER_FIXTURES)
class TestStepRequiresFeatures:
    """Test that step() requires features parameter."""

    @pytest.mark.slow
    def test_step_raises_without_features(self, wrapper_fixture: str, request):
        """Test step raises ValueError when features is None."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        x_t = torch.randn(1, 100, 3, device=wrapper.device)
        t = torch.tensor([1.0])

        with pytest.raises(ValueError, match="features"):
            wrapper.step(x_t, t, features=None)
