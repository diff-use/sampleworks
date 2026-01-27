"""Tests for Boltz model wrappers.

Tests specific to Boltz1Wrapper and Boltz2Wrapper implementations,
including helper functions and model-specific behavior.
"""

import dataclasses
from pathlib import Path
from typing import cast

import pytest
import torch
from sampleworks.models.boltz.wrapper import (
    annotate_structure_for_boltz,
    Boltz1Wrapper,
    Boltz2Wrapper,
    BoltzConditioning,
    BoltzConfig,
    create_boltz_input_from_structure,
)
from torch import Tensor


class TestCreateBoltzInputFromStructure:
    """Test the helper function that creates Boltz YAML from Atomworks structures."""

    def test_creates_yaml_file(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"
        assert (
            yaml_path.name == f"{structure_6b8x.get('metadata', {}).get('id', 'boltz_input')}.yaml"
        )

    def test_yaml_contains_sequences_key(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_protein_chain_in_yaml(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        content = yaml_path.read_text()

        for chain_id, chain_data in structure_6b8x["chain_info"].items():
            if chain_data["chain_type"].is_protein():
                assert f"id: {chain_id}" in content
                assert chain_data["processed_entity_canonical_sequence"] in content

    def test_handles_cif_format(self, structure_1vme: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_1vme, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_handles_pdb_format(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(
            structure_6b8x, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    @pytest.mark.parametrize("structure_fixture", ["structure_1vme", "structure_6b8x"])
    def test_works_with_test_structures(
        self, structure_fixture: str, temp_output_dir: Path, request
    ):
        structure = request.getfixturevalue(structure_fixture)
        yaml_path = create_boltz_input_from_structure(
            structure, temp_output_dir, msa_manager=None, msa_pairing_strategy="greedy"
        )
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert len(content) > 0
        assert "sequences:" in content


class TestAnnotateStructureForBoltz:
    """Test the annotate_structure_for_boltz helper function."""

    def test_annotate_adds_boltz_config(self, structure_6b8x: dict, temp_output_dir: Path):
        result = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        assert "_boltz_config" in result
        assert isinstance(result["_boltz_config"], BoltzConfig)

    def test_annotate_preserves_original_structure(
        self, structure_6b8x: dict, temp_output_dir: Path
    ):
        result = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        for key in structure_6b8x:
            assert key in result
            assert result[key] is structure_6b8x[key]

    def test_annotate_default_values(self, structure_6b8x: dict, temp_output_dir: Path):
        result = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        config = result["_boltz_config"]
        assert config.num_workers == 8
        assert config.ensemble_size == 1
        assert config.recycling_steps == 3

    def test_annotate_custom_values(self, structure_6b8x: dict, temp_output_dir: Path):
        result = annotate_structure_for_boltz(
            structure_6b8x,
            out_dir=temp_output_dir,
            num_workers=4,
            ensemble_size=2,
            recycling_steps=5,
        )
        config = result["_boltz_config"]
        assert config.num_workers == 4
        assert config.ensemble_size == 2
        assert config.recycling_steps == 5

    def test_annotate_out_dir_from_metadata(self, structure_6b8x: dict):
        result = annotate_structure_for_boltz(structure_6b8x)
        config = result["_boltz_config"]
        expected_id = structure_6b8x.get("metadata", {}).get("id", "boltz_output")
        assert config.out_dir == expected_id


class TestBoltzConfig:
    """Test the BoltzConfig dataclass."""

    def test_boltz_config_default_values(self):
        config = BoltzConfig()
        assert config.out_dir is None
        assert config.num_workers == 8
        assert config.ensemble_size == 1
        assert config.recycling_steps == 3

    def test_boltz_config_custom_values(self, temp_output_dir: Path):
        config = BoltzConfig(
            out_dir=temp_output_dir,
            num_workers=4,
            ensemble_size=5,
            recycling_steps=2,
        )
        assert config.out_dir == temp_output_dir
        assert config.num_workers == 4
        assert config.ensemble_size == 5
        assert config.recycling_steps == 2


class TestBoltzConditioning:
    """Test the BoltzConditioning dataclass."""

    def test_boltz_conditioning_is_frozen(self):
        assert dataclasses.is_dataclass(BoltzConditioning)
        fields = dataclasses.fields(BoltzConditioning)
        field_names = [f.name for f in fields]
        assert "s" in field_names
        assert "z" in field_names
        assert "s_inputs" in field_names
        assert "relative_position_encoding" in field_names
        assert "feats" in field_names
        assert "diffusion_conditioning" in field_names


@pytest.mark.slow
class TestBoltz1WrapperInitialization:
    """Test Boltz1Wrapper initialization and setup."""

    def test_wrapper_initializes(self, boltz1_wrapper: Boltz1Wrapper):
        assert boltz1_wrapper is not None
        assert hasattr(boltz1_wrapper, "model")
        assert hasattr(boltz1_wrapper, "device")

    def test_model_on_correct_device(self, boltz1_wrapper: Boltz1Wrapper, device):
        assert boltz1_wrapper.device == device
        assert next(boltz1_wrapper.model.parameters()).device == device

    def test_has_cached_representations_dict(self, boltz1_wrapper: Boltz1Wrapper):
        assert hasattr(boltz1_wrapper, "cached_representations")
        assert isinstance(boltz1_wrapper.cached_representations, dict)


@pytest.mark.slow
class TestBoltz2WrapperInitialization:
    """Test Boltz2Wrapper initialization and setup."""

    def test_wrapper_initializes(self, boltz2_wrapper: Boltz2Wrapper):
        assert boltz2_wrapper is not None
        assert hasattr(boltz2_wrapper, "model")
        assert hasattr(boltz2_wrapper, "device")

    def test_model_on_correct_device(self, boltz2_wrapper: Boltz2Wrapper, device):
        assert boltz2_wrapper.device == device
        assert next(boltz2_wrapper.model.parameters()).device == device

    def test_has_cached_representations_dict(self, boltz2_wrapper: Boltz2Wrapper):
        assert hasattr(boltz2_wrapper, "cached_representations")
        assert isinstance(boltz2_wrapper.cached_representations, dict)


@pytest.mark.slow
class TestBoltz1WrapperFeaturize:
    """Test Boltz1Wrapper featurize method."""

    def test_featurize_returns_generative_model_input(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None
        assert isinstance(features.conditioning, BoltzConditioning)
        assert "token_pad_mask" in features.conditioning.feats
        assert "atom_pad_mask" in features.conditioning.feats

    def test_featurize_with_cif_structure(
        self, boltz1_wrapper: Boltz1Wrapper, structure_1vme: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_1vme, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None

    def test_featurize_with_pdb_structure(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None

    def test_featurize_creates_data_module(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        boltz1_wrapper.featurize(structure)
        assert hasattr(boltz1_wrapper, "data_module")
        assert boltz1_wrapper.data_module is not None

    def test_featurize_clears_cached_representations(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        boltz1_wrapper.cached_representations = {"test_key": "test_value"}
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        boltz1_wrapper.featurize(structure)
        assert "test_key" not in boltz1_wrapper.cached_representations

    def test_featurize_with_ensemble_size(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        ensemble_size = 3
        structure = annotate_structure_for_boltz(
            structure_6b8x, out_dir=temp_output_dir, ensemble_size=ensemble_size
        )
        features = boltz1_wrapper.featurize(structure)
        assert features.x_init.shape[0] == ensemble_size


@pytest.mark.slow
class TestBoltz2WrapperFeaturize:
    """Test Boltz2Wrapper featurize method."""

    def test_featurize_returns_generative_model_input(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None
        assert isinstance(features.conditioning, BoltzConditioning)
        assert "token_pad_mask" in features.conditioning.feats
        assert "atom_pad_mask" in features.conditioning.feats

    def test_featurize_with_cif_structure(
        self, boltz2_wrapper: Boltz2Wrapper, structure_1vme: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_1vme, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None
        assert features.conditioning is not None
        assert "token_pad_mask" in features.conditioning.feats
        assert "atom_pad_mask" in features.conditioning.feats

    def test_featurize_with_pdb_structure(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert features.x_init is not None
        assert features.conditioning is not None
        assert "token_pad_mask" in features.conditioning.feats
        assert "atom_pad_mask" in features.conditioning.feats

    def test_featurize_creates_data_module(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        boltz2_wrapper.featurize(structure)
        assert hasattr(boltz2_wrapper, "data_module")
        assert boltz2_wrapper.data_module is not None

    def test_featurize_clears_cached_representations(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        boltz2_wrapper.cached_representations = {"test_key": "test_value"}
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        boltz2_wrapper.featurize(structure)
        assert "test_key" not in boltz2_wrapper.cached_representations

    def test_featurize_with_ensemble_size(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        ensemble_size = 3
        structure = annotate_structure_for_boltz(
            structure_6b8x, out_dir=temp_output_dir, ensemble_size=ensemble_size
        )
        features = boltz2_wrapper.featurize(structure)
        assert features.x_init.shape[0] == ensemble_size


@pytest.mark.slow
class TestBoltz1WrapperStep:
    """Test Boltz1Wrapper step method."""

    def test_step_returns_tensor(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz1_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)

    def test_step_output_shape_matches_input(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz1_wrapper.step(x_init, t, features=features)

        assert result.shape == x_init.shape
        assert result.shape[-1] == 3

    def test_step_with_high_t(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([160.0])
        result = boltz1_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == x_init.shape

    def test_step_with_low_t(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([0.01])
        result = boltz1_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == x_init.shape

    def test_step_with_float_t(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        result = boltz1_wrapper.step(x_init, 1.0, features=features)

        assert torch.is_tensor(result)

    def test_step_requires_features(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="features"):
            boltz1_wrapper.step(x_init, t, features=None)

    def test_step_denoises_input(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz1_wrapper.step(x_init, t, features=features)

        assert not torch.allclose(result, x_init)


@pytest.mark.slow
class TestBoltz2WrapperStep:
    """Test Boltz2Wrapper step method."""

    def test_step_returns_tensor(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz2_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)

    def test_step_output_shape_matches_input(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz2_wrapper.step(x_init, t, features=features)

        assert result.shape == x_init.shape
        assert result.shape[-1] == 3

    def test_step_with_high_t(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([160.0])
        result = boltz2_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == x_init.shape

    def test_step_with_low_t(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([0.01])
        result = boltz2_wrapper.step(x_init, t, features=features)

        assert torch.is_tensor(result)
        assert result.shape == x_init.shape

    def test_step_with_float_t(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        result = boltz2_wrapper.step(x_init, 1.0, features=features)

        assert torch.is_tensor(result)

    def test_step_requires_features(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        with pytest.raises(ValueError, match="features"):
            boltz2_wrapper.step(x_init, t, features=None)

    def test_step_denoises_input(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        x_init = cast(Tensor, features.x_init)

        t = torch.tensor([1.0])
        result = boltz2_wrapper.step(x_init, t, features=features)

        assert not torch.allclose(result, x_init)


@pytest.mark.slow
class TestBoltz1WrapperInitializeFromPrior:
    """Test Boltz1Wrapper initialize_from_prior method."""

    def test_initialize_from_prior_returns_tensor(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)

        result = boltz1_wrapper.initialize_from_prior(batch_size=2, features=features)

        assert torch.is_tensor(result)

    def test_initialize_from_prior_correct_shape_from_features(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)

        batch_size = 3
        result = boltz1_wrapper.initialize_from_prior(batch_size=batch_size, features=features)

        assert result.shape[0] == batch_size
        assert result.shape[-1] == 3

    def test_initialize_from_prior_correct_shape_from_explicit(self, boltz1_wrapper: Boltz1Wrapper):
        batch_size = 2
        num_atoms = 100
        result = boltz1_wrapper.initialize_from_prior(batch_size=batch_size, shape=(num_atoms, 3))

        assert result.shape == (batch_size, num_atoms, 3)

    def test_initialize_from_prior_batch_dimension(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)

        for batch_size in [1, 2, 5]:
            result = boltz1_wrapper.initialize_from_prior(batch_size=batch_size, features=features)
            assert result.shape[0] == batch_size

    def test_initialize_from_prior_shape_validation(self, boltz1_wrapper: Boltz1Wrapper):
        with pytest.raises(ValueError, match="shape"):
            boltz1_wrapper.initialize_from_prior(batch_size=2, shape=(100,))

        with pytest.raises(ValueError, match="shape"):
            boltz1_wrapper.initialize_from_prior(batch_size=2, shape=(100, 4))

    def test_initialize_from_prior_requires_features_or_shape(self, boltz1_wrapper: Boltz1Wrapper):
        with pytest.raises(ValueError, match="features|shape"):
            boltz1_wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.slow
class TestBoltz2WrapperInitializeFromPrior:
    """Test Boltz2Wrapper initialize_from_prior method."""

    def test_initialize_from_prior_returns_tensor(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)

        result = boltz2_wrapper.initialize_from_prior(batch_size=2, features=features)

        assert torch.is_tensor(result)

    def test_initialize_from_prior_correct_shape_from_features(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)

        batch_size = 3
        result = boltz2_wrapper.initialize_from_prior(batch_size=batch_size, features=features)

        assert result.shape[0] == batch_size
        assert result.shape[-1] == 3

    def test_initialize_from_prior_correct_shape_from_explicit(self, boltz2_wrapper: Boltz2Wrapper):
        batch_size = 2
        num_atoms = 100
        result = boltz2_wrapper.initialize_from_prior(batch_size=batch_size, shape=(num_atoms, 3))

        assert result.shape == (batch_size, num_atoms, 3)

    def test_initialize_from_prior_batch_dimension(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)

        for batch_size in [1, 2, 5]:
            result = boltz2_wrapper.initialize_from_prior(batch_size=batch_size, features=features)
            assert result.shape[0] == batch_size

    def test_initialize_from_prior_shape_validation(self, boltz2_wrapper: Boltz2Wrapper):
        with pytest.raises(ValueError, match="shape"):
            boltz2_wrapper.initialize_from_prior(batch_size=2, shape=(100,))

        with pytest.raises(ValueError, match="shape"):
            boltz2_wrapper.initialize_from_prior(batch_size=2, shape=(100, 4))

    def test_initialize_from_prior_requires_features_or_shape(self, boltz2_wrapper: Boltz2Wrapper):
        with pytest.raises(ValueError, match="features|shape"):
            boltz2_wrapper.initialize_from_prior(batch_size=2)


@pytest.mark.slow
class TestBoltz1WrapperPairformerPass:
    """Test Boltz1Wrapper _pairformer_pass method (internal Pairformer computation)."""

    def test_pairformer_pass_returns_outputs(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        assert features.conditioning is not None
        output = boltz1_wrapper._pairformer_pass(features.conditioning.feats)
        assert isinstance(output, dict)
        assert "s" in output
        assert "z" in output
        assert "feats" in output
        assert "s_inputs" in output
        assert "relative_position_encoding" in output

    def test_pairformer_pass_respects_recycling_steps(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz1_wrapper.featurize(structure)
        assert features.conditioning is not None
        output = boltz1_wrapper._pairformer_pass(features.conditioning.feats, recycling_steps=0)
        assert isinstance(output, dict)


@pytest.mark.slow
class TestBoltz2WrapperPairformerPass:
    """Test Boltz2Wrapper _pairformer_pass method (internal Pairformer computation)."""

    def test_pairformer_pass_returns_outputs(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        assert features.conditioning is not None
        output = boltz2_wrapper._pairformer_pass(features.conditioning.feats)
        assert isinstance(output, dict)
        assert "s" in output
        assert "z" in output
        assert "diffusion_conditioning" in output
        assert "feats" in output

        dc = output["diffusion_conditioning"]
        assert "q" in dc
        assert "c" in dc
        assert "to_keys" in dc
        assert "atom_enc_bias" in dc
        assert "atom_dec_bias" in dc
        assert "token_trans_bias" in dc

    def test_pairformer_pass_respects_recycling_steps(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        structure = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = boltz2_wrapper.featurize(structure)
        assert features.conditioning is not None
        output = boltz2_wrapper._pairformer_pass(features.conditioning.feats, recycling_steps=0)
        assert isinstance(output, dict)


@pytest.mark.slow
class TestBoltzWrappersEndToEnd:
    """End-to-end integration tests for Boltz wrappers."""

    @pytest.mark.parametrize(
        "wrapper_fixture,structure_fixture",
        [
            ("boltz1_wrapper", "structure_6b8x"),
            ("boltz1_wrapper", "structure_1vme"),
            ("boltz2_wrapper", "structure_6b8x"),
            ("boltz2_wrapper", "structure_1vme"),
        ],
        ids=["boltz1-pdb", "boltz1-cif", "boltz2-pdb", "boltz2-cif"],
    )
    def test_full_pipeline(
        self,
        wrapper_fixture: str,
        structure_fixture: str,
        temp_output_dir: Path,
        request,
    ):
        from sampleworks.models.protocol import GenerativeModelInput

        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = request.getfixturevalue(structure_fixture)

        annotated = annotate_structure_for_boltz(structure, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)
        assert isinstance(features, GenerativeModelInput)
        assert isinstance(features.conditioning, BoltzConditioning)

        pairformer_output = wrapper._pairformer_pass(features.conditioning.feats)
        assert isinstance(pairformer_output, dict)

        x_init = wrapper.initialize_from_prior(batch_size=2, features=features)
        assert torch.is_tensor(x_init)
        assert x_init.shape[0] == 2

        t = torch.tensor([1.0])
        output = wrapper.step(x_init, t, features=features)
        assert torch.is_tensor(output)
        assert output.shape == x_init.shape

    @pytest.mark.parametrize("wrapper_fixture", ["boltz1_wrapper", "boltz2_wrapper"])
    def test_multiple_step_calls(
        self,
        wrapper_fixture: str,
        structure_6b8x: dict,
        temp_output_dir: Path,
        request,
    ):
        """Test that multiple step() calls work correctly (simulating sampling loop)."""
        wrapper = request.getfixturevalue(wrapper_fixture)

        annotated = annotate_structure_for_boltz(structure_6b8x, out_dir=temp_output_dir)
        features = wrapper.featurize(annotated)

        x_t = wrapper.initialize_from_prior(batch_size=1, features=features)

        for t_val in [160.0, 80.0, 40.0, 20.0, 10.0]:
            t = torch.tensor([t_val])
            x_t = wrapper.step(x_t, t, features=features)
            assert torch.is_tensor(x_t)
            assert x_t.shape[-1] == 3
