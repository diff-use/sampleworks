"""Tests for Boltz model wrappers.

Tests specific to Boltz1Wrapper and Boltz2Wrapper implementations,
including helper functions and model-specific behavior.
"""

from pathlib import Path

import pytest
import torch
from sampleworks.models.boltz.wrapper import (
    Boltz1Wrapper,
    Boltz2Wrapper,
    create_boltz_input_from_structure,
)


class TestCreateBoltzInputFromStructure:
    """Test the helper function that creates Boltz YAML from Atomworks structures."""

    def test_creates_yaml_file(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(structure_6b8x, temp_output_dir)
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"
        assert yaml_path.name == "boltz_input.yaml"

    def test_yaml_contains_sequences_key(
        self, structure_6b8x: dict, temp_output_dir: Path
    ):
        yaml_path = create_boltz_input_from_structure(structure_6b8x, temp_output_dir)
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_protein_chain_in_yaml(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(structure_6b8x, temp_output_dir)
        content = yaml_path.read_text()

        for chain_id, chain_data in structure_6b8x["chain_info"].items():
            if chain_data["chain_type"].is_protein():
                assert f"id: {chain_id}" in content
                assert chain_data["processed_entity_canonical_sequence"] in content

    def test_handles_cif_format(self, structure_1vme: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(structure_1vme, temp_output_dir)
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    def test_handles_pdb_format(self, structure_6b8x: dict, temp_output_dir: Path):
        yaml_path = create_boltz_input_from_structure(structure_6b8x, temp_output_dir)
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "sequences:" in content

    @pytest.mark.parametrize("structure_fixture", ["structure_1vme", "structure_6b8x"])
    def test_works_with_both_test_structures(
        self, structure_fixture: str, temp_output_dir: Path, request
    ):
        structure = request.getfixturevalue(structure_fixture)
        yaml_path = create_boltz_input_from_structure(structure, temp_output_dir)
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert len(content) > 0
        assert "sequences:" in content


@pytest.mark.slow
class TestBoltz1WrapperInitialization:
    """Test Boltz1Wrapper initialization and setup."""

    def test_wrapper_initializes(self, boltz1_wrapper: Boltz1Wrapper):
        assert boltz1_wrapper is not None
        assert hasattr(boltz1_wrapper, "model")
        assert hasattr(boltz1_wrapper, "device")

    def test_noise_schedule_created(self, boltz1_wrapper: Boltz1Wrapper):
        schedule = boltz1_wrapper.get_noise_schedule()
        assert "sigma_tm" in schedule
        assert "sigma_t" in schedule
        assert "gamma" in schedule

    def test_noise_schedule_shapes(self, boltz1_wrapper: Boltz1Wrapper):
        schedule = boltz1_wrapper.get_noise_schedule()
        assert torch.is_tensor(schedule["sigma_tm"])
        assert torch.is_tensor(schedule["sigma_t"])
        assert torch.is_tensor(schedule["gamma"])
        assert len(schedule["sigma_tm"]) == boltz1_wrapper.predict_args.sampling_steps
        assert len(schedule["sigma_t"]) == boltz1_wrapper.predict_args.sampling_steps
        assert len(schedule["gamma"]) == boltz1_wrapper.predict_args.sampling_steps

    def test_model_on_correct_device(self, boltz1_wrapper: Boltz1Wrapper, device):
        assert boltz1_wrapper.device == device
        assert next(boltz1_wrapper.model.parameters()).device == device


@pytest.mark.slow
class TestBoltz2WrapperInitialization:
    """Test Boltz2Wrapper initialization and setup."""

    def test_wrapper_initializes(self, boltz2_wrapper: Boltz2Wrapper):
        assert boltz2_wrapper is not None
        assert hasattr(boltz2_wrapper, "model")
        assert hasattr(boltz2_wrapper, "device")

    def test_noise_schedule_created(self, boltz2_wrapper: Boltz2Wrapper):
        schedule = boltz2_wrapper.get_noise_schedule()
        assert "sigma_tm" in schedule
        assert "sigma_t" in schedule
        assert "gamma" in schedule

    def test_noise_schedule_shapes(self, boltz2_wrapper: Boltz2Wrapper):
        schedule = boltz2_wrapper.get_noise_schedule()
        assert torch.is_tensor(schedule["sigma_tm"])
        assert torch.is_tensor(schedule["sigma_t"])
        assert torch.is_tensor(schedule["gamma"])
        assert len(schedule["sigma_tm"]) == boltz2_wrapper.predict_args.sampling_steps
        assert len(schedule["sigma_t"]) == boltz2_wrapper.predict_args.sampling_steps
        assert len(schedule["gamma"]) == boltz2_wrapper.predict_args.sampling_steps

    def test_model_on_correct_device(self, boltz2_wrapper: Boltz2Wrapper, device):
        assert boltz2_wrapper.device == device
        assert next(boltz2_wrapper.model.parameters()).device == device


@pytest.mark.slow
class TestBoltz1WrapperFeaturize:
    """Test Boltz1Wrapper featurize method."""

    def test_featurize_returns_features(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0
        # featurize returns raw batch features from dataloader
        assert "token_pad_mask" in features
        assert "atom_pad_mask" in features

    def test_featurize_with_cif_structure(
        self, boltz1_wrapper: Boltz1Wrapper, structure_1vme: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_1vme, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_featurize_with_pdb_structure(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_featurize_creates_data_module(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert hasattr(boltz1_wrapper, "data_module")
        assert boltz1_wrapper.data_module is not None


@pytest.mark.slow
class TestBoltz2WrapperFeaturize:
    """Test Boltz2Wrapper featurize method."""

    def test_featurize_returns_features(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0
        # featurize returns raw batch features from dataloader
        assert "token_pad_mask" in features
        assert "atom_pad_mask" in features

    def test_featurize_with_cif_structure(
        self, boltz2_wrapper: Boltz2Wrapper, structure_1vme: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_1vme, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0
        # featurize returns raw batch features from dataloader
        assert "token_pad_mask" in features
        assert "atom_pad_mask" in features

    def test_featurize_with_pdb_structure(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert isinstance(features, dict)
        assert len(features) > 0
        # featurize returns raw batch features from dataloader
        assert "token_pad_mask" in features
        assert "atom_pad_mask" in features

    def test_featurize_creates_data_module(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        assert hasattr(boltz2_wrapper, "data_module")
        assert boltz2_wrapper.data_module is not None


@pytest.mark.slow
class TestBoltz1WrapperStep:
    """Test Boltz1Wrapper step method."""

    def test_step_returns_outputs(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz1_wrapper.step(features, grad_needed=False)
        assert isinstance(output, dict)
        assert "s" in output
        assert "z" in output
        assert "feats" in output
        assert "s_inputs" in output
        assert "relative_position_encoding" in output

    def test_step_with_grad_enabled(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz1_wrapper.step(features, grad_needed=True)
        assert isinstance(output, dict)

    def test_step_respects_recycling_steps(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz1_wrapper.step(features, grad_needed=False, recycling_steps=0)
        assert isinstance(output, dict)


@pytest.mark.slow
class TestBoltz2WrapperStep:
    """Test Boltz2Wrapper step method."""

    def test_step_returns_outputs(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz2_wrapper.step(features, grad_needed=False)
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

    def test_step_with_grad_enabled(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz2_wrapper.step(features, grad_needed=True)
        assert isinstance(output, dict)

    def test_step_respects_recycling_steps(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        output = boltz2_wrapper.step(features, grad_needed=False, recycling_steps=0)
        assert isinstance(output, dict)


@pytest.mark.slow
class TestBoltz1WrapperNoiseSchedule:
    """Test Boltz1Wrapper noise schedule methods."""

    def test_get_timestep_scaling_at_start(self, boltz1_wrapper: Boltz1Wrapper):
        scaling = boltz1_wrapper.get_timestep_scaling(0)
        assert "t_hat" in scaling
        assert "sigma_t" in scaling
        assert "eps_scale" in scaling
        assert all(isinstance(v, float) for v in scaling.values())

    def test_get_timestep_scaling_at_middle(self, boltz1_wrapper: Boltz1Wrapper):
        mid_timestep = boltz1_wrapper.predict_args.sampling_steps // 2
        scaling = boltz1_wrapper.get_timestep_scaling(mid_timestep)
        assert all(isinstance(v, float) for v in scaling.values())

    def test_get_timestep_scaling_at_end(self, boltz1_wrapper: Boltz1Wrapper):
        last_timestep = boltz1_wrapper.predict_args.sampling_steps - 1
        scaling = boltz1_wrapper.get_timestep_scaling(last_timestep)
        assert all(isinstance(v, float) for v in scaling.values())

    def test_timestep_scalings_are_positive(self, boltz1_wrapper: Boltz1Wrapper):
        for t in [0, 5, 9]:
            scaling = boltz1_wrapper.get_timestep_scaling(t)
            assert scaling["sigma_t"] >= 0
            assert scaling["eps_scale"] >= 0


@pytest.mark.slow
class TestBoltz2WrapperNoiseSchedule:
    """Test Boltz2Wrapper noise schedule methods."""

    def test_get_timestep_scaling_at_start(self, boltz2_wrapper: Boltz2Wrapper):
        scaling = boltz2_wrapper.get_timestep_scaling(0)
        assert "t_hat" in scaling
        assert "sigma_t" in scaling
        assert "eps_scale" in scaling
        assert all(isinstance(v, float) for v in scaling.values())

    def test_get_timestep_scaling_at_middle(self, boltz2_wrapper: Boltz2Wrapper):
        mid_timestep = boltz2_wrapper.predict_args.sampling_steps // 2
        scaling = boltz2_wrapper.get_timestep_scaling(mid_timestep)
        assert all(isinstance(v, float) for v in scaling.values())

    def test_get_timestep_scaling_at_end(self, boltz2_wrapper: Boltz2Wrapper):
        last_timestep = boltz2_wrapper.predict_args.sampling_steps - 1
        scaling = boltz2_wrapper.get_timestep_scaling(last_timestep)
        assert all(isinstance(v, float) for v in scaling.values())

    def test_timestep_scalings_are_positive(self, boltz2_wrapper: Boltz2Wrapper):
        for t in [0, 5, 9]:
            scaling = boltz2_wrapper.get_timestep_scaling(t)
            assert scaling["sigma_t"] >= 0
            assert scaling["eps_scale"] >= 0


@pytest.mark.slow
class TestBoltz1WrapperInitializeFromNoise:
    """Test Boltz1Wrapper initialize_from_noise method."""

    def test_initialize_from_noise_returns_tensor(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict
    ):
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        assert torch.is_tensor(noisy_coords)

    def test_initialize_from_noise_correct_shape(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict
    ):
        coords = structure_6b8x["asym_unit"].coord
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        assert (
            noisy_coords.shape
            == coords[:, structure_6b8x["asym_unit"].occupancy > 0].shape
        )
        assert noisy_coords.shape[-1] == 3

    def test_initialize_from_noise_adds_noise(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict
    ):
        coords = structure_6b8x["asym_unit"].coord
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=5
        )
        assert not torch.allclose(
            noisy_coords,
            torch.tensor(
                coords[:, structure_6b8x["asym_unit"].occupancy > 0],
                device=boltz1_wrapper.device,
            ),
        )

    def test_initialize_from_noise_at_different_levels(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict
    ):
        for noise_level in [0, 5, 9]:
            noisy_coords = boltz1_wrapper.initialize_from_noise(
                structure_6b8x, noise_level=noise_level
            )
            assert torch.is_tensor(noisy_coords)
            assert noisy_coords.shape[-1] == 3

    def test_initialize_from_noise_raises_without_asym_unit(
        self, boltz1_wrapper: Boltz1Wrapper
    ):
        bad_structure = {"metadata": {}}
        with pytest.raises(ValueError, match="asym_unit"):
            boltz1_wrapper.initialize_from_noise(bad_structure, noise_level=0)


@pytest.mark.slow
class TestBoltz2WrapperInitializeFromNoise:
    """Test Boltz2Wrapper initialize_from_noise method."""

    def test_initialize_from_noise_returns_tensor(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict
    ):
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        assert torch.is_tensor(noisy_coords)

    def test_initialize_from_noise_correct_shape(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict
    ):
        coords = structure_6b8x["asym_unit"].coord
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        assert (
            noisy_coords.shape
            == coords[:, structure_6b8x["asym_unit"].occupancy > 0].shape
        )
        assert noisy_coords.shape[-1] == 3

    def test_initialize_from_noise_adds_noise(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict
    ):
        coords = structure_6b8x["asym_unit"].coord
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=5
        )
        assert not torch.allclose(
            noisy_coords,
            torch.tensor(
                coords[:, structure_6b8x["asym_unit"].occupancy > 0],
                device=boltz2_wrapper.device,
            ),
        )

    def test_initialize_from_noise_at_different_levels(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict
    ):
        for noise_level in [0, 5, 9]:
            noisy_coords = boltz2_wrapper.initialize_from_noise(
                structure_6b8x, noise_level=noise_level
            )
            assert torch.is_tensor(noisy_coords)
            assert noisy_coords.shape[-1] == 3

    def test_initialize_from_noise_raises_without_asym_unit(
        self, boltz2_wrapper: Boltz2Wrapper
    ):
        bad_structure = {"metadata": {}}
        with pytest.raises(ValueError, match="asym_unit"):
            boltz2_wrapper.initialize_from_noise(bad_structure, noise_level=0)


@pytest.mark.slow
class TestBoltz1WrapperDenoiseStep:
    """Test Boltz1Wrapper denoise_step method."""

    def test_denoise_step_returns_dict(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz1_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert isinstance(output, dict)
        assert "atom_coords_denoised" in output

    def test_denoise_step_output_is_tensor(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz1_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])
        assert output["atom_coords_denoised"].shape[-1] == 3

    def test_denoise_step_with_grad(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz1_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=True,
            align_to_input=False,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])

    def test_denoise_step_raises_on_missing_features(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict
    ):
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        bad_features = {"s": None}
        with pytest.raises(KeyError, match="token_pad_mask"):
            boltz1_wrapper.denoise_step(
                bad_features,
                noisy_coords,
                timestep=0,
                align_to_input=False,
                overwrite_representations=True,
            )

        bad_features = {
            "s": None,
            "z": None,
            "s_inputs": None,
            "relative_position_encoding": None,
            "feats": None,
        }
        boltz1_wrapper.cached_representations = bad_features

        with pytest.raises(ValueError, match="Missing required features"):
            boltz1_wrapper.denoise_step(
                bad_features,
                noisy_coords,
                timestep=0,
                align_to_input=False,
                overwrite_representations=False,
            )

    def test_denoise_step_with_augmentation_disabled(
        self, boltz1_wrapper: Boltz1Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz1_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz1_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz1_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            augmentation=False,
            align_to_input=False,
            overwrite_representations=True,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])


@pytest.mark.slow
class TestBoltz2WrapperDenoiseStep:
    """Test Boltz2Wrapper denoise_step method."""

    def test_denoise_step_returns_dict(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz2_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert isinstance(output, dict)
        assert "atom_coords_denoised" in output

    def test_denoise_step_output_is_tensor(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz2_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])
        assert output["atom_coords_denoised"].shape[-1] == 3

    def test_denoise_step_with_grad(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz2_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=True,
            align_to_input=False,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])

    def test_denoise_step_raises_on_missing_features(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict
    ):
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        bad_features = {"s": None}

        with pytest.raises(KeyError, match="token_pad_mask"):
            boltz2_wrapper.denoise_step(
                bad_features,
                noisy_coords,
                timestep=0,
                align_to_input=False,
                overwrite_representations=True,
            )

        bad_features = {
            "s": None,
            "z": None,
            "s_inputs": None,
            "relative_position_encoding": None,
            "feats": None,
        }
        boltz2_wrapper.cached_representations = bad_features

        with pytest.raises(ValueError, match="Missing required features"):
            boltz2_wrapper.denoise_step(
                bad_features,
                noisy_coords,
                timestep=0,
                align_to_input=False,
                overwrite_representations=False,
            )

    def test_denoise_step_with_augmentation_disabled(
        self, boltz2_wrapper: Boltz2Wrapper, structure_6b8x: dict, temp_output_dir: Path
    ):
        features = boltz2_wrapper.featurize(structure_6b8x, out_dir=temp_output_dir)
        noisy_coords = boltz2_wrapper.initialize_from_noise(
            structure_6b8x, noise_level=0
        )
        output = boltz2_wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            augmentation=False,
            align_to_input=False,
            overwrite_representations=True,
        )
        assert torch.is_tensor(output["atom_coords_denoised"])


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
        wrapper = request.getfixturevalue(wrapper_fixture)
        structure = request.getfixturevalue(structure_fixture)

        features = wrapper.featurize(structure, out_dir=temp_output_dir)
        assert isinstance(features, dict)

        step_output = wrapper.step(features)
        assert isinstance(step_output, dict)

        schedule = wrapper.get_noise_schedule()
        assert isinstance(schedule, dict)

        scaling = wrapper.get_timestep_scaling(0)
        assert isinstance(scaling, dict)

        noisy_coords = wrapper.initialize_from_noise(structure, noise_level=0)
        assert torch.is_tensor(noisy_coords)

        output = wrapper.denoise_step(
            features,
            noisy_coords,
            timestep=0,
            grad_needed=False,
            align_to_input=False,
            overwrite_representations=True,
        )
        assert "atom_coords_denoised" in output
        assert torch.is_tensor(output["atom_coords_denoised"])
