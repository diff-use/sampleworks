"""Tests for the unified guidance CLI and GuidanceConfig.from_cli()."""

from __future__ import annotations

import pytest
from sampleworks.utils.guidance_script_arguments import GuidanceConfig


COMMON_ARGS = [
    "--protein", "1VME",
    "--structure", "test.cif",
    "--density", "test.ccp4",
    "--resolution", "1.8",
    "--output-dir", "output",
]


class TestFromCliUnified:
    """Test from_cli() when model and guidance_type come from CLI args."""

    def test_boltz2_pure_guidance(self):
        argv = ["--model", "boltz2", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model == "boltz2"
        assert config.guidance_type == "pure_guidance"
        assert config.protein == "1VME"
        assert config.structure == "test.cif"
        assert config.density == "test.ccp4"
        assert config.resolution == 1.8

    def test_rf3_fk_steering(self):
        argv = ["--model", "rf3", "--guidance-type", "fk_steering"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model == "rf3"
        assert config.guidance_type == "fk_steering"
        assert hasattr(config, "num_particles")
        assert hasattr(config, "fk_lambda")

    def test_model_specific_args_boltz2_method(self):
        argv = [
            "--model", "boltz2", "--guidance-type", "pure_guidance",
            "--method", "MD",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.method == "MD"

    def test_model_specific_args_rf3_msa(self):
        argv = [
            "--model", "rf3", "--guidance-type", "pure_guidance",
            "--msa-path", "/data/msa.a3m",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.msa_path == "/data/msa.a3m"

    def test_guidance_specific_args_fk(self):
        argv = [
            "--model", "boltz1", "--guidance-type", "fk_steering",
            "--num-particles", "5",
            "--fk-lambda", "2.0",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_particles == 5
        assert config.fk_lambda == 2.0

    def test_guidance_specific_args_pure(self):
        argv = [
            "--model", "boltz1", "--guidance-type", "pure_guidance",
            "--step-size", "0.5",
            "--step-scaler-type", "dataspace",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.step_size == 0.5
        assert config.step_scaler_type == "dataspace"


class TestFromCliLegacyScripts:
    """Test from_cli() when model/guidance_type are pre-set (legacy script pattern)."""

    def test_preset_model_and_guidance_type(self):
        config = GuidanceConfig.from_cli(
            COMMON_ARGS,
            model="boltz2",
            guidance_type="pure_guidance",
        )
        assert config.model == "boltz2"
        assert config.guidance_type == "pure_guidance"
        assert config.protein == "1VME"

    def test_no_model_guidance_type_on_cli_required(self):
        """Legacy scripts should not require --model/--guidance-type on CLI."""
        config = GuidanceConfig.from_cli(
            COMMON_ARGS,
            model="protenix",
            guidance_type="fk_steering",
        )
        assert config.model == "protenix"
        assert config.guidance_type == "fk_steering"

    @pytest.mark.parametrize("model", ["boltz1", "boltz2", "protenix", "rf3"])
    @pytest.mark.parametrize("guidance_type", ["pure_guidance", "fk_steering"])
    def test_all_model_guidance_combos(self, model, guidance_type):
        config = GuidanceConfig.from_cli(
            COMMON_ARGS, model=model, guidance_type=guidance_type,
        )
        assert config.model == model
        assert config.guidance_type == guidance_type


class TestFromCliValidation:
    """Test error handling for invalid inputs."""

    def test_missing_protein_errors(self):
        argv = [
            "--model", "boltz2", "--guidance-type", "pure_guidance",
            "--structure", "test.cif",
            "--density", "test.ccp4",
            "--resolution", "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_model_errors(self):
        argv = ["--model", "invalid_model", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_guidance_type_errors(self):
        argv = ["--model", "boltz2", "--guidance-type", "invalid_type"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_missing_required_structure_errors(self):
        argv = [
            "--model", "boltz2", "--guidance-type", "pure_guidance",
            "--protein", "1VME",
            "--density", "test.ccp4",
            "--resolution", "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestFromCliDefaults:
    """Test that defaults are applied correctly."""

    def test_generic_defaults(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.output_dir == "output"
        assert config.partial_diffusion_step == 0
        assert config.loss_order == 2
        assert config.guidance_start == -1
        assert config.augmentation is False
        assert config.align_to_input is False
        assert config.ensemble_size == 4

    def test_boolean_flags(self):
        argv = [
            "--model", "boltz1", "--guidance-type", "pure_guidance",
            "--augmentation", "--align-to-input", "--gradient-normalization", "--em",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.augmentation is True
        assert config.align_to_input is True
        assert config.gradient_normalization is True
        assert config.em is True
