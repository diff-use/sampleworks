"""Tests for the unified guidance CLI and GuidanceConfig.from_cli()."""

from __future__ import annotations

import subprocess
import sys

import pytest
from sampleworks.utils.guidance_script_arguments import GuidanceConfig


COMMON_ARGS = [
    "--protein",
    "1VME",
    "--structure",
    "test.cif",
    "--density",
    "test.ccp4",
    "--resolution",
    "1.8",
    "--output-dir",
    "output",
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
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.method == "MD"

    def test_model_specific_args_rf3_msa(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.msa_path == "/data/msa.a3m"  # ty: ignore[unresolved-attribute]

    def test_guidance_specific_args_fk(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--num-particles",
            "5",
            "--fk-lambda",
            "2.0",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_particles == 5
        assert config.fk_lambda == 2.0

    def test_guidance_specific_args_pure(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--step-size",
            "0.5",
            "--step-scaler-type",
            "dataspace",
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
            COMMON_ARGS,
            model=model,
            guidance_type=guidance_type,
        )
        assert config.model == model
        assert config.guidance_type == guidance_type


class TestFromCliValidation:
    """Test error handling for invalid inputs."""

    def test_missing_protein_errors(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
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
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
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
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--augmentation",
            "--align-to-input",
            "--gradient-normalization",
            "--em",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.augmentation is True
        assert config.align_to_input is True
        assert config.gradient_normalization is True
        assert config.em is True


class TestCrossModelArgRejection:
    """Test that model-specific args are rejected for the wrong model."""

    def test_method_rejected_for_protenix(self):
        argv = [
            "--model",
            "protenix",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_method_rejected_for_rf3(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--method",
            "MD",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_msa_path_rejected_for_boltz1(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_conflicting_model_in_preset_mode(self):
        argv = ["--model", "boltz2"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv, model="rf3", guidance_type="fk_steering")

    def test_conflicting_guidance_type_in_preset_mode(self):
        argv = ["--guidance-type", "fk_steering"] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv, model="boltz1", guidance_type="pure_guidance")

    def test_fk_args_rejected_for_pure_guidance(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--num-particles",
            "5",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_pure_args_rejected_for_fk_steering(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--step-scaler-type",
            "dataspace",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_msa_path_rejected_for_boltz2(self):
        argv = [
            "--model",
            "boltz2",
            "--guidance-type",
            "pure_guidance",
            "--msa-path",
            "/data/msa.a3m",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_chiral_flags_rejected_for_boltz1(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--disable-chiral-features",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestPresetMatchingAccepted:
    """Matching preset values should not trigger false-positive rejection."""

    def test_matching_model_accepted(self):
        argv = ["--model", "boltz1"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv, model="boltz1", guidance_type="fk_steering")
        assert config.model == "boltz1"

    def test_matching_guidance_type_accepted(self):
        argv = ["--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv, model="boltz2", guidance_type="pure_guidance")
        assert config.guidance_type == "pure_guidance"


class TestArgPassthrough:
    """Test that non-default argument values propagate to GuidanceConfig."""

    def test_model_checkpoint(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--model-checkpoint",
            "/custom/path.ckpt",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.model_checkpoint == "/custom/path.ckpt"

    def test_device(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--device",
            "cpu",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.device == "cpu"

    def test_output_dir_override(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--output-dir",
            "/custom/output",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
            "--resolution",
            "1.8",
        ]
        config = GuidanceConfig.from_cli(argv)
        assert config.output_dir == "/custom/output"

    def test_partial_diffusion_step(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--partial-diffusion-step",
            "50",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.partial_diffusion_step == 50

    def test_log_path(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--log-path",
            "/tmp/run.log",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.log_path == "/tmp/run.log"

    def test_ensemble_size(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "fk_steering",
            "--ensemble-size",
            "8",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.ensemble_size == 8

    def test_rf3_chiral_features(self):
        argv = [
            "--model",
            "rf3",
            "--guidance-type",
            "pure_guidance",
            "--disable-chiral-features",
            "--track-chiral-features",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.disable_chiral_features is True
        assert config.track_chiral_features is True

    def test_loss_order(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--loss-order",
            "1",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.loss_order == 1

    def test_guidance_start(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--guidance-start",
            "10",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.guidance_start == 10

    def test_recycling_steps(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--recycling-steps",
            "3",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.recycling_steps == 3

    def test_num_diffusion_steps(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--num-diffusion-steps",
            "100",
        ] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_diffusion_steps == 100

    def test_recycling_steps_default_none(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.recycling_steps is None

    def test_num_diffusion_steps_default(self):
        argv = ["--model", "boltz1", "--guidance-type", "pure_guidance"] + COMMON_ARGS
        config = GuidanceConfig.from_cli(argv)
        assert config.num_diffusion_steps == 200


class TestValidationEdgeCases:
    """Additional validation edge cases."""

    def test_no_args_at_all(self):
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli([])

    def test_missing_resolution(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--density",
            "test.ccp4",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_missing_density(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--protein",
            "1VME",
            "--structure",
            "test.cif",
            "--resolution",
            "1.8",
        ]
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_loss_order(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--loss-order",
            "3",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)

    def test_invalid_step_scaler_type(self):
        argv = [
            "--model",
            "boltz1",
            "--guidance-type",
            "pure_guidance",
            "--step-scaler-type",
            "invalid",
        ] + COMMON_ARGS
        with pytest.raises(SystemExit):
            GuidanceConfig.from_cli(argv)


class TestEntrypointSmoke:
    """Smoke test the actual CLI entrypoint."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "sampleworks.cli.guidance",
                "--model",
                "boltz1",
                "--guidance-type",
                "pure_guidance",
                "--help",
            ],
            capture_output=True,
        )
        assert result.returncode == 0
        assert b"--protein" in result.stdout
        assert b"--structure" in result.stdout

    def test_invalid_preset_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            GuidanceConfig.from_cli(COMMON_ARGS, model="typo", guidance_type="fk_steering")

    def test_invalid_preset_guidance_type_raises(self):
        with pytest.raises(ValueError, match="Unknown guidance type"):
            GuidanceConfig.from_cli(COMMON_ARGS, model="boltz1", guidance_type="typo")
