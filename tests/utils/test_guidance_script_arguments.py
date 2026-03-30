"""Tests for guidance script argument handling."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import (
    get_checkpoint,
    GuidanceConfig,
    JobConfig,
    validate_model_checkpoint,
)


# ============================================================================
# get_checkpoint tests
# ============================================================================


def test_get_checkpoint_reads_model_checkpoint():
    """get_checkpoint should return the model_checkpoint value from the namespace."""
    args = Namespace(model_checkpoint="/tmp/model.ckpt")

    assert get_checkpoint(args) == "/tmp/model.ckpt"


def test_get_checkpoint_returns_none_when_missing():
    """get_checkpoint should return None when model_checkpoint is absent."""
    args = Namespace()

    assert get_checkpoint(args) is None


def test_get_checkpoint_treats_empty_string_as_missing():
    """Empty or whitespace-only model_checkpoint should be treated as missing."""
    args = Namespace(model_checkpoint="   ")

    assert get_checkpoint(args) is None


# ============================================================================
# populate_config_for_guidance_type tests
# ============================================================================


def _build_job(model: StructurePredictor) -> JobConfig:
    return JobConfig(
        protein="protein",
        structure_path="/tmp/structure.cif",
        density_path="/tmp/density.mrc",
        resolution=2.0,
        model=model,
        scaler=GuidanceType.PURE_GUIDANCE,
        ensemble_size=1,
        gradient_weight=0.1,
        gd_steps=1,
        method=None,
        output_dir="/tmp/output",
        log_path="/tmp/output/run.log",
    )


@patch(
    "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
    return_value="/checkpoints/mock.ckpt",
)
def test_populate_config_resolves_checkpoint_when_none_provided(_mock_resolve, model_wrapper_type):
    """populate_config_for_guidance_type should auto-resolve checkpoint if no arg exists."""
    config = GuidanceConfig(
        protein="protein",
        structure="/tmp/structure.cif",
        density="/tmp/density.mrc",
        model=model_wrapper_type,
        guidance_type=GuidanceType.PURE_GUIDANCE,
        log_path="/tmp/output/run.log",
    )

    config.populate_config_for_guidance_type(
        _build_job(model_wrapper_type),
        Namespace(use_tweedie=False, step_scaler_type="noisespace"),
    )

    assert config.model_checkpoint == "/checkpoints/mock.ckpt"


def test_populate_config_uses_model_checkpoint_argument(model_wrapper_type):
    """populate_config_for_guidance_type should read the model_checkpoint arg."""
    with patch(
        "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
        return_value="/checkpoints/mock.ckpt",
    ) as mock_resolve:
        config = GuidanceConfig(
            protein="protein",
            structure="/tmp/structure.cif",
            density="/tmp/density.mrc",
            model=model_wrapper_type,
            guidance_type=GuidanceType.PURE_GUIDANCE,
            log_path="/tmp/output/run.log",
        )
        mock_resolve.reset_mock()

        args = Namespace(
            model_checkpoint="/tmp/custom.ckpt",
            use_tweedie=False,
            step_scaler_type="noisespace",
        )
        config.populate_config_for_guidance_type(_build_job(model_wrapper_type), args)
        mock_resolve.assert_not_called()

    assert config.model_checkpoint == "/tmp/custom.ckpt"


# ============================================================================
# validate_model_checkpoint tests
# ============================================================================


@patch(
    "sampleworks.utils.guidance_script_arguments._resolve_checkpoint",
    side_effect=ValueError(
        "Running guidance requires a model checkpoint for 'model'. "
        "Provide --model-checkpoint or bake checkpoints into /checkpoints/."
    ),
)
def test_validate_model_checkpoint_requires_non_empty_value(_mock_resolve, model_wrapper_type):
    """Validation should fail fast when checkpoint is missing and can't be auto-resolved."""
    with pytest.raises(ValueError, match="requires a model checkpoint"):
        validate_model_checkpoint(model_wrapper_type, "")


def test_validate_model_checkpoint_requires_existing_file(model_wrapper_type, tmp_path: Path):
    """Validation should fail for missing files."""
    missing = tmp_path / "does_not_exist.ckpt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_model_checkpoint(model_wrapper_type, str(missing))


def test_validate_model_checkpoint_rejects_directories(model_wrapper_type, tmp_path: Path):
    """Validation should reject directory paths."""
    with pytest.raises(ValueError, match="must be a file"):
        validate_model_checkpoint(model_wrapper_type, str(tmp_path))


def test_validate_model_checkpoint_returns_resolved_path(model_wrapper_type, tmp_path: Path):
    """Validation should return the resolved absolute checkpoint path."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights")

    validated = validate_model_checkpoint(model_wrapper_type, str(checkpoint))

    assert validated == str(checkpoint.resolve())
