"""Tests for grid-search result serialization."""

import json

import pytest
from run_grid_search import get_pixi_env, GridSearchConfig, save_results
from sampleworks.runs.schema import VALID_PIXI_ENVS
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.guidance_script_arguments import JobResult


def test_save_results_normalizes_legacy_model_key(tmp_path) -> None:
    """Existing result records migrate to ``model_name`` without duplicates."""
    legacy_run = {
        "protein": "1abc",
        "model": "boltz2",
        "method": None,
        "scaler": "pure_guidance",
        "ensemble_size": 1,
        "gradient_weight": 0.1,
        "gd_steps": 1,
        "status": "failed",
    }
    (tmp_path / "results.json").write_text(json.dumps({"runs": [legacy_run]}))
    result = JobResult(
        protein="1abc",
        model_name="boltz2",
        method=None,
        scaler="pure_guidance",
        ensemble_size=1,
        gradient_weight=0.1,
        gd_steps=1,
        status="success",
        exit_code=0,
        runtime_seconds=1.0,
        started_at="2026-07-13T00:00:00",
        finished_at="2026-07-13T00:00:01",
        log_path="run.log",
        output_dir="output",
    )
    config = GridSearchConfig(
        model_name="boltz2",
        scalers=["pure_guidance"],
        ensemble_sizes=[1],
        gradient_weights=[0.1],
        gd_steps=[1],
        method="",
        proteins_file="proteins.csv",
        output_dir=str(tmp_path),
    )

    save_results([result], config, str(tmp_path), total_time=1.0)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved["runs"]) == 1
    assert saved["runs"][0]["model_name"] == "boltz2"
    assert "model" not in saved["runs"][0]


def test_every_structure_predictor_has_a_valid_pixi_env() -> None:
    """Each supported model resolves to a pixi environment declared in pyproject.

    A model that reaches the grid search without a mapped environment only
    fails inside the worker subprocess, so this is checked up front.
    """
    for predictor in StructurePredictor:
        assert get_pixi_env(predictor) in VALID_PIXI_ENVS


def test_get_pixi_env_rejects_unknown_model() -> None:
    """An unrecognized model name fails with the valid options listed."""
    with pytest.raises(ValueError, match="Unknown model: not-a-model"):
        get_pixi_env("not-a-model")
