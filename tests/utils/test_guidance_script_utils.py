"""Tests for guidance_script_utils saving helpers."""

import json
from pathlib import Path

import pytest
import torch
from sampleworks.utils.guidance_script_arguments import GuidanceConfig, JobResult
from sampleworks.utils.guidance_script_utils import _write_job_metadata, save_everything

from tests.utils.atom_array_builders import build_test_atom_array


def test_save_everything_uses_model_atom_array_for_mismatch(tmp_path: Path):
    """Mismatch final_state should save with model template when provided."""
    refined_structure = {"asym_unit": build_test_atom_array(n_atoms=3, with_occupancy=True)}
    model_atom_array = build_test_atom_array(n_atoms=5, with_occupancy=False)

    final_state = torch.zeros((1, 5, 3), dtype=torch.float32)

    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )

    save_everything(
        args,
        losses=[],
        refined_structure=refined_structure,
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=final_state,
        model_atom_array=model_atom_array,
    )

    assert (tmp_path / "refined.cif").exists()


def _make_job_result(output_dir: str) -> JobResult:
    return JobResult(
        protein="1l63",
        model="boltz2",
        method=None,
        scaler="pure_guidance",
        ensemble_size=8,
        gradient_weight=0.1,
        gd_steps=200,
        status="success",
        exit_code=0,
        runtime_seconds=12.34,
        started_at="2026-05-05T10:00:00",
        finished_at="2026-05-05T10:00:12.340000",
        log_path=str(Path(output_dir) / "run.log"),
        output_dir=output_dir,
    )


def test_write_job_metadata_without_job_result_writes_guidance_config(tmp_path: Path):
    """Without a JobResult, only GuidanceConfig fields should be written (backup snapshot)."""
    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )

    _write_job_metadata(tmp_path, args)

    metadata_path = tmp_path / "job_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["protein"] == "1l63"
    assert metadata["guidance_type"] == "pure_guidance"
    # JobResult-only fields should not yet be present
    assert "started_at" not in metadata
    assert "finished_at" not in metadata
    assert "runtime_seconds" not in metadata
    assert "status" not in metadata


def test_write_job_metadata_with_job_result_appends_timing_and_status(tmp_path: Path):
    """JobResult fields (timing, status, exit_code) must be merged into job_metadata.json."""
    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(tmp_path),
    )
    job_result = _make_job_result(str(tmp_path))

    _write_job_metadata(tmp_path, args, job_result)

    metadata = json.loads((tmp_path / "job_metadata.json").read_text())
    # GuidanceConfig keys are preserved
    assert metadata["protein"] == "1l63"
    assert metadata["guidance_type"] == "pure_guidance"
    # JobResult-only keys are appended
    assert metadata["started_at"] == "2026-05-05T10:00:00"
    assert metadata["finished_at"] == "2026-05-05T10:00:12.340000"
    assert metadata["runtime_seconds"] == 12.34
    assert metadata["status"] == "success"
    assert metadata["exit_code"] == 0


def test_write_job_metadata_creates_missing_output_dir(tmp_path: Path):
    """Helper should create the output directory if it doesn't exist (failure-path safety)."""
    nested = tmp_path / "does" / "not" / "exist"
    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model="boltz2",
        guidance_type="pure_guidance",
        log_path="dummy",
        output_dir=str(nested),
    )
    job_result = _make_job_result(str(nested))

    _write_job_metadata(nested, args, job_result)

    assert (nested / "job_metadata.json").exists()


def test_write_job_metadata_remaps_job_result_paths_to_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """JobResult's output_dir/log_path must be host-remapped, not left as container paths.

    Without this, the JobResult merge would overwrite the GuidanceConfig host paths with
    container paths, regressing job_metadata.json reproducibility outside the container.
    """
    host_results_dir = str(tmp_path)
    monkeypatch.setenv("SAMPLEWORKS_HOST_RESULTS_DIR", host_results_dir)

    container_output = "/data/results/run42"
    container_log = "/data/results/run42/run.log"
    expected_output = f"{host_results_dir}/run42"
    expected_log = f"{host_results_dir}/run42/run.log"

    args = GuidanceConfig(
        protein="1l63",
        structure=Path("dummy"),
        density=Path("dummy"),
        model="boltz2",
        guidance_type="pure_guidance",
        log_path=container_log,
        output_dir=container_output,
    )
    job_result = JobResult(
        protein="1l63",
        model="boltz2",
        method=None,
        scaler="pure_guidance",
        ensemble_size=8,
        gradient_weight=0.1,
        gd_steps=200,
        status="success",
        exit_code=0,
        runtime_seconds=12.34,
        started_at="2026-05-05T10:00:00",
        finished_at="2026-05-05T10:00:12.340000",
        log_path=container_log,
        output_dir=container_output,
    )

    _write_job_metadata(tmp_path, args, job_result)

    metadata = json.loads((tmp_path / "job_metadata.json").read_text())
    assert metadata["output_dir"] == expected_output
    assert metadata["log_path"] == expected_log
