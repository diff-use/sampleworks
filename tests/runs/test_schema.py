"""Tests for pixi environment discovery in the run preset schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from sampleworks.runs import schema


MANIFEST = """
[tool.pixi.environments]
zebra = {features = ["zebra"]}
antelope = {features = ["antelope"]}
"""


def _write_manifest(directory: Path, body: str = MANIFEST) -> Path:
    """Write a pixi manifest into ``directory`` and return the directory."""
    (directory / "pyproject.toml").write_text(body)
    return directory


def test_workspace_manifest_declares_the_running_environments() -> None:
    """The checked-out workspace resolves to its own declared environments."""
    assert "boltz" in schema.VALID_PIXI_ENVS
    assert "protpardelle" in schema.VALID_PIXI_ENVS


def test_project_dir_override_selects_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``SAMPLEWORKS_PIXI_PROJECT_DIR`` picks the manifest the runner will use."""
    monkeypatch.setenv("SAMPLEWORKS_PIXI_PROJECT_DIR", str(_write_manifest(tmp_path)))
    assert schema._load_valid_pixi_envs() == ("antelope", "zebra")


def test_missing_manifest_yields_no_environments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unreachable manifest returns empty rather than raising on import.

    A wheel install does not ship ``pyproject.toml``; importing the schema must
    still work so the runner can report a missing environment itself.
    """
    monkeypatch.setattr(schema, "_pixi_manifest_path", lambda: None)
    assert schema._load_valid_pixi_envs() == ()


def test_job_env_is_unvalidated_without_a_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no discoverable environments, any ``env`` is accepted."""
    monkeypatch.setattr(schema, "VALID_PIXI_ENVS", ())
    job = schema.Job(name="j", env="not-a-real-env", output_subdir="out", gpus="none")
    assert job.env == "not-a-real-env"


def test_job_env_is_validated_against_declared_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A known environment list still rejects unknown ``env`` values."""
    monkeypatch.setattr(schema, "VALID_PIXI_ENVS", ("boltz", "rf3"))
    with pytest.raises(ValueError, match="env must be one of"):
        schema.Job(name="j", env="not-a-real-env", output_subdir="out", gpus="none")
