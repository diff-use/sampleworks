"""Shared test fixtures for preset-runner tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def force_pixi_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep argv assertions deterministic across dev and ACTL image environments."""
    monkeypatch.delenv("SAMPLEWORKS_GRID_SEARCH_SCRIPT", raising=False)
    monkeypatch.delenv("SAMPLEWORKS_PIXI_PROJECT_DIR", raising=False)
    for var in (
        "RUNTIME_PIXI",
        "SAMPLEWORKS_ALLOW_RUNTIME_PIXI",
        "SAMPLEWORKS_REQUIRE_PREBUILT_PIXI",
        "SAMPLEWORKS_SKIP_ENV_PREPARE",
    ):
        monkeypatch.delenv(var, raising=False)
    for var in list(os.environ):
        if var.startswith("SAMPLEWORKS_") and var.endswith("_PYTHON"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SAMPLEWORKS_FORCE_PIXI", "1")
