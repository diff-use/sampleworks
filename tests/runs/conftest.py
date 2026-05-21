"""Shared test fixtures for preset-runner tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def force_pixi_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep argv assertions deterministic on machines with /app/.pixi present."""
    monkeypatch.setenv("SAMPLEWORKS_FORCE_PIXI", "1")
