"""Utility functions and helpers for sampleworks."""

from typing import Any

from sampleworks.utils.imports import (
    BOLTZ_AVAILABLE,
    check_any_model_available,
    check_boltz_available,
    check_protenix_available,
    PROTENIX_AVAILABLE,
    require_any_model,
    require_boltz,
    require_protenix,
)


def do_nothing(*args: Any, **kwargs: Any) -> None:
    """Does nothing, just returns None"""
    pass


__all__ = [
    "BOLTZ_AVAILABLE",
    "PROTENIX_AVAILABLE",
    "check_any_model_available",
    "check_boltz_available",
    "check_protenix_available",
    "require_any_model",
    "require_boltz",
    "require_protenix",
]
