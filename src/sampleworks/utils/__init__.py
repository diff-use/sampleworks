"""Utility functions and helpers for sampleworks.

Note: Model availability flags and decorators are available via lazy import
from sampleworks.utils.imports to avoid circular import issues.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sampleworks.utils.imports import (
        BOLTZ_AVAILABLE,
        check_any_model_available,
        check_boltz_available,
        check_protenix_available,
        check_rf3_available,
        PROTENIX_AVAILABLE,
        require_any_model,
        require_boltz,
        require_protenix,
        require_rf3,
        RF3_AVAILABLE,
    )


def __getattr__(name: str):
    """Lazy import for model availability flags and decorators."""
    _imports_exports = {
        "BOLTZ_AVAILABLE",
        "PROTENIX_AVAILABLE",
        "RF3_AVAILABLE",
        "check_any_model_available",
        "check_boltz_available",
        "check_protenix_available",
        "check_rf3_available",
        "require_any_model",
        "require_boltz",
        "require_protenix",
        "require_rf3",
    }
    if name in _imports_exports:
        from sampleworks.utils import imports

        return getattr(imports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BOLTZ_AVAILABLE",
    "PROTENIX_AVAILABLE",
    "RF3_AVAILABLE",
    "check_any_model_available",
    "check_boltz_available",
    "check_protenix_available",
    "check_rf3_available",
    "require_any_model",
    "require_boltz",
    "require_protenix",
    "require_rf3",
]
