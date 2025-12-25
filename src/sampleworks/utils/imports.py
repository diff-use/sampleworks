from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

BOLTZ_AVAILABLE = False
PROTENIX_AVAILABLE = False
RF3_AVAILABLE = False

try:
    from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper

    BOLTZ_AVAILABLE = True
    del Boltz1Wrapper, Boltz2Wrapper
except (ImportError, ModuleNotFoundError):
    pass

try:
    # we were testing whether we could load our own modules, but
    # that increases the likelihood of a circular import, and this
    # try/except construction makes those hard to debug, so just test
    # that the actual requirements are available.
    from protenix.model.protenix import Protenix
    from runner.msa_search import msa_search

    PROTENIX_AVAILABLE = True
    del Protenix, msa_search
except (ImportError, ModuleNotFoundError):
    pass

try:
    from sampleworks.models.rf3.wrapper import RF3Wrapper

    RF3_AVAILABLE = True
    del RF3Wrapper
except (ImportError, ModuleNotFoundError):
    pass

F = TypeVar("F", bound=Callable[..., Any])


def require_boltz(message: str | None = None) -> Callable[[F], F]:
    """Decorator to require Boltz model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_boltz
    ... def train_boltz_model():
    ...     pass

    >>> @require_boltz("Custom error message")
    ... def custom_function():
    ...     pass
    """
    default_message = "Boltz model wrapper is not available. Install with: pixi install -e boltz"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not BOLTZ_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_protenix(message: str | None = None) -> Callable[[F], F]:
    """Decorator to require Protenix model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_protenix
    ... def train_protenix_model():
    ...     pass

    >>> @require_protenix("Custom error message")
    ... def custom_function():
    ...     pass
    """
    default_message = (
        "Protenix model wrapper is not available. Install with: pixi install -e protenix"
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROTENIX_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_rf3(message: str | None = None) -> Callable[[F], F]:
    """Decorator to require RF3 model availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_rf3
    ... def train_rf3_model():
    ...     pass

    >>> @require_rf3("Custom error message")
    ... def custom_function():
    ...     pass
    """
    default_message = "RF3 model wrapper is not available. Install with: pixi install -e rf3"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not RF3_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_any_model(message: str | None = None) -> Callable[[F], F]:
    """Decorator to require at least one model wrapper availability.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @require_any_model
    ... def train_any_model():
    ...     pass

    >>> @require_any_model("Need at least one model")
    ... def custom_function():
    ...     pass
    """
    default_message = (
        "Neither Boltz nor Protenix wrappers are available. "
        "Please install at least one model wrapper with the appropriate feature group: "
        "'pixi install -e boltz' or 'pixi install -e protenix'"
    )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not BOLTZ_AVAILABLE and not PROTENIX_AVAILABLE:
                error_msg = message or default_message
                try:
                    import pytest

                    pytest.skip(error_msg)
                except ImportError:
                    raise ImportError(error_msg) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def check_boltz_available(message: str | None = None) -> None:
    """Check if Boltz is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If Boltz model wrapper is not available.
    """
    if not BOLTZ_AVAILABLE:
        default_message = (
            "Boltz model wrapper is not available. Install with: pixi install -e boltz"
        )
        raise ImportError(message or default_message)


def check_protenix_available(message: str | None = None) -> None:
    """Check if Protenix is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If Protenix model wrapper is not available.
    """
    if not PROTENIX_AVAILABLE:
        default_message = (
            "Protenix model wrapper is not available. Install with: pixi install -e protenix"
        )
        raise ImportError(message or default_message)


def check_rf3_available(message: str | None = None) -> None:
    """Check if RF3 is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If RF3 model wrapper is not available.
    """
    if not RF3_AVAILABLE:
        default_message = "RF3 model wrapper is not available. Install with: pixi install -e rf3"
        raise ImportError(message or default_message)


def check_any_model_available(message: str | None = None) -> None:
    """Check if at least one model is available, raise ImportError if not.

    Parameters
    ----------
    message: str, optional
        Custom error message. If None, uses default message.

    Raises
    ------
    ImportError
        If no model wrapper is available.
    """
    if not BOLTZ_AVAILABLE and not PROTENIX_AVAILABLE and not RF3_AVAILABLE:
        default_message = (
            "No model wrappers are available. "
            "Please install at least one model wrapper with the appropriate "
            "feature group: 'pixi install -e boltz', 'pixi install -e protenix', "
            "or 'pixi install -e rf3'"
        )
        raise ImportError(message or default_message)
