"""Tests for the centralized imports module."""

import pytest
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


class TestAvailabilityFlags:
    """Test that availability flags are properly set."""

    def test_boltz_available_is_bool(self):
        """BOLTZ_AVAILABLE should be a boolean."""
        assert isinstance(BOLTZ_AVAILABLE, bool)

    def test_protenix_available_is_bool(self):
        """PROTENIX_AVAILABLE should be a boolean."""
        assert isinstance(PROTENIX_AVAILABLE, bool)

    def test_at_least_one_available(self):
        """At least one model should be available in test environment."""
        assert BOLTZ_AVAILABLE or PROTENIX_AVAILABLE


class TestCheckFunctions:
    """Test the check_*_available helper functions."""

    def test_check_any_model_available_no_error(self):
        """check_any_model_available should not raise if any model available."""
        if BOLTZ_AVAILABLE or PROTENIX_AVAILABLE:
            check_any_model_available()

    def test_check_boltz_available(self):
        """check_boltz_available should match BOLTZ_AVAILABLE flag."""
        if BOLTZ_AVAILABLE:
            check_boltz_available()
        else:
            with pytest.raises(ImportError, match="Boltz model wrapper"):
                check_boltz_available()

    def test_check_protenix_available(self):
        """check_protenix_available should match PROTENIX_AVAILABLE flag."""
        if PROTENIX_AVAILABLE:
            check_protenix_available()
        else:
            with pytest.raises(ImportError, match="Protenix model wrapper"):
                check_protenix_available()

    def test_check_functions_custom_message(self):
        """Check functions should support custom error messages."""
        custom_msg = "Custom error message for testing"

        if not BOLTZ_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_boltz_available(custom_msg)

        if not PROTENIX_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_protenix_available(custom_msg)

        if not BOLTZ_AVAILABLE and not PROTENIX_AVAILABLE:
            with pytest.raises(ImportError, match=custom_msg):
                check_any_model_available(custom_msg)


class TestRequireDecorators:
    """Test the require_* decorators."""

    def test_require_boltz_decorator(self):
        """@require_boltz should skip if Boltz unavailable."""

        @require_boltz()
        def boltz_function():
            return "executed"

        if BOLTZ_AVAILABLE:
            assert boltz_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                boltz_function()

    def test_require_protenix_decorator(self):
        """@require_protenix should skip if Protenix unavailable."""

        @require_protenix()
        def protenix_function():
            return "executed"

        if PROTENIX_AVAILABLE:
            assert protenix_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                protenix_function()

    def test_require_any_model_decorator(self):
        """@require_any_model should skip if no models available."""

        @require_any_model()
        def any_model_function():
            return "executed"

        if BOLTZ_AVAILABLE or PROTENIX_AVAILABLE:
            assert any_model_function() == "executed"
        else:
            with pytest.raises(pytest.skip.Exception):
                any_model_function()

    def test_decorator_preserves_function_metadata(self):
        """Decorators should preserve function name and docstring."""

        @require_any_model()
        def example_function():
            """Example docstring."""
            pass

        assert example_function.__name__ == "example_function"
        assert example_function.__doc__ == "Example docstring."

    def test_decorator_with_arguments(self):
        """Decorators should work with functions that have arguments."""

        @require_any_model()
        def function_with_args(x: int, y: int) -> int:
            return x + y

        if BOLTZ_AVAILABLE or PROTENIX_AVAILABLE:
            assert function_with_args(2, 3) == 5

    def test_decorator_custom_message(self):
        """Decorators should support custom error messages."""
        custom_msg = "Custom decorator error message"

        @require_boltz(custom_msg)
        def custom_boltz_function():
            return "executed"

        if not BOLTZ_AVAILABLE:
            with pytest.raises(pytest.skip.Exception, match=custom_msg):
                custom_boltz_function()


class TestReexports:
    """Test that re-exports from utils.__init__ work."""

    def test_reexports_from_utils_package(self):
        """All imports should be available from sampleworks.utils."""
        from sampleworks.utils import (
            BOLTZ_AVAILABLE as BOLTZ_REEXPORT,
            check_any_model_available as check_any_reexport,
            check_boltz_available as check_boltz_reexport,
            check_protenix_available as check_protenix_reexport,
            PROTENIX_AVAILABLE as PROTENIX_REEXPORT,
            require_any_model as require_any_reexport,
            require_boltz as require_boltz_reexport,
            require_protenix as require_protenix_reexport,
        )

        assert BOLTZ_REEXPORT == BOLTZ_AVAILABLE
        assert PROTENIX_REEXPORT == PROTENIX_AVAILABLE
        assert check_any_reexport is check_any_model_available
        assert check_boltz_reexport is check_boltz_available
        assert check_protenix_reexport is check_protenix_available
        assert require_any_reexport is require_any_model
        assert require_boltz_reexport is require_boltz
        assert require_protenix_reexport is require_protenix


class TestDecoratorIntegrationWithPytest:
    """Test decorator integration with pytest test functions."""

    @require_boltz()
    def test_boltz_specific_feature(self):
        """This test should be skipped if Boltz is not available."""
        assert BOLTZ_AVAILABLE

    @require_protenix()
    def test_protenix_specific_feature(self):
        """This test should be skipped if Protenix is not available."""
        assert PROTENIX_AVAILABLE

    @require_any_model()
    def test_any_model_feature(self):
        """This test should be skipped if no models are available."""
        assert BOLTZ_AVAILABLE or PROTENIX_AVAILABLE
