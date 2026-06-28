"""
Tests that numerical diagnostics are routed through Python's warnings
module and are not written directly to stderr.
"""

import warnings

import numpy as np
import pytest

from rustgression import NumericalWarning, OlsRegressor, TlsRegressor


class TestNumericalWarning:
    """NumericalWarning must be a subclass of UserWarning."""

    def test_numerical_warning_is_subclass_of_user_warning(self):
        assert issubclass(NumericalWarning, UserWarning)

    def test_numerical_warning_is_importable_from_package(self):
        from rustgression import NumericalWarning as W

        assert W is not None


class TestConditionNumberWarning:
    """Large SVD condition number must emit NumericalWarning, not write to stderr."""

    def test_large_condition_number_emits_numerical_warning(self):
        x = np.array([1e-8, 2e-8, 3e-8])
        y = np.array([2e-8, 4e-8, 6e-8])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) >= 1

    def test_condition_number_warning_message_describes_condition_number(self):
        x = np.array([1e-8, 2e-8, 3e-8])
        y = np.array([2e-8, 4e-8, 6e-8])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert any("condition number" in str(w.message).lower() for w in numerical)

    def test_condition_number_warning_can_be_suppressed(self):
        x = np.array([1e-8, 2e-8, 3e-8])
        y = np.array([2e-8, 4e-8, 6e-8])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", NumericalWarning)
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) == 0

    def test_condition_number_warning_can_be_escalated_to_error(self):
        x = np.array([1e-8, 2e-8, 3e-8])
        y = np.array([2e-8, 4e-8, 6e-8])
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumericalWarning)
            with pytest.raises(NumericalWarning):
                TlsRegressor(x, y)

    def test_well_conditioned_data_does_not_emit_numerical_warning(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) == 0


class TestSubnormalWarning:
    """Subnormal input values must emit NumericalWarning from both OLS and TLS."""

    def test_tls_subnormal_y_values_emit_numerical_warning(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1e-320, 2e-320, 3e-320])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) >= 1

    def test_ols_subnormal_y_values_emit_numerical_warning(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1e-320, 2e-320, 3e-320])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            OlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) >= 1

    def test_subnormal_warning_message_identifies_subnormal_and_array_name(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1e-320, 2e-320, 3e-320])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert any("subnormal" in str(w.message).lower() for w in numerical)

    def test_subnormal_warning_can_be_suppressed(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1e-320, 2e-320, 3e-320])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", NumericalWarning)
            TlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) == 0

    def test_normal_values_do_not_emit_subnormal_warning(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            OlsRegressor(x, y)
        numerical = [w for w in caught if issubclass(w.category, NumericalWarning)]
        assert len(numerical) == 0
