"""
Test cases for edge cases and error handling to improve coverage.
"""

import numpy as np
import pytest
from scipy import stats

from rustgression import NumericalWarning, OlsRegressor, TlsRegressor


class TestRegressorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_repr_includes_class_name_slope_intercept_and_r_value(self):
        """Test string representation of regressors."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        ols = OlsRegressor(x, y)
        with pytest.warns(NumericalWarning):
            tls = TlsRegressor(x, y)

        # Test __repr__ method (line 135 in base.py)
        ols_repr = repr(ols)
        tls_repr = repr(tls)

        assert "OlsRegressor" in ols_repr
        assert "slope=" in ols_repr
        assert "intercept=" in ols_repr
        assert "r_value=" in ols_repr

        assert "TlsRegressor" in tls_repr
        assert "slope=" in tls_repr
        assert "intercept=" in tls_repr
        assert "r_value=" in tls_repr

    def test_prediction_returns_values_matching_linear_formula(self):
        """Test predict method for complete coverage."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        regressor = OlsRegressor(x, y)

        # Test predict method
        x_new = np.array([5.0, 6.0])
        predictions = regressor.predict(x_new)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(x_new)

        # For perfect linear relationship, predictions should be exact
        expected = regressor.slope() * x_new + regressor.intercept()
        np.testing.assert_array_almost_equal(predictions, expected, decimal=10)

    def test_base_regressor_cannot_be_instantiated_directly(self):
        """Test abstract method behaviors."""
        from rustgression.regression.base import BaseRegressor

        # BaseRegressor is abstract and cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseRegressor(np.array([1, 2]), np.array([1, 2]))

    def test_params_expose_slope_intercept_and_r_value(self):
        """Test get_params method for coverage."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        ols = OlsRegressor(x, y)
        with pytest.warns(NumericalWarning):
            tls = TlsRegressor(x, y)

        with pytest.warns(DeprecationWarning):
            ols_params = ols.get_params()
        with pytest.warns(DeprecationWarning):
            tls_params = tls.get_params()

        # Basic validation that params are returned
        assert ols_params is not None
        assert tls_params is not None

        # Check params have expected attributes
        assert hasattr(ols_params, "slope")
        assert hasattr(ols_params, "intercept")
        assert hasattr(ols_params, "r_value")

        assert hasattr(tls_params, "slope")
        assert hasattr(tls_params, "intercept")
        assert hasattr(tls_params, "r_value")

    def test_slope_intercept_and_r_value_match_params_returned_by_get_params(self):
        """Test that regressor properties are consistent with get_params."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 3.5, 5.5, 7.5])

        regressor = OlsRegressor(x, y)
        with pytest.warns(DeprecationWarning):
            params = regressor.get_params()

        # Test that properties match params
        assert abs(regressor.slope() - params.slope) < 1e-10
        assert abs(regressor.intercept() - params.intercept) < 1e-10
        assert abs(regressor.r_value() - params.r_value) < 1e-10

    def test_computes_correct_slope_for_million_scale_inputs(self):
        """Test handling of large numbers."""
        x = np.array([1e6, 2e6, 3e6])
        y = np.array([2e6, 4e6, 6e6])

        regressor = OlsRegressor(x, y)

        # Should handle large numbers without issues
        assert abs(regressor.slope() - 2.0) < 1e-10
        assert abs(regressor.intercept()) < 1e-6
        assert abs(regressor.r_value() - 1.0) < 1e-10

    def test_computes_correct_slope_for_micro_scale_inputs(self):
        """Test handling of very small numbers."""
        x = np.array([1e-6, 2e-6, 3e-6])
        y = np.array([2e-6, 4e-6, 6e-6])

        regressor = OlsRegressor(x, y)

        # Should handle small numbers without issues
        assert abs(regressor.slope() - 2.0) < 1e-10
        assert abs(regressor.intercept()) < 1e-12
        assert abs(regressor.r_value() - 1.0) < 1e-10

    def test_matches_scipy_slope_and_r_value_for_small_magnitude_inputs(self):
        """OLS must produce finite, accurate results for well-defined data at small absolute scale.

        The relationship is exact (slope=2, intercept=0), so both slope and
        r_value must agree with scipy.stats.linregress to numerical tolerance.
        """
        x = np.array([1e-8, 2e-8, 3e-8])
        y = np.array([2e-8, 4e-8, 6e-8])
        ref = stats.linregress(x, y)

        regressor = OlsRegressor(x, y)

        assert abs(regressor.slope() - ref.slope) < 1e-6
        assert abs(regressor.intercept() - ref.intercept) < 1e-20
        assert abs(regressor.r_value() - ref.rvalue) < 1e-10

    def test_slope_is_invariant_to_input_scale(self):
        """OLS slope must be scale-equivariant: scaling x and y must not change the slope."""
        x_small = np.array([1e-8, 2e-8, 3e-8])
        y_small = np.array([2e-8, 4e-8, 6e-8])
        x_normal = x_small * 1e8
        y_normal = y_small * 1e8

        slope_small = OlsRegressor(x_small, y_small).slope()
        slope_normal = OlsRegressor(x_normal, y_normal).slope()

        assert abs(slope_small - slope_normal) < 1e-10

    def test_r_value_is_accurate_when_product_of_variances_would_underflow(self):
        """OLS r_value must be finite and accurate even at extreme small scale.

        At extreme small scale, computing the product of two variance terms before
        taking the square root can underflow to zero, producing NaN. Computing
        individual square roots first avoids this.
        """
        x = np.array([1e-150, 2e-150, 3e-150])
        y = np.array([2e-150, 4e-150, 6e-150])
        with pytest.warns(RuntimeWarning):
            ref = stats.linregress(x, y)

        regressor = OlsRegressor(x, y)

        assert abs(regressor.r_value() - ref.rvalue) < 1e-10
