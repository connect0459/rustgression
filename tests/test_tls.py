"""
Tests for Total Least Squares (TLS) regression.
"""

import numpy as np
import pytest

from rustgression import TlsRegressionParams, TlsRegressor


@pytest.fixture
def sample_data():
    """Generate sample data for testing.

    Returns:
        tuple: A tuple containing the input features (x) and the target values (y).
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    # y = 2x + 1 + noise
    y = 2 * x + 1 + np.random.normal(0, 0.5, 100)
    return x, y


class TestTlsRegressor:
    """Tests for the TlsRegressor class."""

    def test_returns_accurate_slope_intercept_and_r_value_for_noisy_data(
        self, sample_data
    ):
        """Test normal regression analysis."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)

        # Test new property method API
        assert 1.9 < regressor.slope() < 2.1
        assert 0.8 < regressor.intercept() < 1.2
        assert 0.95 < regressor.r_value() < 1.0

        # Test legacy get_params() API (with deprecation warning)
        with pytest.warns(DeprecationWarning):
            params = regressor.get_params()
            assert isinstance(params, TlsRegressionParams)
            assert 1.9 < params.slope < 2.1

    def test_predicted_values_yield_r_squared_above_0_95(self, sample_data):
        """Test prediction functionality."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)

        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape

        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95

    @pytest.mark.parametrize(
        "x_data,y_data,expected_slope,expected_intercept",
        [
            ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0], 2.0, 0.0),
            ([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 7.0, 9.0, 11.0, 13.0], 2.0, 5.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0], -2.0, 12.0),
            ([1.0, 2.0, 3.0], [5.0, 10.0, 15.0], 5.0, 0.0),
        ],
    )
    def test_correctly_fits_various_linear_patterns(
        self, x_data, y_data, expected_slope, expected_intercept
    ):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)

        regressor = TlsRegressor(x, y)

        assert abs(regressor.slope() - expected_slope) < 1e-10, (
            f"Slope mismatch: expected {expected_slope}, got {regressor.slope()}"
        )
        assert abs(regressor.intercept() - expected_intercept) < 1e-10, (
            f"Intercept mismatch: expected {expected_intercept}, got {regressor.intercept()}"
        )

    def test_computes_exact_slope_and_intercept_for_two_data_points(self):
        """Test boundary conditions."""
        # Minimum data points: y = 3x, so TLS slope=3, intercept=0
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 6.0])
        regressor = TlsRegressor(x, y)
        assert abs(regressor.slope() - 3.0) < 1e-10
        assert abs(regressor.intercept() - 0.0) < 1e-10

    def test_r_squared_equals_square_of_r_value(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        assert abs(regressor.r_squared() - regressor.r_value() ** 2) < 1e-12

    def test_r_squared_is_within_unit_interval_for_noisy_data(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        assert 0.0 <= regressor.r_squared() <= 1.0

    def test_residuals_shape_matches_y(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        assert regressor.residuals().shape == y.shape

    def test_residuals_reflect_vertical_distances_from_fitted_line(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        expected = y - regressor.predict(x)
        np.testing.assert_allclose(regressor.residuals(), expected, atol=1e-12)

    def test_residuals_squared_sum_equals_ss_res(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        y_pred = regressor.predict(x)
        ss_res_expected = np.sum((y - y_pred) ** 2)
        assert abs((regressor.residuals() ** 2).sum() - ss_res_expected) < 1e-10
