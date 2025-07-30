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

    def test_normal_regression(self, sample_data):
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

    def test_prediction(self, sample_data):
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
    def test_edge_cases_table_driven(
        self, x_data, y_data, expected_slope, expected_intercept
    ):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)

        regressor = TlsRegressor(x, y)

        # Test with new property method API
        assert abs(regressor.slope() - expected_slope) < 0.1, (
            f"Slope mismatch: expected {expected_slope}, got {regressor.slope()}"
        )
        assert abs(regressor.intercept() - expected_intercept) < 0.1, (
            f"Intercept mismatch: expected {expected_intercept}, got {regressor.intercept()}"
        )

    def test_boundary_cases(self):
        """Test boundary conditions."""
        # Minimum data points
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 6.0])
        regressor = TlsRegressor(x, y)
        assert abs(regressor.slope() - 3.0) < 1e-10
        assert abs(regressor.intercept() - 0.0) < 1e-10
