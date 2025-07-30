"""
Test code for regression analysis classes.

This module contains unit tests for the OlsRegressor and TlsRegressor classes,
including tests for input validation, regression analysis, and comparison of
the two regression methods.

Functions:
----------
- sample_data: Fixture to generate sample data for testing.
- test_create_regressor: Tests the factory function for creating regressors.
- test_input_validation: Tests input validation for the OlsRegressor.
- TestOlsRegressor: Class containing tests for the OlsRegressor.
- TestTlsRegressor: Class containing tests for the TlsRegressor.
- test_compare_methods: Tests the comparison between OLS and TLS regression methods.
"""

import numpy as np
import pytest

from rustgression import (
    OlsRegressionParams,
    OlsRegressor,
    TlsRegressionParams,
    TlsRegressor,
    create_regressor,
)


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


class TestCreateRegressor:
    """Tests for the factory function for creating regressors."""

    def test_normal_case(self, sample_data):
        """Test normal regressor creation."""
        x, y = sample_data

        ols = create_regressor(x, y, "ols")
        assert isinstance(ols, OlsRegressor)

        tls = create_regressor(x, y, "tls")
        assert isinstance(tls, TlsRegressor)

    def test_default_behavior(self, sample_data):
        """Test default regressor creation."""
        x, y = sample_data
        default = create_regressor(x, y)
        assert isinstance(default, OlsRegressor)

    def test_invalid_method(self, sample_data):
        """Test error handling for invalid methods."""
        x, y = sample_data
        with pytest.raises(ValueError, match="Unknown regression method"):
            create_regressor(x, y, "invalid")


class TestInputValidation:
    """Tests for input validation."""

    def test_mismatched_array_lengths(self):
        """Test mismatched array length validation."""
        with pytest.raises(ValueError, match="The lengths of the input arrays do not match."):
            OlsRegressor(np.array([1, 2]), np.array([1]))

    def test_insufficient_data_points(self):
        """Test insufficient data points validation."""
        with pytest.raises(
            ValueError, match="At least two data points are required for regression"
        ):
            OlsRegressor(np.array([1]), np.array([1]))

    @pytest.mark.parametrize("regressor_class", [OlsRegressor, TlsRegressor])
    def test_edge_cases_table_driven(self, regressor_class):
        """Test edge cases with different input patterns."""
        test_cases = [
            ("empty_arrays", np.array([]), np.array([]), ValueError),
            ("single_point", np.array([1.0]), np.array([2.0]), ValueError),
            ("length_mismatch", np.array([1.0, 2.0]), np.array([1.0]), ValueError),
        ]
        
        for name, x, y, expected_exception in test_cases:
            with pytest.raises(expected_exception):
                regressor_class(x, y)


class TestOlsRegressor:
    """Tests for the OlsRegressor class."""

    def test_normal_regression(self, sample_data):
        """Test normal regression analysis."""
        x, y = sample_data
        regressor = OlsRegressor(x, y)

        params = regressor.get_params()
        assert isinstance(params, OlsRegressionParams)
        assert 1.9 < params.slope < 2.1
        assert 0.8 < params.intercept < 1.2
        assert 0.95 < params.r_value < 1.0
        assert params.p_value < 0.05
        assert params.stderr > 0

    def test_prediction(self, sample_data):
        """Test prediction functionality."""
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape
        
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95

    @pytest.mark.parametrize("x_data,y_data,expected_slope,expected_intercept", [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0], 2.0, 0.0),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 7.0, 9.0, 11.0, 13.0], 2.0, 5.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0], -2.0, 12.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0], 0.0, 3.0),
    ])
    def test_edge_cases_table_driven(self, x_data, y_data, expected_slope, expected_intercept):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)
        
        regressor = OlsRegressor(x, y)
        params = regressor.get_params()
        
        assert abs(params.slope - expected_slope) < 0.1, f"Slope mismatch: expected {expected_slope}, got {params.slope}"
        assert abs(params.intercept - expected_intercept) < 0.1, f"Intercept mismatch: expected {expected_intercept}, got {params.intercept}"

    def test_boundary_cases(self):
        """Test boundary conditions."""
        # Minimum data points
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = OlsRegressor(x, y)
        params = regressor.get_params()
        assert abs(params.slope - 2.0) < 1e-10
        assert abs(params.intercept - 0.0) < 1e-10


class TestTlsRegressor:
    """Tests for the TlsRegressor class."""

    def test_normal_regression(self, sample_data):
        """Test normal regression analysis."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)

        params = regressor.get_params()
        assert isinstance(params, TlsRegressionParams)
        assert 1.9 < params.slope < 2.1
        assert 0.8 < params.intercept < 1.2
        assert 0.95 < params.r_value < 1.0

    def test_prediction(self, sample_data):
        """Test prediction functionality."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape
        
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95

    @pytest.mark.parametrize("x_data,y_data,expected_slope,expected_intercept", [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0], 2.0, 0.0),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 7.0, 9.0, 11.0, 13.0], 2.0, 5.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0], -2.0, 12.0),
        ([1.0, 2.0, 3.0], [5.0, 10.0, 15.0], 5.0, 0.0),
    ])
    def test_edge_cases_table_driven(self, x_data, y_data, expected_slope, expected_intercept):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)
        
        regressor = TlsRegressor(x, y)
        params = regressor.get_params()
        
        assert abs(params.slope - expected_slope) < 0.1, f"Slope mismatch: expected {expected_slope}, got {params.slope}"
        assert abs(params.intercept - expected_intercept) < 0.1, f"Intercept mismatch: expected {expected_intercept}, got {params.intercept}"

    def test_boundary_cases(self):
        """Test boundary conditions."""
        # Minimum data points
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 6.0])
        regressor = TlsRegressor(x, y)
        params = regressor.get_params()
        assert abs(params.slope - 3.0) < 1e-10
        assert abs(params.intercept - 0.0) < 1e-10


class TestMethodComparison:
    """Tests for comparing OLS and TLS regression methods."""

    def test_parameter_similarity(self, sample_data):
        """Test parameter similarity between methods."""
        x, y = sample_data

        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)

        ols_params = ols.get_params()
        tls_params = tls.get_params()

        assert abs(ols_params.slope - tls_params.slope) < 0.1
        assert abs(ols_params.intercept - tls_params.intercept) < 0.1
        assert abs(ols_params.r_value - tls_params.r_value) < 0.1

    @pytest.mark.parametrize("x_data,y_data", [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 5.0, 7.0, 9.0]),
    ])
    def test_perfect_correlation_cases(self, x_data, y_data):
        """Test methods with perfect correlation data."""
        x = np.array(x_data)
        y = np.array(y_data)
        
        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)
        
        ols_params = ols.get_params()
        tls_params = tls.get_params()
        
        # For perfect correlation, both methods should yield identical results
        assert abs(ols_params.slope - tls_params.slope) < 1e-10
        assert abs(ols_params.intercept - tls_params.intercept) < 1e-10
