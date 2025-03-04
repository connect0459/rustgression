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
    RegressionParams,
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


def test_create_regressor(sample_data):
    """Test the factory function for creating regressors.

    This function tests the creation of OlsRegressor and TlsRegressor
    instances, as well as the default behavior and error handling for
    invalid regression methods.
    """
    x, y = sample_data

    # Normal case
    ols = create_regressor(x, y, "ols")
    assert isinstance(ols, OlsRegressor)

    tls = create_regressor(x, y, "tls")
    assert isinstance(tls, TlsRegressor)

    # Default value
    default = create_regressor(x, y)
    assert isinstance(default, OlsRegressor)

    # Error case
    with pytest.raises(ValueError, match="Unknown regression method"):
        create_regressor(x, y, "invalid")


def test_input_validation():
    """Test input validation for the OlsRegressor.

    This function tests the behavior of the OlsRegressor when provided
    with invalid input data, such as mismatched array lengths and
    insufficient data points.
    """
    # Mismatched array lengths
    with pytest.raises(ValueError, match="The lengths of the input arrays do not match."):
        OlsRegressor(np.array([1, 2]), np.array([1]))

    # Insufficient data points
    with pytest.raises(
        ValueError, match="At least two data points are required for regression"
    ):
        OlsRegressor(np.array([1]), np.array([1]))


class TestOlsRegressor:
    """Tests for the OlsRegressor class."""

    def test_regression(self, sample_data):
        """Test regression analysis with OlsRegressor.

        This function tests the regression analysis performed by the
        OlsRegressor, including parameter validation and prediction accuracy.
        """
        x, y = sample_data
        regressor = OlsRegressor(x, y)

        # Parameter validation
        params = regressor.get_params()
        assert isinstance(params, OlsRegressionParams)
        assert 1.9 < params.slope < 2.1  # Theoretical value is 2.0
        assert 0.8 < params.intercept < 1.2  # Theoretical value is 1.0
        assert 0.95 < params.r_value < 1.0
        assert params.p_value < 0.05
        assert params.stderr > 0

        # Prediction
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape

        # Check prediction accuracy (evaluated with R²)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95


class TestTlsRegressor:
    """Tests for the TlsRegressor class."""

    def test_regression(self, sample_data):
        """Test regression analysis with TlsRegressor.

        This function tests the regression analysis performed by the
        TlsRegressor, including parameter validation and prediction accuracy.
        """
        x, y = sample_data
        regressor = TlsRegressor(x, y)

        # Parameter validation
        params = regressor.get_params()
        assert isinstance(params, RegressionParams)
        assert 1.9 < params.slope < 2.1  # Theoretical value is 2.0
        assert 0.8 < params.intercept < 1.2  # Theoretical value is 1.0
        assert 0.95 < params.r_value < 1.0

        # Prediction
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape

        # Check prediction accuracy (evaluated with R²)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95


def test_compare_methods(sample_data):
    """Test comparison between OLS and TLS regression methods.

    This function tests the similarity of parameters obtained from
    both OlsRegressor and TlsRegressor to ensure they yield close
    results.
    """
    x, y = sample_data

    # Instantiate both models
    ols = OlsRegressor(x, y)
    tls = TlsRegressor(x, y)

    # Compare parameters (should be close)
    ols_params = ols.get_params()
    tls_params = tls.get_params()

    assert abs(ols_params.slope - tls_params.slope) < 0.1
    assert abs(ols_params.intercept - tls_params.intercept) < 0.1
    assert abs(ols_params.r_value - tls_params.r_value) < 0.1
