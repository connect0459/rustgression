"""
Tests for base functionality and factory functions.
"""

import numpy as np
import pytest

from rustgression import (
    OlsRegressor,
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
        with pytest.raises(
            ValueError, match="The lengths of the input arrays do not match."
        ):
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

        for _, x, y, expected_exception in test_cases:
            with pytest.raises(expected_exception):
                regressor_class(x, y)
