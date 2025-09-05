"""
Tests for comparing OLS and TLS regression methods.
"""

import numpy as np
import pytest

from rustgression import OlsRegressor, TlsRegressor


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


class TestMethodComparison:
    """Tests for comparing OLS and TLS regression methods."""

    def test_parameter_similarity(self, sample_data):
        """Test parameter similarity between methods."""
        x, y = sample_data

        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)

        # Test with new property method API
        assert abs(ols.slope() - tls.slope()) < 0.1
        assert abs(ols.intercept() - tls.intercept()) < 0.1
        assert abs(ols.r_value() - tls.r_value()) < 0.1

    @pytest.mark.parametrize(
        "x_data,y_data",
        [
            ([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]),
            ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 5.0, 7.0, 9.0]),
        ],
    )
    def test_perfect_correlation_cases(self, x_data, y_data):
        """Test methods with perfect correlation data."""
        x = np.array(x_data)
        y = np.array(y_data)

        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)

        # For perfect correlation, correlation coefficient should be 1.0 or -1.0
        # but slope and intercept may differ due to TLS algorithm characteristics
        assert abs(abs(tls.r_value()) - 1.0) < 1e-10
        assert abs(abs(ols.r_value()) - 1.0) < 1e-10

        # Both methods should agree on the sign of correlation
        assert (ols.slope() > 0) == (tls.slope() > 0)
