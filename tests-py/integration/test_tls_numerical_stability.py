"""
Numerical stability tests for TLS regression.
"""

import numpy as np
import pytest

from rustgression import TlsRegressor


class TestTlsNumericalStability:
    """Tests that verify TLS regression handles numerically unstable inputs safely."""

    def test_raises_error_when_tls_problem_is_numerically_unstable_due_to_extreme_slope(
        self,
    ):
        np.random.seed(42)
        x = np.linspace(0, 1e-10, 50) + np.random.normal(0, 1e-12, 50)
        y = np.linspace(0, 1.0, 50)
        with pytest.raises(RuntimeError, match="numerically unstable"):
            TlsRegressor(x, y)

    def test_produces_accurate_slope_for_data_with_moderate_noise(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)
        regressor = TlsRegressor(x, y)
        assert 1.9 < regressor.slope() < 2.1

    def test_produces_accurate_slope_for_strongly_correlated_data(self):
        np.random.seed(0)
        x = np.linspace(0, 10, 50)
        y = 3.0 * x + 5.0 + np.random.normal(0, 0.01, 50)
        regressor = TlsRegressor(x, y)
        assert abs(regressor.slope() - 3.0) < 0.01
