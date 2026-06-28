"""
Double-Validation tests for TLS regression against odrpack.

For linear ODR with equal error weights, odrpack and TLS (SVD-based) yield
the same mathematical solution. Comparing their outputs validates numerical
correctness of the rustgression TLS implementation.
"""

import numpy as np
import pytest
from odrpack import odr_fit

from rustgression import TlsRegressor


def _linear_model(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


@pytest.fixture
def bivariate_noisy_data():
    rng = np.random.default_rng(42)
    n = 200
    x_true = np.linspace(0, 10, n)
    y_true = 2.0 * x_true + 1.0
    x = x_true + rng.normal(0, 0.3, n)
    y = y_true + rng.normal(0, 0.3, n)
    return x, y


@pytest.fixture
def strong_signal_data():
    rng = np.random.default_rng(0)
    n = 500
    x_true = np.linspace(0, 10, n)
    y_true = 3.0 * x_true + 5.0
    x = x_true + rng.normal(0, 0.05, n)
    y = y_true + rng.normal(0, 0.05, n)
    return x, y


@pytest.fixture
def negative_slope_data():
    rng = np.random.default_rng(99)
    n = 300
    x_true = np.linspace(0, 10, n)
    y_true = -1.5 * x_true + 8.0
    x = x_true + rng.normal(0, 0.2, n)
    y = y_true + rng.normal(0, 0.2, n)
    return x, y


class TestTlsDoubleValidation:
    """Double-Validation tests comparing TLS outputs against odrpack."""

    def test_slope_and_intercept_match_odrpack_for_bivariate_noisy_data(
        self, bivariate_noisy_data
    ):
        x, y = bivariate_noisy_data
        sol = odr_fit(_linear_model, x, y, beta0=[1.0, 2.0])
        tls = TlsRegressor(x, y)

        assert abs(tls.slope() - sol.beta[1]) < 1e-6
        assert abs(tls.intercept() - sol.beta[0]) < 1e-6

    def test_slope_and_intercept_match_odrpack_for_strong_signal_data(
        self, strong_signal_data
    ):
        x, y = strong_signal_data
        sol = odr_fit(_linear_model, x, y, beta0=[5.0, 3.0])
        tls = TlsRegressor(x, y)

        assert abs(tls.slope() - sol.beta[1]) < 1e-6
        assert abs(tls.intercept() - sol.beta[0]) < 1e-6

    def test_slope_and_intercept_match_odrpack_for_negative_slope_data(
        self, negative_slope_data
    ):
        x, y = negative_slope_data
        sol = odr_fit(_linear_model, x, y, beta0=[8.0, -1.5])
        tls = TlsRegressor(x, y)

        assert abs(tls.slope() - sol.beta[1]) < 1e-6
        assert abs(tls.intercept() - sol.beta[0]) < 1e-6

    def test_slope_and_intercept_match_odrpack_for_high_noise_data(self):
        rng = np.random.default_rng(7)
        n = 100
        x_true = np.linspace(0, 10, n)
        y_true = 2.0 * x_true + 3.0
        x = x_true + rng.normal(0, 1.5, n)
        y = y_true + rng.normal(0, 1.5, n)

        sol = odr_fit(_linear_model, x, y, beta0=[3.0, 2.0])
        tls = TlsRegressor(x, y)

        # High noise reduces odrpack's iterative solver precision vs TLS closed-form.
        # 1e-4 is sufficient to confirm both methods converge to the same solution.
        assert abs(tls.slope() - sol.beta[1]) < 1e-4
        assert abs(tls.intercept() - sol.beta[0]) < 1e-4
