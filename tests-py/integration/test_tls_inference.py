"""
Inference value tests for TLS regression (p_value, stderr, intercept_stderr).
"""

import numpy as np
import pytest
from scipy import stats

from rustgression import TlsRegressor


@pytest.fixture
def significant_linear_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)
    return x, y


@pytest.fixture
def uncorrelated_data():
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = np.random.normal(0, 1.0, 100)
    return x, y


@pytest.fixture
def unit_slope_y_noise_data():
    np.random.seed(7)
    x = np.linspace(0, 10, 200)
    y = 1.0 * x + 2.0 + np.random.normal(0, 0.3, 200)
    return x, y


class TestTlsInferenceValues:
    """Tests that TLS exposes statistically sound inference values."""

    def test_provides_positive_stderr_for_well_conditioned_data(
        self, significant_linear_data
    ):
        x, y = significant_linear_data
        assert TlsRegressor(x, y).stderr() > 0

    def test_provides_p_value_in_valid_range_for_well_conditioned_data(
        self, significant_linear_data
    ):
        x, y = significant_linear_data
        p = TlsRegressor(x, y).p_value()
        assert 0.0 <= p <= 1.0

    def test_provides_positive_intercept_stderr_for_well_conditioned_data(
        self, significant_linear_data
    ):
        x, y = significant_linear_data
        assert TlsRegressor(x, y).intercept_stderr() > 0

    def test_produces_significant_p_value_for_clearly_correlated_data(
        self, significant_linear_data
    ):
        x, y = significant_linear_data
        assert TlsRegressor(x, y).p_value() < 0.05

    def test_produces_non_significant_p_value_for_uncorrelated_data(
        self, uncorrelated_data
    ):
        x, y = uncorrelated_data
        assert TlsRegressor(x, y).p_value() > 0.05

    def test_tls_stderr_matches_ols_when_only_y_has_noise(
        self, unit_slope_y_noise_data
    ):
        x, y = unit_slope_y_noise_data
        tls = TlsRegressor(x, y)
        ols = stats.linregress(x, y)
        assert abs(tls.stderr() - ols.stderr) / ols.stderr < 0.05

    def test_smaller_noise_yields_smaller_p_value(self):
        np.random.seed(1)
        x = np.linspace(0, 1, 20)
        low_noise = TlsRegressor(x, 0.5 * x + 1.0 + np.random.normal(0, 0.05, 20))
        high_noise = TlsRegressor(x, 0.5 * x + 1.0 + np.random.normal(0, 0.5, 20))
        assert low_noise.p_value() < high_noise.p_value()
