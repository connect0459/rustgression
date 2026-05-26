"""
Double-Validation tests for OLS regression against scipy.stats.linregress.
"""

import numpy as np
import pytest
from scipy import stats

from rustgression import OlsRegressor


@pytest.fixture
def standard_noisy_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)
    return x, y


@pytest.fixture
def perfect_linear_data():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = 3.0 * x + 2.5
    return x, y


@pytest.fixture
def negatively_correlated_data():
    np.random.seed(99)
    x = np.linspace(0, 10, 100)
    y = -1.5 * x + 8.0 + np.random.normal(0, 0.3, 100)
    return x, y


@pytest.fixture
def high_condition_number_data():
    np.random.seed(7)
    x = np.linspace(10000, 10100, 200)
    y = 2.5 * x - 24900.0 + np.random.normal(0, 1.0, 200)
    return x, y


class TestOlsDoubleValidation:
    """Double-Validation tests comparing OLS outputs against scipy.stats.linregress."""

    def test_ols_produces_accurate_slope_intercept_and_r_value_for_standard_noisy_data(
        self, standard_noisy_data
    ):
        x, y = standard_noisy_data
        ref_slope, ref_intercept, ref_r_value, _, _ = stats.linregress(x, y)
        ols = OlsRegressor(x, y)

        assert abs(ols.slope() - ref_slope) < 1e-10
        assert abs(ols.intercept() - ref_intercept) < 1e-10
        assert abs(ols.r_value() - ref_r_value) < 1e-10

    def test_ols_produces_exact_parameters_for_perfectly_linear_data(
        self, perfect_linear_data
    ):
        x, y = perfect_linear_data
        ref_slope, ref_intercept, ref_r_value, _, _ = stats.linregress(x, y)
        ols = OlsRegressor(x, y)

        assert abs(ols.slope() - ref_slope) < 1e-10
        assert abs(ols.intercept() - ref_intercept) < 1e-10
        assert abs(ols.r_value() - ref_r_value) < 1e-10

    def test_ols_produces_accurate_slope_intercept_and_r_value_for_negatively_correlated_data(
        self, negatively_correlated_data
    ):
        x, y = negatively_correlated_data
        ref_slope, ref_intercept, ref_r_value, _, _ = stats.linregress(x, y)
        ols = OlsRegressor(x, y)

        assert abs(ols.slope() - ref_slope) < 1e-10
        assert abs(ols.intercept() - ref_intercept) < 1e-10
        assert abs(ols.r_value() - ref_r_value) < 1e-10

    def test_ols_maintains_numerical_accuracy_under_high_condition_number(
        self, high_condition_number_data
    ):
        x, y = high_condition_number_data
        ref_slope, ref_intercept, ref_r_value, _, _ = stats.linregress(x, y)
        ols = OlsRegressor(x, y)

        assert abs(ols.slope() - ref_slope) < 1e-6
        assert abs(ols.intercept() - ref_intercept) < 1e-6
        assert abs(ols.r_value() - ref_r_value) < 1e-6
