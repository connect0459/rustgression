"""
Bootstrap confidence and prediction interval tests for TlsRegressor.
"""

import numpy as np
import pytest

from rustgression import TlsRegressor


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)
    return x, y


class TestTlsRegressorConfidenceInterval:
    """Tests for TlsRegressor.confidence_interval()."""

    def test_returns_dict_with_slope_and_intercept_keys(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        ci = regressor.confidence_interval(n_bootstrap=200, random_state=0)
        assert set(ci.keys()) == {"slope", "intercept"}

    def test_each_value_is_a_two_tuple(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        ci = regressor.confidence_interval(n_bootstrap=200, random_state=0)
        assert len(ci["slope"]) == 2
        assert len(ci["intercept"]) == 2

    def test_lower_bound_is_below_upper_bound(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        ci = regressor.confidence_interval(n_bootstrap=200, random_state=0)
        assert ci["slope"][0] < ci["slope"][1]
        assert ci["intercept"][0] < ci["intercept"][1]

    def test_true_slope_is_within_interval_for_known_parameters(self):
        rng = np.random.default_rng(0)
        true_slope, true_intercept = 3.0, 2.0
        x = np.linspace(0, 10, 200)
        y = true_slope * x + true_intercept + rng.normal(0, 1.0, 200)
        regressor = TlsRegressor(x, y)
        ci = regressor.confidence_interval(alpha=0.05, n_bootstrap=1000, random_state=0)
        assert ci["slope"][0] < true_slope < ci["slope"][1]
        assert ci["intercept"][0] < true_intercept < ci["intercept"][1]

    def test_coverage_probability_approximates_nominal_level(self):
        rng = np.random.default_rng(1)
        true_slope, true_intercept = 2.0, 5.0
        n_trials = 50
        covered_slope = 0
        covered_intercept = 0
        for trial in range(n_trials):
            x = np.linspace(0, 10, 50)
            y = true_slope * x + true_intercept + rng.normal(0, 1.0, 50)
            regressor = TlsRegressor(x, y)
            ci = regressor.confidence_interval(
                alpha=0.05, n_bootstrap=300, random_state=trial
            )
            if ci["slope"][0] < true_slope < ci["slope"][1]:
                covered_slope += 1
            if ci["intercept"][0] < true_intercept < ci["intercept"][1]:
                covered_intercept += 1
        assert covered_slope / n_trials >= 0.82
        assert covered_intercept / n_trials >= 0.82

    def test_wider_interval_for_smaller_alpha(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        ci_95 = regressor.confidence_interval(
            alpha=0.05, n_bootstrap=500, random_state=0
        )
        ci_90 = regressor.confidence_interval(
            alpha=0.10, n_bootstrap=500, random_state=0
        )
        width_95 = ci_95["slope"][1] - ci_95["slope"][0]
        width_90 = ci_90["slope"][1] - ci_90["slope"][0]
        assert width_95 > width_90

    def test_raises_for_exactly_two_data_points(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = TlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.confidence_interval()

    def test_raises_for_alpha_at_or_outside_boundary(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.confidence_interval(alpha=0.0)
        with pytest.raises(ValueError):
            regressor.confidence_interval(alpha=1.0)

    def test_raises_for_n_bootstrap_less_than_one(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.confidence_interval(n_bootstrap=0)

    def test_results_are_consistent_given_identical_inputs(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        ci_a = regressor.confidence_interval(n_bootstrap=200, random_state=42)
        ci_b = regressor.confidence_interval(n_bootstrap=200, random_state=42)
        assert ci_a["slope"] == ci_b["slope"]
        assert ci_a["intercept"] == ci_b["intercept"]


class TestTlsRegressorPredictionInterval:
    """Tests for TlsRegressor.prediction_interval()."""

    def test_returns_array_of_shape_n_by_2(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.linspace(0, 10, 20)
        pi = regressor.prediction_interval(x_new, n_bootstrap=200, random_state=0)
        assert pi.shape == (20, 2)

    def test_lower_bound_is_below_upper_bound(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.linspace(0, 10, 20)
        pi = regressor.prediction_interval(x_new, n_bootstrap=200, random_state=0)
        assert np.all(pi[:, 0] < pi[:, 1])

    def test_predicted_value_is_within_interval(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.linspace(1, 9, 10)
        pi = regressor.prediction_interval(x_new, n_bootstrap=1000, random_state=0)
        y_hat = regressor.predict(x_new)
        assert np.all(pi[:, 0] < y_hat)
        assert np.all(y_hat < pi[:, 1])

    def test_wider_interval_for_smaller_alpha(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.array([5.0])
        pi_95 = regressor.prediction_interval(
            x_new, alpha=0.05, n_bootstrap=500, random_state=0
        )
        pi_90 = regressor.prediction_interval(
            x_new, alpha=0.10, n_bootstrap=500, random_state=0
        )
        width_95 = pi_95[0, 1] - pi_95[0, 0]
        width_90 = pi_90[0, 1] - pi_90[0, 0]
        assert width_95 > width_90

    def test_raises_for_exactly_two_data_points(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = TlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.prediction_interval(np.array([1.5]))

    def test_raises_for_alpha_at_or_outside_boundary(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.array([5.0])
        with pytest.raises(ValueError):
            regressor.prediction_interval(x_new, alpha=0.0)
        with pytest.raises(ValueError):
            regressor.prediction_interval(x_new, alpha=1.0)

    def test_raises_for_n_bootstrap_less_than_one(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.prediction_interval(np.array([5.0]), n_bootstrap=0)

    def test_accepts_2d_x_new_and_returns_shape_n_by_2(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new_row = np.array([[1.0, 5.0, 9.0]])
        pi = regressor.prediction_interval(x_new_row, n_bootstrap=200, random_state=0)
        assert pi.shape == (3, 2)

    def test_results_are_consistent_given_identical_inputs(self, linear_data):
        x, y = linear_data
        regressor = TlsRegressor(x, y)
        x_new = np.array([2.0, 5.0, 8.0])
        pi_a = regressor.prediction_interval(x_new, n_bootstrap=200, random_state=42)
        pi_b = regressor.prediction_interval(x_new, n_bootstrap=200, random_state=42)
        np.testing.assert_array_equal(pi_a, pi_b)
