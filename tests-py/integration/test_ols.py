"""
Tests for Ordinary Least Squares (OLS) regression.
"""

import numpy as np
import pytest
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from rustgression import OlsRegressionParams, OlsRegressor


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


class TestOlsRegressor:
    """Tests for the OlsRegressor class."""

    def test_returns_accurate_slope_intercept_r_value_p_value_and_stderr_for_noisy_data(
        self, sample_data
    ):
        """Test normal regression analysis."""
        x, y = sample_data
        regressor = OlsRegressor(x, y)

        # Test new property method API
        assert 1.9 < regressor.slope() < 2.1
        assert 0.8 < regressor.intercept() < 1.2
        assert 0.95 < regressor.r_value() < 1.0
        assert regressor.p_value() < 0.05
        assert regressor.stderr() > 0
        assert regressor.intercept_stderr() > 0

        # Test legacy get_params() API (with deprecation warning)
        with pytest.warns(DeprecationWarning):
            params = regressor.get_params()
            assert isinstance(params, OlsRegressionParams)
            assert 1.9 < params.slope < 2.1

    def test_predicted_values_yield_r_squared_above_0_95(self, sample_data):
        """Test prediction functionality."""
        x, y = sample_data
        regressor = OlsRegressor(x, y)

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
            ([1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0, 3.0], 0.0, 3.0),
        ],
    )
    def test_correctly_fits_various_linear_patterns(
        self, x_data, y_data, expected_slope, expected_intercept
    ):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)

        regressor = OlsRegressor(x, y)

        # Test with new property method API
        assert abs(regressor.slope() - expected_slope) < 0.1, (
            f"Slope mismatch: expected {expected_slope}, got {regressor.slope()}"
        )
        assert abs(regressor.intercept() - expected_intercept) < 0.1, (
            f"Intercept mismatch: expected {expected_intercept}, got {regressor.intercept()}"
        )

    def test_computes_exact_slope_and_zero_intercept_for_two_data_points(self):
        """Test boundary conditions."""
        # Minimum data points
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = OlsRegressor(x, y)
        assert abs(regressor.slope() - 2.0) < 1e-10
        assert abs(regressor.intercept() - 0.0) < 1e-10

    def test_r_squared_is_within_unit_interval_for_noisy_data(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        assert 0.0 <= regressor.r_squared() <= 1.0

    def test_r_squared_equals_proportion_of_variance_explained(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        y_pred = regressor.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        expected = 1.0 - ss_res / ss_tot
        assert abs(regressor.r_squared() - expected) < 1e-10

    def test_r_squared_matches_scipy_linregress(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        scipy_result = stats.linregress(x, y)
        expected = scipy_result.rvalue**2
        assert abs(regressor.r_squared() - expected) < 1e-10

    def test_residuals_shape_matches_y(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        assert regressor.residuals().shape == y.shape

    def test_residuals_sum_to_zero(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        assert abs(regressor.residuals().sum()) < 1e-8

    def test_residuals_match_scipy_linregress(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        scipy_result = stats.linregress(x, y)
        expected_residuals = y - (scipy_result.slope * x + scipy_result.intercept)
        np.testing.assert_allclose(
            regressor.residuals(), expected_residuals, atol=1e-10
        )


class TestOlsRegressorConfidenceInterval:
    """Tests for OlsRegressor.confidence_interval()."""

    def test_returns_dict_with_slope_and_intercept_keys(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        ci = regressor.confidence_interval()
        assert set(ci.keys()) == {"slope", "intercept"}

    def test_each_value_is_a_two_tuple(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        ci = regressor.confidence_interval()
        assert len(ci["slope"]) == 2
        assert len(ci["intercept"]) == 2

    def test_lower_bound_is_below_upper_bound(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        ci = regressor.confidence_interval()
        assert ci["slope"][0] < ci["slope"][1]
        assert ci["intercept"][0] < ci["intercept"][1]

    def test_true_slope_is_within_interval_for_known_parameters(self):
        rng = np.random.default_rng(0)
        true_slope, true_intercept = 3.0, 2.0
        x = np.linspace(0, 10, 200)
        y = true_slope * x + true_intercept + rng.normal(0, 1.0, 200)
        regressor = OlsRegressor(x, y)
        ci = regressor.confidence_interval(alpha=0.05)
        assert ci["slope"][0] < true_slope < ci["slope"][1]
        assert ci["intercept"][0] < true_intercept < ci["intercept"][1]

    def test_coverage_probability_approximates_nominal_level(self):
        rng = np.random.default_rng(1)
        true_slope, true_intercept = 2.0, 5.0
        n_trials = 100
        covered_slope = 0
        covered_intercept = 0
        for _ in range(n_trials):
            x = np.linspace(0, 10, 50)
            y = true_slope * x + true_intercept + rng.normal(0, 1.0, 50)
            regressor = OlsRegressor(x, y)
            ci = regressor.confidence_interval(alpha=0.05)
            if ci["slope"][0] < true_slope < ci["slope"][1]:
                covered_slope += 1
            if ci["intercept"][0] < true_intercept < ci["intercept"][1]:
                covered_intercept += 1
        assert covered_slope / n_trials >= 0.90
        assert covered_intercept / n_trials >= 0.90

    def test_matches_statsmodels_conf_int(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        ci = regressor.confidence_interval(alpha=0.05)

        sm_model = OLS(y, add_constant(x)).fit()
        sm_ci = sm_model.conf_int(alpha=0.05)
        expected_intercept = (sm_ci[0, 0], sm_ci[0, 1])
        expected_slope = (sm_ci[1, 0], sm_ci[1, 1])

        np.testing.assert_allclose(ci["slope"], expected_slope, rtol=1e-6)
        np.testing.assert_allclose(ci["intercept"], expected_intercept, rtol=1e-6)

    def test_wider_interval_for_smaller_alpha(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        ci_95 = regressor.confidence_interval(alpha=0.05)
        ci_90 = regressor.confidence_interval(alpha=0.10)
        width_95 = ci_95["slope"][1] - ci_95["slope"][0]
        width_90 = ci_90["slope"][1] - ci_90["slope"][0]
        assert width_95 > width_90

    def test_raises_for_exactly_two_data_points(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = OlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.confidence_interval()


class TestOlsRegressorPredictionInterval:
    """Tests for OlsRegressor.prediction_interval()."""

    def test_returns_array_of_shape_n_by_2(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new = np.linspace(0, 10, 20)
        pi = regressor.prediction_interval(x_new)
        assert pi.shape == (20, 2)

    def test_lower_bound_is_below_upper_bound(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new = np.linspace(0, 10, 20)
        pi = regressor.prediction_interval(x_new)
        assert np.all(pi[:, 0] < pi[:, 1])

    def test_predicted_value_is_within_interval(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new = np.linspace(0, 10, 20)
        pi = regressor.prediction_interval(x_new)
        y_hat = regressor.predict(x_new)
        assert np.all(pi[:, 0] < y_hat)
        assert np.all(y_hat < pi[:, 1])

    def test_matches_statsmodels_get_prediction(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new = np.array([1.0, 5.0, 9.0])
        pi = regressor.prediction_interval(x_new, alpha=0.05)

        sm_model = OLS(y, add_constant(x)).fit()
        sm_pred = sm_model.get_prediction(add_constant(x_new))
        sm_pi = sm_pred.summary_frame(alpha=0.05)[
            ["obs_ci_lower", "obs_ci_upper"]
        ].values

        np.testing.assert_allclose(pi[:, 0], sm_pi[:, 0], rtol=1e-6)
        np.testing.assert_allclose(pi[:, 1], sm_pi[:, 1], rtol=1e-6)

    def test_wider_interval_for_smaller_alpha(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new = np.array([5.0])
        pi_95 = regressor.prediction_interval(x_new, alpha=0.05)
        pi_90 = regressor.prediction_interval(x_new, alpha=0.10)
        width_95 = pi_95[0, 1] - pi_95[0, 0]
        width_90 = pi_90[0, 1] - pi_90[0, 0]
        assert width_95 > width_90

    def test_intervals_are_wider_at_extremes_than_at_mean(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_mean = np.array([x.mean()])
        x_extreme = np.array([x.max()])
        pi_mean = regressor.prediction_interval(x_mean)
        pi_extreme = regressor.prediction_interval(x_extreme)
        width_mean = pi_mean[0, 1] - pi_mean[0, 0]
        width_extreme = pi_extreme[0, 1] - pi_extreme[0, 0]
        assert width_extreme > width_mean

    def test_raises_for_exactly_two_data_points(self):
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        regressor = OlsRegressor(x, y)
        with pytest.raises(ValueError):
            regressor.prediction_interval(np.array([1.5]))

    def test_raises_for_constant_x_values(self):
        x = np.array([3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            regressor = OlsRegressor(x, y)
            regressor.prediction_interval(np.array([3.0]))

    def test_accepts_2d_x_new_and_returns_shape_n_by_2(self, sample_data):
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        x_new_row = np.array([[1.0, 5.0, 9.0]])
        pi = regressor.prediction_interval(x_new_row)
        assert pi.shape == (3, 2)
