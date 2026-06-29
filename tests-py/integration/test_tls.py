"""
Tests for Total Least Squares (TLS) regression.
"""

import numpy as np
import pytest

from rustgression import NumericalWarning, TlsRegressionParams, TlsRegressor


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


class TestTlsRegressor:
    """Tests for the TlsRegressor class."""

    def test_returns_accurate_slope_intercept_and_r_value_for_noisy_data(
        self, sample_data
    ):
        """Test normal regression analysis."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)

        # Test new property method API
        assert 1.9 < regressor.slope() < 2.1
        assert 0.8 < regressor.intercept() < 1.2
        assert 0.95 < regressor.r_value() < 1.0

        # Test legacy get_params() API (with deprecation warning)
        with pytest.warns(DeprecationWarning):
            params = regressor.get_params()
            assert isinstance(params, TlsRegressionParams)
            assert 1.9 < params.slope < 2.1

    def test_predicted_values_yield_r_squared_above_0_95(self, sample_data):
        """Test prediction functionality."""
        x, y = sample_data
        regressor = TlsRegressor(x, y)

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
            ([1.0, 2.0, 3.0], [5.0, 10.0, 15.0], 5.0, 0.0),
        ],
    )
    def test_correctly_fits_various_linear_patterns(
        self, x_data, y_data, expected_slope, expected_intercept
    ):
        """Test various regression patterns."""
        x = np.array(x_data)
        y = np.array(y_data)

        with pytest.warns(NumericalWarning):
            regressor = TlsRegressor(x, y)

        assert abs(regressor.slope() - expected_slope) < 1e-10, (
            f"Slope mismatch: expected {expected_slope}, got {regressor.slope()}"
        )
        assert abs(regressor.intercept() - expected_intercept) < 1e-10, (
            f"Intercept mismatch: expected {expected_intercept}, got {regressor.intercept()}"
        )

    def test_computes_exact_slope_and_intercept_for_two_data_points(self):
        """Test boundary conditions."""
        # Minimum data points: y = 3x, so TLS slope=3, intercept=0
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 6.0])
        with pytest.warns(NumericalWarning):
            regressor = TlsRegressor(x, y)
        assert abs(regressor.slope() - 3.0) < 1e-10
        assert abs(regressor.intercept() - 0.0) < 1e-10

    def test_r_squared_equals_squared_pearson_correlation(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        expected = np.corrcoef(x, y)[0, 1] ** 2
        assert abs(regressor.r_squared() - expected) < 1e-12

    def test_r_squared_is_within_unit_interval_for_noisy_data(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        assert 0.0 <= regressor.r_squared() <= 1.0

    def test_residuals_shape_matches_y(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        assert regressor.residuals().shape == y.shape


def _odrpack_linear_fit(xn, yn):
    """Fit linear ODR with equal weights; returns the odrpack result."""
    import odrpack

    def lin(x, b):
        return b[0] * x + b[1]

    return odrpack.odr_fit(lin, xn, yn, beta0=[1.0, 0.0])


class TestTlsRegressorStderr:
    """Tests that TLS standard errors match the Deming regression reference."""

    @pytest.mark.parametrize(
        "true_slope,noise",
        [
            (5.0, 1.0),
            (10.0, 2.0),
            (20.0, 3.0),
        ],
    )
    def test_stderr_matches_odrpack_deming_reference(self, true_slope, noise):
        """Slope stderr must agree with odrpack equal-weight ODR to within 1%."""
        rng = np.random.default_rng(seed=1)
        n = 60
        x = np.linspace(0, 10, n)
        xn = x + rng.normal(0, noise, n)
        yn = true_slope * x + rng.normal(0, noise, n)

        regressor = TlsRegressor(xn, yn)
        res = _odrpack_linear_fit(xn, yn)

        relative_error = abs(regressor.stderr() - res.sd_beta[0]) / res.sd_beta[0]
        assert relative_error < 0.01, (
            f"slope~{true_slope}: rustgression stderr={regressor.stderr():.4f}, "
            f"odrpack={res.sd_beta[0]:.4f}, relative_error={relative_error * 100:.1f}%"
        )

    def test_intercept_stderr_matches_odrpack_deming_reference(self):
        """Intercept stderr must agree with odrpack equal-weight ODR to within 1%."""
        rng = np.random.default_rng(seed=42)
        n = 60
        x = np.linspace(0, 10, n)
        noise = 2.0
        xn = x + rng.normal(0, noise, n)
        yn = 10.0 * x + rng.normal(0, noise, n)

        regressor = TlsRegressor(xn, yn)
        res = _odrpack_linear_fit(xn, yn)

        relative_error = (
            abs(regressor.intercept_stderr() - res.sd_beta[1]) / res.sd_beta[1]
        )
        assert relative_error < 0.01, (
            f"intercept_stderr: rustgression={regressor.intercept_stderr():.4f}, "
            f"odrpack={res.sd_beta[1]:.4f}, relative_error={relative_error * 100:.1f}%"
        )

    def test_stderr_exceeds_vertical_residual_approximation_when_x_has_measurement_noise(
        self,
    ):
        """TLS stderr must exceed vertical-residual-only approximation when both variables have noise."""
        rng = np.random.default_rng(seed=1)
        n = 60
        true_slope = 20.0
        noise = 3.0
        x = np.linspace(0, 10, n)
        xn = x + rng.normal(0, noise, n)
        yn = true_slope * x + rng.normal(0, noise, n)

        regressor = TlsRegressor(xn, yn)

        beta = regressor.slope()
        residuals = yn - beta * xn - regressor.intercept()
        ss_res = np.sum(residuals**2)
        ss_xx = np.sum((xn - np.mean(xn)) ** 2)
        s = np.sqrt(ss_res / (n - 2))
        vertical_residual_stderr = s / np.sqrt(ss_xx)

        ratio = regressor.stderr() / vertical_residual_stderr
        assert ratio > 1.10, (
            f"TLS stderr should be at least 10% larger than vertical-residual "
            f"approximation (got ratio {ratio:.3f})"
        )


class TestTlsRegressorIntervalsNotImplemented:
    """Tests that TLS interval methods raise NotImplementedError."""

    def test_confidence_interval_raises_not_implemented_error(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        with pytest.raises(NotImplementedError):
            regressor.confidence_interval()

    def test_prediction_interval_raises_not_implemented_error(self, sample_data):
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        x_new = np.linspace(0, 10, 5)
        with pytest.raises(NotImplementedError):
            regressor.prediction_interval(x_new)
