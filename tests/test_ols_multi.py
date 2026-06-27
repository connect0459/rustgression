"""
Unit tests and sklearn double-validation for OlsMultiRegressor.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from rustgression import OlsMultiRegressionParams, OlsMultiRegressor, OlsRegressor


@pytest.fixture
def standard_noisy_data():
    """y = 2*x1 + 0.5*x2 + 1 + noise"""
    rng = np.random.default_rng(42)
    n = 100
    x_mat = rng.random((n, 2)) * 10
    y = 2.0 * x_mat[:, 0] + 0.5 * x_mat[:, 1] + 1.0 + rng.normal(0, 0.5, n)
    return x_mat, y


@pytest.fixture
def perfect_linear_data():
    """y = 3*x1 + 1.5*x2 + 2.5, zero residuals, linearly independent features"""
    x_mat = np.array(
        [[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 2.0], [5.0, 7.0], [6.0, 1.0]],
        dtype=float,
    )
    y = 3.0 * x_mat[:, 0] + 1.5 * x_mat[:, 1] + 2.5
    return x_mat, y


@pytest.fixture
def high_condition_number_data():
    """x1 in [10000, 10100], x2 in [0, 1]"""
    rng = np.random.default_rng(7)
    n = 200
    x1 = np.linspace(10000, 10100, n)
    x2 = rng.random(n)
    x_mat = np.column_stack([x1, x2])
    y = 2.5 * x1 + 0.3 * x2 - 24900.0 + rng.normal(0, 1.0, n)
    return x_mat, y


class TestOlsMultiRegressor:
    """Unit tests for OlsMultiRegressor."""

    def test_returns_coefficients_close_to_known_values_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        model = OlsMultiRegressor(x_mat, y)
        coefs = model.coefficients()

        assert abs(coefs[0] - 2.5) < 1e-8
        assert abs(coefs[1] - 3.0) < 1e-8
        assert abs(coefs[2] - 1.5) < 1e-8

    def test_returns_r_squared_of_one_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.r_squared() - 1.0) < 1e-10

    def test_returns_coefficients_close_to_known_values_for_standard_noisy_data(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        model = OlsMultiRegressor(x_mat, y)
        coefs = model.coefficients()

        assert abs(coefs[1] - 2.0) < 0.15
        assert abs(coefs[2] - 0.5) < 0.15
        assert abs(coefs[0] - 1.0) < 0.5

    def test_predict_returns_exact_values_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        model = OlsMultiRegressor(x_mat, y)
        y_pred = model.predict(x_mat)

        np.testing.assert_array_almost_equal(y_pred, y, decimal=8)

    def test_predict_accepts_1d_input_as_single_sample(self, perfect_linear_data):
        x_mat, y = perfect_linear_data
        model = OlsMultiRegressor(x_mat, y)
        single_sample = x_mat[0]
        result = model.predict(single_sample)

        assert result.shape == (1,)
        assert abs(result[0] - y[0]) < 1e-8

    def test_matches_ols_regressor_output_when_given_single_predictor(self):
        rng = np.random.default_rng(42)
        x = rng.random(50) * 10
        y = 3.0 * x + 2.0 + rng.normal(0, 0.5, 50)
        x_mat = x.reshape(-1, 1)

        multi = OlsMultiRegressor(x_mat, y)
        single = OlsRegressor(x, y)

        assert abs(multi.intercept() - single.intercept()) < 1e-10
        assert abs(multi.coefficients()[1] - single.slope()) < 1e-10

    def test_raises_value_error_for_collinear_predictors(self):
        x_mat = np.column_stack(
            [
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
            ]
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(ValueError):
            OlsMultiRegressor(x_mat, y)

    def test_raises_value_error_when_observations_do_not_exceed_predictors(self):
        x_mat = np.random.randn(3, 4)
        y = np.random.randn(3)

        with pytest.raises(ValueError):
            OlsMultiRegressor(x_mat, y)

    def test_raises_value_error_when_n_equals_p(self):
        x_mat = np.random.randn(3, 3)
        y = np.random.randn(3)

        with pytest.raises(ValueError):
            OlsMultiRegressor(x_mat, y)

    def test_raises_value_error_for_1d_x_input(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            OlsMultiRegressor(x, y)

    def test_raises_value_error_when_feature_count_mismatches_at_predict_time(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        model = OlsMultiRegressor(x_mat, y)
        x_wrong = np.random.randn(10, 3)

        with pytest.raises(ValueError):
            model.predict(x_wrong)

    def test_get_params_returns_ols_multi_regression_params_instance(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        model = OlsMultiRegressor(x_mat, y)

        with pytest.warns(DeprecationWarning):
            params = model.get_params()

        assert isinstance(params, OlsMultiRegressionParams)

    def test_coefficients_returns_defensive_copy(self, standard_noisy_data):
        x_mat, y = standard_noisy_data
        model = OlsMultiRegressor(x_mat, y)
        coefs_before = model.coefficients().copy()
        model.coefficients()[0] = 9999.0
        coefs_after = model.coefficients()

        np.testing.assert_array_equal(coefs_before, coefs_after)

    def test_repr_includes_class_name_intercept_and_r_squared(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        model = OlsMultiRegressor(x_mat, y)
        representation = repr(model)

        assert "OlsMultiRegressor" in representation
        assert "intercept=" in representation
        assert "r_squared=" in representation


class TestOlsMultiDoubleValidation:
    """Double-validation tests comparing OlsMultiRegressor against sklearn."""

    def test_coefficients_match_sklearn_for_standard_noisy_data(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.intercept() - ref.intercept_) < 1e-8
        assert abs(model.coefficients()[1] - ref.coef_[0]) < 1e-8
        assert abs(model.coefficients()[2] - ref.coef_[1]) < 1e-8

    def test_r_squared_matches_sklearn_for_standard_noisy_data(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.r_squared() - ref.score(x_mat, y)) < 1e-10

    def test_predictions_match_sklearn_for_standard_noisy_data(
        self, standard_noisy_data
    ):
        x_mat, y = standard_noisy_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        np.testing.assert_array_almost_equal(
            model.predict(x_mat), ref.predict(x_mat), decimal=8
        )

    def test_coefficients_match_sklearn_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.intercept() - ref.intercept_) < 1e-8
        assert abs(model.coefficients()[1] - ref.coef_[0]) < 1e-8
        assert abs(model.coefficients()[2] - ref.coef_[1]) < 1e-8

    def test_r_squared_matches_sklearn_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.r_squared() - ref.score(x_mat, y)) < 1e-10

    def test_predictions_match_sklearn_for_perfect_linear_data(
        self, perfect_linear_data
    ):
        x_mat, y = perfect_linear_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        np.testing.assert_array_almost_equal(
            model.predict(x_mat), ref.predict(x_mat), decimal=8
        )

    def test_coefficients_match_sklearn_for_high_condition_number_data(
        self, high_condition_number_data
    ):
        x_mat, y = high_condition_number_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.intercept() - ref.intercept_) < 1e-4
        assert abs(model.coefficients()[1] - ref.coef_[0]) < 1e-6
        assert abs(model.coefficients()[2] - ref.coef_[1]) < 1e-4

    def test_r_squared_matches_sklearn_for_high_condition_number_data(
        self, high_condition_number_data
    ):
        x_mat, y = high_condition_number_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        assert abs(model.r_squared() - ref.score(x_mat, y)) < 1e-6

    def test_predictions_match_sklearn_for_high_condition_number_data(
        self, high_condition_number_data
    ):
        x_mat, y = high_condition_number_data
        ref = LinearRegression().fit(x_mat, y)
        model = OlsMultiRegressor(x_mat, y)

        np.testing.assert_array_almost_equal(
            model.predict(x_mat), ref.predict(x_mat), decimal=4
        )
