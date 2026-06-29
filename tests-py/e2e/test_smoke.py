import numpy as np
import pytest

import rustgression
from rustgression import NumericalWarning


def test_package_exposes_non_empty_version():
    assert rustgression.__version__


def test_ols_regressor_returns_finite_results_for_basic_input():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    regressor = rustgression.OlsRegressor(x, y)
    assert np.isfinite(regressor.slope())
    assert np.isfinite(regressor.intercept())
    assert np.isfinite(regressor.r_value())


def test_tls_regressor_returns_finite_results_even_when_condition_number_is_infinite():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    with pytest.warns(NumericalWarning):
        regressor = rustgression.TlsRegressor(x, y)
    assert np.isfinite(regressor.slope())
    assert np.isfinite(regressor.intercept())
    assert np.isfinite(regressor.r_value())
