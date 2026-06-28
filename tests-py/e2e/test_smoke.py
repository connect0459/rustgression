import numpy as np
import pytest

import rustgression


def test_package_exposes_non_empty_version():
    assert rustgression.__version__


@pytest.mark.parametrize(
    "regressor_cls", [rustgression.OlsRegressor, rustgression.TlsRegressor]
)
def test_regressor_returns_finite_results_for_basic_input(regressor_cls):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    regressor = regressor_cls(x, y)
    assert np.isfinite(regressor.slope())
    assert np.isfinite(regressor.intercept())
    assert np.isfinite(regressor.r_value())
