import numpy as np
import pytest
from python import TlsRegressor, calculate_tls_regression


def test_tls_regressor_basic():
    # テストデータの生成
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.0
    true_intercept = 1.0
    y = true_slope * x + true_intercept + np.random.normal(0, 0.5, 100)

    # TlsRegressorのテスト
    model = TlsRegressor()
    model.fit(x, y)

    # パラメータの取得
    slope, intercept, correlation = model.get_params()

    # 予測値の計算
    y_pred = model.predict(x)

    # アサーション
    assert (
        abs(slope - true_slope) < 0.1
    ), f"Slope {slope} is too far from true value {true_slope}"
    assert (
        abs(intercept - true_intercept) < 0.1
    ), f"Intercept {intercept} is too far from true value {true_intercept}"
    assert correlation > 0.95, f"Correlation {correlation} is too low"
    assert y_pred.shape == y.shape, "Prediction shape mismatch"


def test_calculate_tls_regression_direct():
    # テストデータの生成
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.0
    true_intercept = 1.0
    y = true_slope * x + true_intercept + np.random.normal(0, 0.5, 100)

    # Rust関数の直接テスト
    y_pred, slope, intercept, correlation = calculate_tls_regression(x, y)

    # アサーション
    assert (
        abs(slope - true_slope) < 0.1
    ), f"Slope {slope} is too far from true value {true_slope}"
    assert (
        abs(intercept - true_intercept) < 0.1
    ), f"Intercept {intercept} is too far from true value {true_intercept}"
    assert correlation > 0.95, f"Correlation {correlation} is too low"
    assert y_pred.shape == y.shape, "Prediction shape mismatch"


def test_input_validation():
    model = TlsRegressor()

    # 異なる長さの入力配列
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]), np.array([1, 2]))

    # 1点のみのデータ
    with pytest.raises(ValueError):
        model.fit(np.array([1]), np.array([1]))

    # フィット前の予測
    with pytest.raises(RuntimeError):
        model.predict(np.array([1, 2, 3]))
