"""
回帰分析クラスのテストコード
"""

import numpy as np
import pytest

from rustgression import (
    OlsRegressionParams,
    OlsRegressor,
    RegressionParams,
    TlsRegressor,
    create_regressor,
)


@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを生成"""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    # y = 2x + 1 + ノイズ
    y = 2 * x + 1 + np.random.normal(0, 0.5, 100)
    return x, y


def test_create_regressor(sample_data):
    """ファクトリ関数のテスト"""
    x, y = sample_data
    
    # 正常系
    ols = create_regressor(x, y, "ols")
    assert isinstance(ols, OlsRegressor)
    
    tls = create_regressor(x, y, "tls")
    assert isinstance(tls, TlsRegressor)
    
    # デフォルト値
    default = create_regressor(x, y)
    assert isinstance(default, OlsRegressor)
    
    # 異常系
    with pytest.raises(ValueError, match="未知の回帰手法です"):
        create_regressor(x, y, "invalid")


def test_input_validation():
    """入力バリデーションのテスト"""
    # 配列長不一致
    with pytest.raises(ValueError, match="入力配列の長さが一致しません"):
        OlsRegressor(np.array([1, 2]), np.array([1]))
    
    # データ点不足
    with pytest.raises(ValueError, match="回帰には少なくとも2つのデータポイントが必要です"):
        OlsRegressor(np.array([1]), np.array([1]))


class TestOlsRegressor:
    """OLSRegressorのテスト"""
    
    def test_regression(self, sample_data):
        """回帰分析のテスト"""
        x, y = sample_data
        regressor = OlsRegressor(x, y)
        
        # パラメータの確認
        params = regressor.get_params()
        assert isinstance(params, OlsRegressionParams)
        assert 1.9 < params.slope < 2.1  # 理論値は2.0
        assert 0.8 < params.intercept < 1.2  # 理論値は1.0
        assert 0.95 < params.correlation < 1.0
        assert params.p_value < 0.05
        assert params.std_err > 0
        
        # 予測
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape
        
        # 予測値の精度確認（R²で評価）
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95


class TestTlsRegressor:
    """TLSRegressorのテスト"""
    
    def test_regression(self, sample_data):
        """回帰分析のテスト"""
        x, y = sample_data
        regressor = TlsRegressor(x, y)
        
        # パラメータの確認
        params = regressor.get_params()
        assert isinstance(params, RegressionParams)
        assert 1.9 < params.slope < 2.1  # 理論値は2.0
        assert 0.8 < params.intercept < 1.2  # 理論値は1.0
        assert 0.95 < params.correlation < 1.0
        
        # 予測
        y_pred = regressor.predict(x)
        assert y_pred.shape == y.shape
        
        # 予測値の精度確認（R²で評価）
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        assert r2 > 0.95


def test_compare_methods(sample_data):
    """OLSとTLSの比較テスト"""
    x, y = sample_data
    
    # 両方のモデルをインスタンス化
    ols = OlsRegressor(x, y)
    tls = TlsRegressor(x, y)
    
    # パラメータの比較（近い値になるはず）
    ols_params = ols.get_params()
    tls_params = tls.get_params()
    
    assert abs(ols_params.slope - tls_params.slope) < 0.1
    assert abs(ols_params.intercept - tls_params.intercept) < 0.1
    assert abs(ols_params.correlation - tls_params.correlation) < 0.1