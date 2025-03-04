"""
regression_tools  - 高速Total Least Squares回帰

このパッケージはRustバックエンドを使った高速なTLS (直交) 回帰を提供します。
"""

# Rustモジュールから直接インポート
from .regression.regressor import (
    OlsRegressionParams,
    OlsRegressor,
    RegressionParams,
    TlsRegressor,
    create_regressor,
)
from .regression_tools import calculate_ols_regression, calculate_tls_regression

__all__ = [
    "OlsRegressionParams",
    "OlsRegressor",
    "RegressionParams",
    "TlsRegressor",
    "calculate_ols_regression",
    "calculate_tls_regression",
    "create_regressor",
]
