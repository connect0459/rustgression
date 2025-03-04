"""
rustgression  - 高速Total Least Squares回帰

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
from .rustgression import calculate_ols_regression, calculate_tls_regression

__all__ = [
    "OlsRegressionParams",
    "OlsRegressor",
    "RegressionParams",
    "TlsRegressor",
    "calculate_ols_regression",
    "calculate_tls_regression",
    "create_regressor",
]

__version__ = "0.0.1"
