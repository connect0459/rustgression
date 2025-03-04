"""
tls_regressor - 高速Total Least Squares回帰

このパッケージはRustバックエンドを使った高速なTLS (直交) 回帰を提供します。
"""

# Rustモジュールから直接インポート
from .tls_regression import TlsRegressor
from .tls_regressor import calculate_tls_regression

# from .tls_regression import calculate_tls_regression, TlsRegressor

__all__ = ["TlsRegressor", "calculate_tls_regression"]
