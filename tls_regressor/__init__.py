"""
tls_regressor - 高速Total Least Squares回帰

このパッケージはRustバックエンドを使った高速なTLS (直交) 回帰を提供します。
"""

from .tls import TLSRegressor, calculate_tls_regression

__all__ = ["TLSRegressor", "calculate_tls_regression"]
