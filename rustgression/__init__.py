"""
rustgression
===========

A Python package that implements fast Total Least Squares (TLS) regression.

This package provides high-performance TLS (orthogonal) regression analysis using a backend implemented in Rust. It supports both ordinary least squares (OLS) regression and TLS regression.

Main Features
-------------
- Fast Rust backend
- Total Least Squares (TLS) regression
- Ordinary Least Squares (OLS) regression
- User-friendly Python interface

Classes
-------
- OlsRegressor
    Class for performing regression analysis using ordinary least squares.

- TlsRegressor
    Class for performing regression analysis using Total Least Squares.

Functions
---------
- create_regressor
    Factory function for creating a regression analyzer.

References
----------
Van Huffel, S., & Vandewalle, J. (1991). The Total Least Squares Problem:
Computational Aspects and Analysis. SIAM.

Examples
--------
>>> import rustgression
>>> regressor = rustgression.create_regressor()
>>> result = regressor.fit(X, y)
"""

# パッケージのバージョン
__version__ = "0.1.4"

# Rustモジュールをまず直接インポート
try:
    from .rustgression import calculate_ols_regression, calculate_tls_regression
except ImportError as e:
    import sys

    print(f"Error importing Rust module: {e}", file=sys.stderr)
    print("Rust extension was not properly compiled or installed.", file=sys.stderr)

# 次にPythonラッパーをインポート
try:
    from .regression import (
        OlsRegressionParams,
        OlsRegressor,
        TlsRegressionParams,
        TlsRegressor,
        create_regressor,
    )

    __all__ = [
        "OlsRegressionParams",
        "OlsRegressor",
        "TlsRegressionParams",
        "TlsRegressor",
        "create_regressor",
    ]
except ImportError as e:
    import sys

    print(f"Error importing regression module: {e}", file=sys.stderr)
