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

# Directly import from Rust module
from .regression.mod import (
    OlsRegressionParams,
    OlsRegressor,
    RegressionParams,
    TlsRegressor,
    create_regressor,
)

__all__ = [
    "OlsRegressionParams",
    "OlsRegressor",
    "RegressionParams",
    "TlsRegressor",
    "create_regressor",
]

__version__ = "0.1.0"
