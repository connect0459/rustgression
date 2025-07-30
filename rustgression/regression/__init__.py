"""
Regression analysis module for rustgression.

This module provides high-performance regression analysis tools implemented
in Rust and exposed through a Python interface.
"""

from .base import BaseRegressor, OlsRegressionParams, TlsRegressionParams
from .factory import create_regressor
from .ols import OlsRegressor
from .tls import TlsRegressor

__all__ = [
    "BaseRegressor",
    "OlsRegressionParams",
    "OlsRegressor",
    "TlsRegressionParams",
    "TlsRegressor",
    "create_regressor",
]
