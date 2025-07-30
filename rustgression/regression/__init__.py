"""
Regression analysis module for rustgression.

This module provides high-performance regression analysis tools implemented
in Rust and exposed through a Python interface.
"""

from .base import BaseRegressor, OlsRegressionParams, TlsRegressionParams
from .ols import OlsRegressor
from .tls import TlsRegressor
from .factory import create_regressor

__all__ = [
    "BaseRegressor",
    "OlsRegressionParams",
    "TlsRegressionParams",
    "OlsRegressor",
    "TlsRegressor",
    "create_regressor",
]
