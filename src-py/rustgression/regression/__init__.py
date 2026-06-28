"""
Regression analysis module for rustgression.

This module provides high-performance regression analysis tools implemented
in Rust and exposed through a Python interface.
"""

from .base import (
    BaseRegressor,
    OlsMultiRegressionParams,
    OlsRegressionParams,
    TlsRegressionParams,
)
from .factory import create_regressor
from .ols import OlsRegressor
from .ols_multi import OlsMultiRegressor
from .tls import TlsRegressor

__all__ = [
    "BaseRegressor",
    "OlsMultiRegressionParams",
    "OlsMultiRegressor",
    "OlsRegressionParams",
    "OlsRegressor",
    "TlsRegressionParams",
    "TlsRegressor",
    "create_regressor",
]
