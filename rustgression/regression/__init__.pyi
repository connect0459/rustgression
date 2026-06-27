"""Type stubs for rustgression.regression module."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .base import (
    BaseRegressor,
    OlsMultiRegressionParams,
    OlsRegressionParams,
    TlsRegressionParams,
)
from .ols import OlsRegressor
from .ols_multi import OlsMultiRegressor
from .tls import TlsRegressor

def create_regressor(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    method: Literal["ols", "tls", "ols_multi"] = "ols",
) -> OlsRegressor | TlsRegressor | OlsMultiRegressor: ...

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
