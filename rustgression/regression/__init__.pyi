"""Type stubs for rustgression.regression module."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .base import BaseRegressor, OlsRegressionParams, TlsRegressionParams
from .ols import OlsRegressor
from .tls import TlsRegressor

def create_regressor(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    method: Literal["ols", "tls"] = "ols",
) -> OlsRegressor | TlsRegressor: ...

__all__ = [
    "BaseRegressor",
    "OlsRegressionParams",
    "OlsRegressor",
    "TlsRegressionParams",
    "TlsRegressor",
    "create_regressor",
]
