"""Type stubs for rustgression package."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .regression.base import OlsRegressionParams, TlsRegressionParams
from .regression.ols import OlsRegressor
from .regression.tls import TlsRegressor

__version__: str

def create_regressor(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    method: Literal["ols", "tls"] = "ols",
) -> OlsRegressor | TlsRegressor: ...

__all__ = [
    "OlsRegressionParams",
    "OlsRegressor",
    "TlsRegressionParams",
    "TlsRegressor",
    "create_regressor",
]
