"""Type stubs for rustgression package."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .regression.base import (
    OlsMultiRegressionParams,
    OlsRegressionParams,
    TlsRegressionParams,
)
from .regression.ols import OlsRegressor
from .regression.ols_multi import OlsMultiRegressor
from .regression.tls import TlsRegressor

__version__: str

class NumericalWarning(UserWarning): ...

def create_regressor(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    method: Literal["ols", "tls", "ols_multi"] = "ols",
) -> OlsRegressor | TlsRegressor | OlsMultiRegressor: ...

__all__ = [
    "NumericalWarning",
    "OlsMultiRegressionParams",
    "OlsMultiRegressor",
    "OlsRegressionParams",
    "OlsRegressor",
    "TlsRegressionParams",
    "TlsRegressor",
    "create_regressor",
]
