"""Type stubs for rustgression.regression.factory module."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .ols import OlsRegressor
from .tls import TlsRegressor

def create_regressor(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    method: Literal["ols", "tls"] = "ols",
) -> OlsRegressor | TlsRegressor: ...
