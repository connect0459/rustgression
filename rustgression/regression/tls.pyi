"""Type stubs for rustgression.regression.tls module."""

import numpy as np
from numpy.typing import NDArray

from .base import BaseRegressor, TlsRegressionParams

class TlsRegressor(BaseRegressor[TlsRegressionParams]):
    """Class for calculating Total Least Squares (TLS) regression."""
    
    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> None: ...
    
    def _fit(self) -> None: ...
    
    def slope(self) -> float: ...
    
    def intercept(self) -> float: ...
    
    def r_value(self) -> float: ...
    
    def get_params(self) -> TlsRegressionParams: ...