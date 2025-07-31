"""Type stubs for rustgression.regression.ols module."""

import numpy as np
from numpy.typing import NDArray

from .base import BaseRegressor, OlsRegressionParams

class OlsRegressor(BaseRegressor[OlsRegressionParams]):
    """Class for calculating Ordinary Least Squares (OLS) regression."""
    
    _p_value: float
    _stderr: float
    _intercept_stderr: float
    
    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> None: ...
    
    def _fit(self) -> None: ...
    
    def slope(self) -> float: ...
    
    def intercept(self) -> float: ...
    
    def r_value(self) -> float: ...
    
    def p_value(self) -> float: ...
    
    def stderr(self) -> float: ...
    
    def intercept_stderr(self) -> float: ...
    
    def get_params(self) -> OlsRegressionParams: ...