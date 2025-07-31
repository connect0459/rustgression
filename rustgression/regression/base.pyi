"""Type stubs for rustgression.regression.base module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

@dataclass
class OlsRegressionParams:
    """Data class to store parameters for Ordinary Least Squares (OLS) regression."""
    slope: float
    intercept: float
    r_value: float
    p_value: float
    stderr: float
    intercept_stderr: float

@dataclass
class TlsRegressionParams:
    """Data class to store parameters for Total Least Squares (TLS) regression."""
    slope: float
    intercept: float
    r_value: float

class BaseRegressor(ABC, Generic[T]):
    """Base class for regression analysis."""
    
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    _slope: float
    _intercept: float
    _r_value: float
    
    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> None: ...
    
    @abstractmethod
    def _fit(self) -> None: ...
    
    def predict(self, x: NDArray[np.floating]) -> NDArray[np.floating]: ...
    
    @abstractmethod
    def get_params(self) -> T: ...
    
    def __repr__(self) -> str: ...