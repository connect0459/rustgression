"""
Base classes and data structures for regression analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class OlsRegressionParams:
    """Data class to store parameters for Ordinary Least Squares (OLS) regression.

    Attributes
    ----------
    slope : float
        The slope of the regression line.
    intercept : float
        The y-intercept of the regression line.
    r_value : float
        The correlation coefficient indicating the strength of the relationship.
    p_value : float
        The p-value associated with the regression slope.
    stderr : float
        The standard error of the regression slope.
    intercept_stderr : float
        The standard error of the intercept.
    """

    slope: float
    intercept: float
    r_value: float
    p_value: float
    stderr: float
    intercept_stderr: float


@dataclass
class TlsRegressionParams:
    """Data class to store parameters for Total Least Squares (TLS) regression.

    Attributes
    ----------
    slope : float
        The slope of the regression line.
    intercept : float
        The y-intercept of the regression line.
    r_value : float
        The correlation coefficient indicating the strength of the relationship.
    """

    slope: float
    intercept: float
    r_value: float


class BaseRegressor(ABC, Generic[T]):
    """Base class for regression analysis.

    This class defines a common interface for all regression implementations.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Initialize and fit the regression model.

        Parameters
        ----------
        x : np.ndarray
            The independent variable data (x-axis).
        y : np.ndarray
            The dependent variable data (y-axis).
        """
        # Validate and preprocess input data
        self.x = np.asarray(x, dtype=np.float64).flatten()
        self.y = np.asarray(y, dtype=np.float64).flatten()

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("The lengths of the input arrays do not match.")

        if self.x.shape[0] < 2:
            raise ValueError("At least two data points are required for regression.")

        # Initialize basic parameters (private attributes)
        self._slope: float
        self._intercept: float
        self._r_value: float

        # Execute fitting
        self._fit()

    @abstractmethod
    def _fit(self) -> None:
        """Abstract method to perform regression."""
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using the regression model.

        Parameters
        ----------
        x : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        x = np.asarray(x, dtype=np.float64)
        return self._slope * x + self._intercept

    @abstractmethod
    def get_params(self) -> T:
        """Retrieve regression parameters.

        Returns
        -------
        T
            A data class containing the regression parameters.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the regression model.

        Returns
        -------
        str
            A string representation of the regression model.
        """
        return (
            f"{self.__class__.__name__}("
            f"slope={self._slope:.6f}, "
            f"intercept={self._intercept:.6f}, "
            f"r_value={self._r_value:.6f})"
        )