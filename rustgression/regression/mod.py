"""
Python interface for regression analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import numpy as np

from ..rustgression import calculate_ols_regression, calculate_tls_regression

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

        # Initialize basic parameters
        self.slope: float
        self.intercept: float
        self.r_value: float

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
        return self.slope * x + self.intercept

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
            f"slope={self.slope:.6f}, "
            f"intercept={self.intercept:.6f}, "
            f"r_value={self.r_value:.6f})"
        )


class OlsRegressor(BaseRegressor[OlsRegressionParams]):
    """Class for calculating Ordinary Least Squares (OLS) regression.

    This class implements the standard least squares method, which minimizes
    the errors in the y-direction.

    Parameters
    ----------
    x : np.ndarray
        Input data for the independent variable (x-axis).
    y : np.ndarray
        Input data for the dependent variable (y-axis).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Initialize the OlsRegressor and fit the model.

        Parameters
        ----------
        x : np.ndarray
            Input data for the independent variable (x-axis).
        y : np.ndarray
            Input data for the dependent variable (y-axis).
        """
        self.p_value: float
        self.stderr: float
        self.intercept_stderr: float
        super().__init__(x, y)

    def _fit(self) -> None:
        """Perform OLS regression."""
        # Call the Rust implementation
        (
            _,
            self.slope,
            self.intercept,
            self.r_value,
            self.p_value,
            self.stderr,
            self.intercept_stderr,
        ) = calculate_ols_regression(self.x, self.y)

    def get_params(self) -> OlsRegressionParams:
        """Retrieve regression parameters.

        Returns
        -------
        OlsRegressionParams
            A data class containing all regression parameters, including
            slope, intercept, r_value, p_value, stderr, and intercept_stderr.
        """
        return OlsRegressionParams(
            slope=self.slope,
            intercept=self.intercept,
            r_value=self.r_value,
            p_value=self.p_value,
            stderr=self.stderr,
            intercept_stderr=self.intercept_stderr,
        )


class TlsRegressor(BaseRegressor[TlsRegressionParams]):
    """Class for calculating Total Least Squares (TLS) regression.

    Unlike Ordinary Least Squares (OLS), which minimizes errors only in the
    y-direction, TLS considers errors in both variables (x and y). This
    approach is more appropriate when measurement errors exist in both
    variables.

    """

    def _fit(self) -> None:
        """Perform TLS regression."""
        # Call the Rust implementation
        _, self.slope, self.intercept, self.r_value = calculate_tls_regression(
            self.x, self.y
        )

    def get_params(self) -> TlsRegressionParams:
        """Retrieve regression parameters.

        Returns
        -------
        TlsRegressionParams
            A data class containing the regression parameters.
        """
        return TlsRegressionParams(
            slope=self.slope, intercept=self.intercept, r_value=self.r_value
        )


def create_regressor(
    x: np.ndarray, y: np.ndarray, method: Literal["ols", "tls"] = "ols"
) -> BaseRegressor:
    """Factory function for creating a regression model.

    Parameters
    ----------
    x : np.ndarray
        Input data for the independent variable (x-axis).
    y : np.ndarray
        Input data for the dependent variable (y-axis).
    method : str
        The regression method to use ("ols" or "tls").

    Returns
    -------
    BaseRegressor
        An instance of the specified regression model.

    Raises
    ------
    ValueError
        If an unknown regression method is specified.
    """
    if method == "ols":
        return OlsRegressor(x, y)
    elif method == "tls":
        return TlsRegressor(x, y)
    else:
        raise ValueError(f"Unknown regression method: {method}")
