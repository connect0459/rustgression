"""
Multiple Ordinary Least Squares (OLS) regression implementation.
"""

import warnings

import numpy as np

from ._rust_imports import calculate_ols_multi_regression
from .base import OlsMultiRegressionParams


class OlsMultiRegressor:
    """Class for multiple Ordinary Least Squares (OLS) regression.

    Fits a model of the form y = b0 + b1*x1 + ... + bp*xp using the ordinary
    least squares method, which minimizes the sum of squared residuals in y.

    Parameters
    ----------
    x : np.ndarray, shape (n, p)
        Predictor matrix with n observations and p features.
    y : np.ndarray, shape (n,)
        Response vector.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """Initialize the OlsMultiRegressor and fit the model.

        Parameters
        ----------
        x : np.ndarray, shape (n, p)
            Predictor matrix with n observations and p features.
        y : np.ndarray, shape (n,)
            Response vector.

        Raises
        ------
        ValueError
            If x is not 2D, arrays have mismatched lengths, fewer than 2
            observations are provided, or the number of observations does not
            exceed the number of predictors (n <= p).
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).flatten()

        if self.x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n, p).")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("The number of rows in x must match the length of y.")

        if self.x.shape[0] < 2:
            raise ValueError("At least two data points are required for regression.")

        n, p = self.x.shape
        if n <= p:
            raise ValueError(
                f"Number of observations ({n}) must exceed number of predictors ({p})."
            )

        self._coefficients: np.ndarray
        self._r_squared: float
        self._f_statistic: float
        self._p_value: float

        self._fit()

    def _fit(self) -> None:
        """Perform multiple OLS regression via the Rust backend."""
        _, coefficients, r_squared, f_statistic, p_value = (
            calculate_ols_multi_regression(self.x, self.y)
        )
        self._coefficients = np.asarray(coefficients, dtype=np.float64)
        self._r_squared = float(r_squared)
        self._f_statistic = float(f_statistic)
        self._p_value = float(p_value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict response values for a predictor matrix.

        Parameters
        ----------
        x : np.ndarray, shape (p,) or (m, p)
            Predictor input. A 1D array of shape (p,) is treated as a single
            sample and returns a result of shape (1,). A 2D array of shape
            (m, p) returns results of shape (m,).

        Returns
        -------
        np.ndarray, shape (m,)
            Predicted response values.

        Raises
        ------
        ValueError
            If x cannot be interpreted as a 1D or 2D array, or if the number
            of features does not match the number used during fitting.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(
                "x must be a 1D array of shape (p,) or 2D array of shape (m, p)."
            )
        p_fit = len(self._coefficients) - 1
        if x.shape[1] != p_fit:
            raise ValueError(f"Expected {p_fit} features, got {x.shape[1]}.")
        intercept = self._coefficients[0]
        slopes = self._coefficients[1:]
        return x @ slopes + intercept

    def coefficients(self) -> np.ndarray:
        """Return all regression coefficients.

        Returns
        -------
        np.ndarray, shape (p+1,)
            Array with intercept at index 0 followed by p slope coefficients.
        """
        return self._coefficients.copy()

    def intercept(self) -> float:
        """Return the intercept of the regression model.

        Returns
        -------
        float
            The intercept (bias) term.
        """
        return float(self._coefficients[0])

    def r_squared(self) -> float:
        """Return the coefficient of determination (R²).

        Returns
        -------
        float
            R² value indicating the proportion of variance explained by the model.
        """
        return self._r_squared

    def f_statistic(self) -> float:
        """Return the F-statistic for the overall model significance test.

        Returns
        -------
        float
            The F-statistic value.
        """
        return self._f_statistic

    def p_value(self) -> float:
        """Return the p-value for the overall model significance test.

        Returns
        -------
        float
            The p-value associated with the F-statistic.
        """
        return self._p_value

    def get_params(self) -> OlsMultiRegressionParams:
        """Retrieve regression parameters as a data class.

        .. deprecated:: 0.6.0
            Use accessor methods instead: coefficients(), intercept(),
            r_squared(), f_statistic(), p_value().
            This method will be removed in v1.0.0.

        Returns
        -------
        OlsMultiRegressionParams
            A data class containing all regression parameters.
        """
        warnings.warn(
            "get_params() is deprecated and will be removed in v1.0.0. "
            "Use accessor methods instead: coefficients(), intercept(), "
            "r_squared(), f_statistic(), p_value()",
            DeprecationWarning,
            stacklevel=2,
        )
        return OlsMultiRegressionParams(
            coefficients=self._coefficients,
            r_squared=self._r_squared,
            f_statistic=self._f_statistic,
            p_value=self._p_value,
        )

    def __repr__(self) -> str:
        """Return a string representation of the regression model.

        Returns
        -------
        str
            A string representation including the class name, intercept, and R².
        """
        return (
            f"{self.__class__.__name__}("
            f"intercept={self.intercept():.6f}, "
            f"r_squared={self._r_squared:.6f})"
        )
