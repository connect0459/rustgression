"""
Base classes and data structures for regression analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from scipy.stats import t as t_dist

T = TypeVar("T")


@dataclass(frozen=True)
class OlsMultiRegressionParams:
    """Data class to store parameters for multiple Ordinary Least Squares regression.

    Attributes
    ----------
    coefficients : np.ndarray
        Shape (p+1,) array with intercept at index 0 followed by p slope coefficients.
    r_squared : float
        The coefficient of determination (R²) indicating goodness of fit.
    f_statistic : float
        The F-statistic for the overall model significance test.
    p_value : float
        The p-value for the F-statistic.
    """

    coefficients: np.ndarray
    r_squared: float
    f_statistic: float
    p_value: float


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    p_value : float
        The p-value for the hypothesis test that slope equals zero.
    stderr : float
        The standard error of the slope estimate.
    intercept_stderr : float
        The standard error of the intercept estimate.
    """

    slope: float
    intercept: float
    r_value: float
    p_value: float
    stderr: float
    intercept_stderr: float


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
        self._stderr: float
        self._intercept_stderr: float

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

    def r_squared(self) -> float:
        """Return the squared Pearson correlation coefficient.

        For OLS, this is mathematically equivalent to the coefficient of
        determination (R²), i.e. 1 - SS_res / SS_tot.  For TLS the
        identity does not hold in general because TLS minimises orthogonal
        distances while residuals() returns vertical residuals.

        Returns
        -------
        float
            The square of the Pearson correlation coefficient.
        """
        return self._r_value**2

    def residuals(self) -> np.ndarray:
        """Return the vertical residuals of the fitted model.

        Returns
        -------
        np.ndarray
            The difference between observed and predicted values: y - predict(x).
        """
        return self.y - self.predict(self.x)

    def confidence_interval(
        self, alpha: float = 0.05
    ) -> dict[str, tuple[float, float]]:
        """Return confidence intervals for the slope and intercept.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to 0.05 (95% CI).

        Returns
        -------
        dict[str, tuple[float, float]]
            A dict with keys ``"slope"`` and ``"intercept"``, each mapped to
            a ``(lower, upper)`` tuple.
        """
        n = len(self.x)
        t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=n - 2)
        slope_lo = self._slope - t_crit * self._stderr
        slope_hi = self._slope + t_crit * self._stderr
        int_lo = self._intercept - t_crit * self._intercept_stderr
        int_hi = self._intercept + t_crit * self._intercept_stderr
        return {"slope": (slope_lo, slope_hi), "intercept": (int_lo, int_hi)}

    def prediction_interval(self, x_new: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Return prediction intervals for new observations.

        Parameters
        ----------
        x_new : np.ndarray
            New x values for which to compute prediction intervals.
        alpha : float, optional
            Significance level. Defaults to 0.05 (95% PI).

        Returns
        -------
        np.ndarray
            Array of shape ``(len(x_new), 2)`` where each row is
            ``[lower, upper]``.
        """
        x_new = np.asarray(x_new, dtype=np.float64)
        n = len(self.x)
        x_mean = self.x.mean()
        ss_xx = np.sum((self.x - x_mean) ** 2)
        s = np.sqrt(np.sum(self.residuals() ** 2) / (n - 2))
        t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=n - 2)
        y_hat = self.predict(x_new)
        se_pred = s * np.sqrt(1.0 + 1.0 / n + (x_new - x_mean) ** 2 / ss_xx)
        return np.column_stack([y_hat - t_crit * se_pred, y_hat + t_crit * se_pred])

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
