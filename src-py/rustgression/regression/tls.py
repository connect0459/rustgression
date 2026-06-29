"""
Total Least Squares (TLS) regression implementation.
"""

import warnings
from typing import NoReturn

import numpy as np

from ._rust_imports import calculate_tls_regression
from .base import BaseRegressor, TlsRegressionParams


class TlsRegressor(BaseRegressor[TlsRegressionParams]):
    """Class for calculating Total Least Squares (TLS) regression.

    Unlike Ordinary Least Squares (OLS), which minimizes errors only in the
    y-direction, TLS considers errors in both variables (x and y). This
    approach is more appropriate when measurement errors exist in both
    variables.

    Note on goodness-of-fit metrics: residuals() returns *vertical* residuals
    (y - predict(x)), not orthogonal residuals.  r_squared() returns the
    squared Pearson correlation coefficient and does NOT satisfy
    r_squared() == 1 - SS_res / SS_tot for TLS; combining both metrics for
    a classical R² check will not produce consistent results.

    """

    def _fit(self) -> None:
        """Perform TLS regression."""
        (
            _,
            self._slope,
            self._intercept,
            self._r_value,
            self._p_value,
            self._stderr,
            self._intercept_stderr,
        ) = calculate_tls_regression(self.x, self.y)

    def slope(self) -> float:
        """Return the slope of the regression line

        Returns
        -------
        float
            The slope of the regression line
        """
        return self._slope

    def intercept(self) -> float:
        """Return the intercept of the regression line

        Returns
        -------
        float
            The intercept of the regression line
        """
        return self._intercept

    def r_value(self) -> float:
        """Return the correlation coefficient

        Returns
        -------
        float
            The correlation coefficient
        """
        return self._r_value

    def p_value(self) -> float:
        """Return the two-tailed p-value for the slope.

        Computed via a t-test with n-2 degrees of freedom using the Deming
        (orthogonal, λ=1) slope variance estimator, which matches equal-weight
        ODR (scipy.odr / odrpack).

        Returns
        -------
        float
            The p-value for the hypothesis test that slope equals zero.
        """
        return self._p_value

    def stderr(self) -> float:
        """Return the standard error of the slope.

        Uses the Deming (orthogonal, λ=1) variance estimator: the effective
        sum of squares Sxx* = Σ(xi* - x̄)², where xi* is the foot of the
        perpendicular from each observed point to the TLS line.  This matches
        equal-weight ODR (scipy.odr / odrpack) to within ~1%.

        Returns
        -------
        float
            The standard error of the slope estimate.
        """
        return self._stderr

    def intercept_stderr(self) -> float:
        """Return the standard error of the intercept.

        Returns
        -------
        float
            The standard error of the intercept estimate.
        """
        return self._intercept_stderr

    @staticmethod
    def _interval_not_supported(method_name: str) -> NoReturn:
        raise NotImplementedError(
            f"{method_name}() is not supported for TlsRegressor. "
            "TLS confidence and prediction intervals require bootstrap or "
            "jackknife inference."
        )

    def confidence_interval(self, alpha: float = 0.05) -> NoReturn:
        """Not implemented for TLS.

        Raises
        ------
        NotImplementedError
            TLS confidence intervals require bootstrap or jackknife inference.
        """
        self._interval_not_supported("confidence_interval")

    def prediction_interval(self, x_new: np.ndarray, alpha: float = 0.05) -> NoReturn:
        """Not implemented for TLS.

        Raises
        ------
        NotImplementedError
            TLS prediction intervals require bootstrap or jackknife inference.
        """
        self._interval_not_supported("prediction_interval")

    def r_squared(self) -> float:
        """Return the squared Pearson correlation coefficient.

        For TLS, this is the squared Pearson correlation between x and y
        and does NOT equal 1 - SS_res / SS_tot (the classical coefficient
        of determination), because TLS minimises orthogonal distances while
        residuals() returns vertical residuals.

        Returns
        -------
        float
            The square of the Pearson correlation coefficient.
        """
        return self._r_value**2

    def get_params(self) -> TlsRegressionParams:
        """Retrieve regression parameters.

        .. deprecated:: 0.2.0
            Use property methods instead: slope(), intercept(), r_value(),
            p_value(), stderr(), intercept_stderr()
            This method will be removed in v1.0.0.

        Returns
        -------
        TlsRegressionParams
            A data class containing the regression parameters.
        """
        warnings.warn(
            "get_params() is deprecated and will be removed in v1.0.0. "
            "Use property methods instead: slope(), intercept(), r_value(), "
            "p_value(), stderr(), intercept_stderr()",
            DeprecationWarning,
            stacklevel=2,
        )
        return TlsRegressionParams(
            slope=self._slope,
            intercept=self._intercept,
            r_value=self._r_value,
            p_value=self._p_value,
            stderr=self._stderr,
            intercept_stderr=self._intercept_stderr,
        )
