"""
Total Least Squares (TLS) regression implementation.
"""

import warnings

from ._rust_imports import calculate_tls_regression
from .base import BaseRegressor, TlsRegressionParams


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
        _, self._slope, self._intercept, self._r_value = calculate_tls_regression(
            self.x, self.y
        )

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

    def get_params(self) -> TlsRegressionParams:
        """Retrieve regression parameters.

        .. deprecated:: 0.2.0
            Use property methods instead: slope(), intercept(), r_value()
            This method will be removed in v1.0.0.

        Returns
        -------
        TlsRegressionParams
            A data class containing the regression parameters.
        """
        warnings.warn(
            "get_params() is deprecated and will be removed in v1.0.0. "
            "Use property methods instead: slope(), intercept(), r_value()",
            DeprecationWarning,
            stacklevel=2,
        )
        return TlsRegressionParams(
            slope=self._slope, intercept=self._intercept, r_value=self._r_value
        )
