"""
Total Least Squares (TLS) regression implementation.
"""

import warnings as _warnings

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
    def _percentile_bounds(alpha: float) -> tuple[float, float]:
        lo = 100.0 * alpha / 2.0
        return lo, 100.0 - lo

    def confidence_interval(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Return bootstrap confidence intervals for slope and intercept."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}.")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}.")
        n = len(self.x)
        if n < 3:
            raise ValueError(
                "confidence_interval() requires at least 3 data points "
                "(needs at least 1 degree of freedom)."
            )

        rng = np.random.default_rng(random_state)
        boot_slopes = np.empty(n_bootstrap)
        boot_intercepts = np.empty(n_bootstrap)

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                try:
                    _, slope_b, intercept_b, *_ = calculate_tls_regression(
                        self.x[idx], self.y[idx]
                    )
                    boot_slopes[i] = slope_b
                    boot_intercepts[i] = intercept_b
                except (ValueError, RuntimeError):
                    boot_slopes[i] = np.nan
                    boot_intercepts[i] = np.nan

        valid = int(np.sum(np.isfinite(boot_slopes) & np.isfinite(boot_intercepts)))
        if valid < n_bootstrap // 2:
            raise RuntimeError(
                f"Bootstrap failed: only {valid}/{n_bootstrap} resamples "
                "produced finite estimates. The data may be nearly singular."
            )

        lo, hi = self._percentile_bounds(alpha)
        return {
            "slope": (
                float(np.nanpercentile(boot_slopes, lo)),
                float(np.nanpercentile(boot_slopes, hi)),
            ),
            "intercept": (
                float(np.nanpercentile(boot_intercepts, lo)),
                float(np.nanpercentile(boot_intercepts, hi)),
            ),
        }

    def prediction_interval(
        self,
        x_new: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return residual-bootstrap prediction intervals for new observations."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}.")
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap}.")
        x_new = np.asarray(x_new, dtype=np.float64).ravel()
        n = len(self.x)
        if n < 3:
            raise ValueError(
                "prediction_interval() requires at least 3 data points "
                "(needs at least 1 degree of freedom)."
            )

        rng = np.random.default_rng(random_state)
        original_residuals = self.residuals()
        m = len(x_new)
        boot_preds = np.empty((n_bootstrap, m))

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                # Pre-sample residual indices so the RNG state advances by the
                # same number of steps whether or not the regression succeeds,
                # preserving reproducibility across mixed success/failure runs.
                eps_idx = rng.integers(0, n, size=m)
                try:
                    _, slope_b, intercept_b, *_ = calculate_tls_regression(
                        self.x[idx], self.y[idx]
                    )
                    boot_preds[i] = (
                        slope_b * x_new + intercept_b + original_residuals[eps_idx]
                    )
                except (ValueError, RuntimeError):
                    boot_preds[i] = np.nan

        failed = int(np.sum(np.all(~np.isfinite(boot_preds), axis=1)))
        if failed > n_bootstrap // 2:
            raise RuntimeError(
                f"Bootstrap failed: {failed}/{n_bootstrap} resamples produced "
                "non-finite predictions. The data may be nearly singular."
            )

        lo, hi = self._percentile_bounds(alpha)
        lower = np.nanpercentile(boot_preds, lo, axis=0)
        upper = np.nanpercentile(boot_preds, hi, axis=0)
        return np.column_stack([lower, upper])

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
        _warnings.warn(
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
