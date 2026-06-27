"""
Factory function for creating regression models.
"""

from typing import Literal

import numpy as np

from .ols import OlsRegressor
from .ols_multi import OlsMultiRegressor
from .tls import TlsRegressor


def create_regressor(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["ols", "tls", "ols_multi"] = "ols",
) -> OlsRegressor | TlsRegressor | OlsMultiRegressor:
    """Factory function for creating a regression model.

    Parameters
    ----------
    x : np.ndarray
        Input data for the independent variable. Use a 1D array for "ols" and
        "tls", or a 2D array of shape (n, p) for "ols_multi".
    y : np.ndarray
        Input data for the dependent variable (response vector).
    method : str
        The regression method to use: "ols", "tls", or "ols_multi".

    Returns
    -------
    OlsRegressor | TlsRegressor | OlsMultiRegressor
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
    elif method == "ols_multi":
        return OlsMultiRegressor(x, y)
    else:
        raise ValueError(f"Unknown regression method: {method}")
