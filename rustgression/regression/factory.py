"""
Factory function for creating regression models.
"""

from typing import Literal

import numpy as np

from .ols import OlsRegressor
from .tls import TlsRegressor


def create_regressor(
    x: np.ndarray, y: np.ndarray, method: Literal["ols", "tls"] = "ols"
) -> OlsRegressor | TlsRegressor:
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
    Union[OlsRegressor, TlsRegressor]
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
