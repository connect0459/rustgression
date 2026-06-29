"""
rustgression
===========

A Python package for fast OLS and TLS regression using a Rust backend.

This package provides high-performance regression analysis with a backend
implemented in Rust. It supports ordinary least squares (OLS) regression
for single and multiple predictors, and Total Least Squares (TLS) orthogonal
regression.

Main Features
-------------
- Fast Rust backend
- Ordinary Least Squares (OLS) single-predictor regression
- Ordinary Least Squares (OLS) multi-predictor regression
- Total Least Squares (TLS) orthogonal regression
- User-friendly Python interface

Classes
-------
- OlsRegressor
    Single-predictor OLS regression.

- OlsMultiRegressor
    Multi-predictor OLS regression.

- OlsMultiRegressionParams
    Result dataclass returned by OlsMultiRegressor.get_params().

- TlsRegressor
    Total Least Squares (orthogonal) regression.

Functions
---------
- create_regressor
    Factory function for creating a regressor (supports "ols", "tls",
    "ols_multi").

References
----------
Van Huffel, S., & Vandewalle, J. (1991). The Total Least Squares Problem:
Computational Aspects and Analysis. SIAM.

Examples
--------
>>> import numpy as np
>>> import rustgression
>>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
>>> y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
>>> regressor = rustgression.create_regressor(x, y)
>>> regressor.slope()
2.0...
"""

# Package version
__version__ = "0.6.0"

# Check availability of Rust module (actual import is done in _rust_imports.py)
try:
    from .rustgression import calculate_ols_regression

    # Do nothing on successful import (actual usage is done in other modules)
    del calculate_ols_regression
except ImportError as e:
    import sys

    print(f"Error importing Rust module: {e}", file=sys.stderr)
    print("Rust extension was not properly compiled or installed.", file=sys.stderr)

# Next, import Python wrapper and public Rust symbols together so that
# __all__ is only populated when every advertised name is actually bound.
try:
    from .regression import (
        OlsMultiRegressionParams,
        OlsMultiRegressor,
        OlsRegressionParams,
        OlsRegressor,
        TlsRegressionParams,
        TlsRegressor,
        create_regressor,
    )
    from .rustgression import NumericalWarning

    __all__ = [
        "NumericalWarning",
        "OlsMultiRegressionParams",
        "OlsMultiRegressor",
        "OlsRegressionParams",
        "OlsRegressor",
        "TlsRegressionParams",
        "TlsRegressor",
        "create_regressor",
    ]
except ImportError as e:
    import sys

    print(f"Error importing regression module: {e}", file=sys.stderr)
