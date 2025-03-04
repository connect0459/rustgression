# rustgression

This project provides fast regression analysis (OLS, TLS) as a Python package.

## Overview

`rustgression` provides high-performance regression analysis tools implemented in Rust as a Python package.
It includes the following features:

- **Ordinary Least Squares (OLS)**: Traditional least squares method. Minimizes errors only in the y-direction.
- **Total Least Squares (TLS)**: Orthogonal regression. Considers errors in both variables (x-axis and y-axis).

This package targets Python version `3.11` and above.

## Installation

```bash
pip install rustgression
```

## Usage

```python
import numpy as np
from rustgression import (
    OlsRegressionParams,
    OlsRegressor,
    TlsRegressionParams,
    TlsRegressor,
)

# Prepare data
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# OLS model
ols_model = OlsRegressor(x, y)
ols_params: OlsRegressionParams = ols_model.get_params()
ols_slope = ols_params.slope
ols_intercept = ols_params.intercept
r_value = ols_params.r_value

# TLS model
tls_model = TlsRegressor(x, y)
tls_params: TlsRegressionParams = tls_model.get_params()
tls_slope = tls_params.slope
tls_intercept = tls_params.intercept
```

## Author

[connect0459](https://github.com/connect0459)
