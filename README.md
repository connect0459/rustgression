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

def generate_sample_data(size: int = 100, noise_std: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data for regression example.
    
    Args:
        size: Number of data points
        noise_std: Standard deviation of noise
    
    Returns:
        Tuple of (x, y) arrays
    """
    x = np.linspace(0, 10, size)
    true_slope, true_intercept = 2.0, 1.0
    y = true_slope * x + true_intercept + np.random.normal(0, noise_std, size)
    return x, y

def main():
    # Generate sample data
    x, y = generate_sample_data()
    
    # Ordinary Least Squares (OLS) Regression
    print("=== Ordinary Least Squares (OLS) Results ===")
    ols_model = OlsRegressor(x, y)
    ols_params: OlsRegressionParams = ols_model.get_params()
    print(f"Slope: {ols_params.slope:.4f}")
    print(f"Intercept: {ols_params.intercept:.4f}")
    print(f"R-value: {ols_params.r_value:.4f}")
    print(f"P-value: {ols_params.p_value:.4e}")
    print(f"Standard Error: {ols_params.stderr:.4f}")
    print(f"Intercept Standard Error: {ols_params.intercept_stderr:.4f}\n")
    
    # Total Least Squares (TLS) Regression
    print("=== Total Least Squares (TLS) Results ===")
    tls_model = TlsRegressor(x, y)
    tls_params: TlsRegressionParams = tls_model.get_params()
    print(f"Slope: {tls_params.slope:.4f}")
    print(f"Intercept: {tls_params.intercept:.4f}")
    print(f"R-value: {tls_params.r_value:.4f}")

if __name__ == "__main__":
    main()
```

## Author

[connect0459](https://github.com/connect0459)
