# rustgression

[![PyPI Downloads](https://static.pepy.tech/badge/rustgression)](https://pepy.tech/projects/rustgression)

This project provides fast regression analysis (OLS, TLS) as a Python package.

- WebSite: <https://pypi.org/project/rustgression/>
- Documentation: <https://github.com/connect0459/rustgression/blob/main/README.md>
- Source code: <https://github.com/connect0459/rustgression>
- Report bugs or security vulnerabilities: <https://github.com/connect0459/rustgression/issues>

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
from rustgression import OlsRegressor, TlsRegressor

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
    print(f"Slope: {ols_model.slope():.4f}")
    print(f"Intercept: {ols_model.intercept():.4f}")
    print(f"R-value: {ols_model.r_value():.4f}")
    print(f"P-value: {ols_model.p_value():.4e}")
    print(f"Standard Error: {ols_model.stderr():.4f}")
    print(f"Intercept Standard Error: {ols_model.intercept_stderr():.4f}\n")

    # Total Least Squares (TLS) Regression
    print("=== Total Least Squares (TLS) Results ===")
    tls_model = TlsRegressor(x, y)
    print(f"Slope: {tls_model.slope():.4f}")
    print(f"Intercept: {tls_model.intercept():.4f}")
    print(f"R-value: {tls_model.r_value():.4f}")

if __name__ == "__main__":
    main()
```

## Documentation

For detailed documentation in your preferred language:

- **🇺🇸 [English Documentation](docs/en/README.md)**
- **🇯🇵 [日本語ドキュメント](docs/ja/README.md)**

For developers and contributors:

- **🔧 [Developer Documentation (English)](docs/en/development.md)**
- **🔧 [開発者ドキュメント（日本語）](docs/ja/development.md)**

## Author

[connect0459](https://github.com/connect0459)
