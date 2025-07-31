# rustgression

[![PyPI Downloads](https://static.pepy.tech/badge/rustgression)](https://pepy.tech/projects/rustgression)

This project provides fast regression analysis (OLS, TLS) as a Python package.

- **WebSite**: <https://pypi.org/project/rustgression/>
- **Documentation**: <https://github.com/connect0459/rustgression/blob/main/README.md>
- **Source code**: <https://github.com/connect0459/rustgression>
- **Report bugs or security vulnerabilities**: <https://github.com/connect0459/rustgression/issues>

## Overview

`rustgression` provides high-performance regression analysis tools implemented in Rust as a Python package.
It includes the following features:

- **Ordinary Least Squares (OLS)**: Traditional least squares method. Minimizes errors only in the y-direction.
- **Total Least Squares (TLS)**: Orthogonal regression. Considers errors in both variables (x-axis and y-axis).

This package targets Python version `3.11` and above.

## Installation

Install from PyPI:

```bash
pip install rustgression
```

## Quick Start

Here's a simple example to get you started:

```python
import numpy as np
from rustgression import OlsRegressor, TlsRegressor

# Generate sample data
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# Ordinary Least Squares (OLS)
ols_model = OlsRegressor(x, y)
print(f"OLS - Slope: {ols_model.slope():.4f}, Intercept: {ols_model.intercept():.4f}")

# Total Least Squares (TLS)
tls_model = TlsRegressor(x, y)
print(f"TLS - Slope: {tls_model.slope():.4f}, Intercept: {tls_model.intercept():.4f}")
```

## API Reference

### OlsRegressor

Ordinary Least Squares regression implementation.

- Constructor

```python
OlsRegressor(x: np.ndarray, y: np.ndarray)
```

- Methods

- `slope() -> float`: Returns the slope of the regression line
- `intercept() -> float`: Returns the y-intercept of the regression line
- `r_value() -> float`: Returns the correlation coefficient
- `p_value() -> float`: Returns the p-value for the slope
- `stderr() -> float`: Returns the standard error of the slope
- `intercept_stderr() -> float`: Returns the standard error of the intercept

### TlsRegressor

Total Least Squares (orthogonal) regression implementation.

- Constructor

```python
TlsRegressor(x: np.ndarray, y: np.ndarray)
```

- Methods

- `slope() -> float`: Returns the slope of the regression line
- `intercept() -> float`: Returns the y-intercept of the regression line  
- `r_value() -> float`: Returns the correlation coefficient

## Examples

For more detailed examples, check out:

- [Simple Example](../../examples/simple_example.py)
- [Scientific Example](../../examples/scientific_example.py)

## Performance

`rustgression` is implemented in Rust for optimal performance. It provides:

- Fast computation for large datasets
- Memory-efficient algorithms
- Reliable numerical stability

## Comparison: OLS vs TLS

- **OLS (Ordinary Least Squares)**: Minimizes vertical distances. Assumes errors only in y-values.
- **TLS (Total Least Squares)**: Minimizes orthogonal distances. Considers errors in both x and y values.

Use TLS when both variables have measurement errors, and OLS when only the dependent variable has errors.

## Contributing

We welcome contributions! For development setup and guidelines, see:

**🔗 [Developer Documentation](development.md)**

## License

This project is licensed under the MIT License.

## Author

[connect0459](https://github.com/connect0459)
