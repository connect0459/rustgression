# API Reference

## OlsRegressor

Ordinary Least Squares regression implementation.

### Constructor

```python
OlsRegressor(x: np.ndarray, y: np.ndarray)
```

### Methods

| Method | Return type | Description |
| :--- | :--- | :--- |
| `slope()` | `float` | Slope of the regression line |
| `intercept()` | `float` | Y-intercept of the regression line |
| `r_value()` | `float` | Correlation coefficient |
| `p_value()` | `float` | P-value for the slope |
| `stderr()` | `float` | Standard error of the slope |
| `intercept_stderr()` | `float` | Standard error of the intercept |

### Example

```python
import numpy as np
from rustgression import OlsRegressor

x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

model = OlsRegressor(x, y)
print(f"Slope: {model.slope():.4f}")
print(f"Intercept: {model.intercept():.4f}")
print(f"R-value: {model.r_value():.4f}")
print(f"P-value: {model.p_value():.4e}")
print(f"Std Error: {model.stderr():.4f}")
print(f"Intercept Std Error: {model.intercept_stderr():.4f}")
```

## TlsRegressor

Total Least Squares (orthogonal) regression implementation.

### Constructor

```python
TlsRegressor(x: np.ndarray, y: np.ndarray)
```

### Methods

| Method | Return type | Description |
| :--- | :--- | :--- |
| `slope()` | `float` | Slope of the regression line |
| `intercept()` | `float` | Y-intercept of the regression line |
| `r_value()` | `float` | Correlation coefficient |
| `p_value()` | `float` | P-value for the slope |
| `stderr()` | `float` | Standard error of the slope |
| `intercept_stderr()` | `float` | Standard error of the intercept |

### Example

```python
import numpy as np
from rustgression import TlsRegressor

x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

model = TlsRegressor(x, y)
print(f"Slope: {model.slope():.4f}")
print(f"Intercept: {model.intercept():.4f}")
print(f"R-value: {model.r_value():.4f}")
print(f"P-value: {model.p_value():.4e}")
print(f"Std Error: {model.stderr():.4f}")
print(f"Intercept Std Error: {model.intercept_stderr():.4f}")
```

## OLS vs TLS

| | OLS | TLS |
| :--- | :--- | :--- |
| Error direction | Y only | Both X and Y |
| Use case | Only dependent variable has measurement error | Both variables have measurement error |
| Algorithm | Linear regression via scipy | SVD-based orthogonal regression |

Use **TLS** when both variables have measurement errors (e.g., sensor data, scientific measurements).
Use **OLS** when only the dependent variable has errors (e.g., time-series prediction).

## More Examples

- [Simple Example](../examples/simple_example.py)
- [Scientific Example](../examples/scientific_example.py)
