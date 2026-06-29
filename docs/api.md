# API Reference

## OlsRegressor

Ordinary Least Squares regression. Minimizes the sum of squared residuals in the
y-direction (Σ(yᵢ − ŷᵢ)²). Assumes errors exist only in the dependent variable y.

### Constructor

```python
OlsRegressor(x: np.ndarray, y: np.ndarray)
```

### Methods

| Method | Return type | Description |
| :--- | :--- | :--- |
| `slope()` | `float` | Slope of the regression line |
| `intercept()` | `float` | Y-intercept of the regression line |
| `r_value()` | `float` | Pearson correlation coefficient |
| `r_squared()` | `float` | Coefficient of determination (R²) |
| `p_value()` | `float` | P-value for the slope |
| `stderr()` | `float` | Standard error of the slope |
| `intercept_stderr()` | `float` | Standard error of the intercept |
| `predict(x)` | `np.ndarray` | Predicted y values for input x |
| `residuals()` | `np.ndarray` | Vertical residuals (y − ŷ) |
| `confidence_interval(alpha=0.05)` | `dict[str, tuple[float, float]]` | (1−alpha)×100% CI for slope and intercept |
| `prediction_interval(x_new, alpha=0.05)` | `np.ndarray`, shape `(n, 2)` | Prediction intervals; each row is `[lower, upper]` |

### Example

```python
import numpy as np
from rustgression import OlsRegressor

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)

model = OlsRegressor(x, y)
print(f"Slope: {model.slope():.4f}")
print(f"Intercept: {model.intercept():.4f}")
print(f"R-value: {model.r_value():.4f}")
print(f"R²: {model.r_squared():.4f}")
print(f"P-value: {model.p_value():.4e}")
print(f"Std Error: {model.stderr():.4f}")
print(f"Intercept Std Error: {model.intercept_stderr():.4f}")

ci = model.confidence_interval()
print(f"Slope 95% CI: {ci['slope']}")
print(f"Intercept 95% CI: {ci['intercept']}")

pi = model.prediction_interval(np.array([5.0, 10.0]))
print(f"Prediction intervals: {pi}")
```

## TlsRegressor

Total Least Squares (orthogonal) regression. Minimizes the sum of squared orthogonal
distances from each point to the regression line (Σdᵢ²). Assumes measurement errors
exist in both x and y.

> **Note on r_squared():** Returns the squared Pearson correlation coefficient
> (r²), which does **not** equal 1 − SS\_res / SS\_tot for TLS. TLS minimises
> orthogonal distances while `residuals()` returns vertical residuals, so the
> classical coefficient-of-determination identity does not hold.
>
> **Raises RuntimeError:** When the null-vector y-component falls below the
> numerical stability threshold, a `RuntimeError` is raised rather than
> returning silently incorrect output.

### Constructor

```python
TlsRegressor(x: np.ndarray, y: np.ndarray)
```

### Methods

| Method | Return type | Description |
| :--- | :--- | :--- |
| `slope()` | `float` | Slope of the regression line |
| `intercept()` | `float` | Y-intercept of the regression line |
| `r_value()` | `float` | Pearson correlation coefficient |
| `r_squared()` | `float` | Squared Pearson correlation (see note above) |
| `p_value()` | `float` | P-value for the slope (Deming/ODR estimator) |
| `stderr()` | `float` | Standard error of the slope (Deming/ODR estimator) |
| `intercept_stderr()` | `float` | Standard error of the intercept |
| `predict(x)` | `np.ndarray` | Predicted y values for input x |
| `residuals()` | `np.ndarray` | Vertical residuals (y − ŷ) |
| `confidence_interval(alpha=0.05)` | — | **Not implemented** — raises `NotImplementedError` |
| `prediction_interval(x_new, alpha=0.05)` | — | **Not implemented** — raises `NotImplementedError` |

### Example

```python
import numpy as np
from rustgression import TlsRegressor

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)

model = TlsRegressor(x, y)
print(f"Slope: {model.slope():.4f}")
print(f"Intercept: {model.intercept():.4f}")
print(f"R-value: {model.r_value():.4f}")
print(f"R² (squared Pearson r): {model.r_squared():.4f}")
print(f"P-value: {model.p_value():.4e}")
print(f"Std Error: {model.stderr():.4f}")
print(f"Intercept Std Error: {model.intercept_stderr():.4f}")

# predict and residuals are supported for TLS
print(f"Predictions: {model.predict(np.array([5.0, 10.0]))}")
print(f"Residuals (first 5): {model.residuals()[:5]}")

# confidence_interval and prediction_interval raise NotImplementedError for TLS
try:
    model.confidence_interval()
except NotImplementedError as e:
    print(f"Not supported: {e}")
```

## OlsMultiRegressor

Multiple Ordinary Least Squares regression. Fits a model of the form
y = b₀ + b₁x₁ + … + bₚxₚ by minimizing the sum of squared residuals in y.

### Constructor

```python
OlsMultiRegressor(x: np.ndarray, y: np.ndarray)
```

`x` must be a 2D array of shape `(n, p)` where n is the number of observations
and p is the number of predictor variables. `y` is a 1D response vector of length n.

### Methods

| Method | Return type | Description |
| :--- | :--- | :--- |
| `coefficients()` | `np.ndarray` | Shape `(p+1,)` — intercept at index 0, then p slopes |
| `intercept()` | `float` | Intercept term (b₀) |
| `r_squared()` | `float` | Coefficient of determination (R²) |
| `f_statistic()` | `float` | F-statistic for overall model significance |
| `p_value()` | `float` | P-value associated with the F-statistic |
| `predict(x)` | `np.ndarray` | Predicted y values; accepts shape `(p,)` or `(m, p)` |

### Example

```python
import numpy as np
from rustgression import OlsMultiRegressor

rng = np.random.default_rng(0)
x = rng.standard_normal((100, 2))
y = 1.5 * x[:, 0] - 0.8 * x[:, 1] + 2.0 + rng.standard_normal(100) * 0.3

model = OlsMultiRegressor(x, y)
print(f"Coefficients: {model.coefficients()}")
print(f"Intercept: {model.intercept():.4f}")
print(f"R²: {model.r_squared():.4f}")
print(f"F-statistic: {model.f_statistic():.4f}")
print(f"P-value: {model.p_value():.4e}")

x_new = np.array([[1.0, -0.5], [0.0, 2.0]])
print(f"Predictions: {model.predict(x_new)}")
```

## create_regressor

Factory function that instantiates a regression model by method name.

```python
create_regressor(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["ols", "tls", "ols_multi"] = "ols",
) -> OlsRegressor | TlsRegressor | OlsMultiRegressor
```

| `method` | Returns |
| :--- | :--- |
| `"ols"` | `OlsRegressor` |
| `"tls"` | `TlsRegressor` |
| `"ols_multi"` | `OlsMultiRegressor` |

Raises `ValueError` if an unknown method string is provided.

### Example

```python
import numpy as np
from rustgression import create_regressor

# "ols" and "tls" accept a 1D x array
rng = np.random.default_rng(42)
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 100)

ols_model = create_regressor(x, y, method="ols")
tls_model = create_regressor(x, y, method="tls")

# "ols_multi" requires a 2D x array of shape (n, p)
rng = np.random.default_rng(42)
x_multi = rng.standard_normal((100, 2))
y_multi = 1.5 * x_multi[:, 0] - 0.8 * x_multi[:, 1] + 2.0 + rng.standard_normal(100) * 0.3
multi_model = create_regressor(x_multi, y_multi, method="ols_multi")
```

## OLS vs TLS vs Multi-OLS

| | OLS | TLS | Multi-OLS |
| :--- | :--- | :--- | :--- |
| Variables | 1 predictor | 1 predictor | p predictors |
| Error direction | Y only | Both X and Y | Y only |
| Use case | Errors only in y | Both variables have measurement error | Multiple predictors, errors only in y |
| Algorithm | Rust-backed linear regression | Rust-backed SVD orthogonal regression | Rust-backed multiple linear regression |

## More Examples

Please see <https://github.com/connect0459/rustgression/tree/main/examples>.
