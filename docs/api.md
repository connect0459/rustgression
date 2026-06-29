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
| `confidence_interval(alpha=0.05)` | `dict[str, tuple[float, float]]` | 95% CI for slope and intercept |
| `prediction_interval(x_new, alpha=0.05)` | `np.ndarray` | Prediction intervals for new x values |

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

## OLS vs TLS

| | OLS | TLS |
| :--- | :--- | :--- |
| Error direction | Y only | Both X and Y |
| Use case | Only dependent variable has measurement error | Both variables have measurement error |
| Algorithm | Rust-backed linear regression | Rust-backed SVD orthogonal regression |

Use **TLS** when both variables have measurement errors (e.g., sensor data, scientific measurements).
Use **OLS** when only the dependent variable has errors (e.g., time-series prediction).

## More Examples

Please see <https://github.com/connect0459/rustgression/tree/main/examples>.
