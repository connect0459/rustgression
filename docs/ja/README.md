# rustgression

本プロジェクトは、高速な回帰分析（OLS、TLS）をPythonパッケージとして提供します。

## 概要

`rustgression`はRustを利用した高速な回帰分析ツールをPythonパッケージとして提供します。
以下の機能を含みます：

- **Ordinary Least Squares (OLS)**: 通常の最小二乗法。y方向の誤差のみを最小化します。
- **Total Least Squares (TLS)**: 直交回帰。両方の変数（x軸とy軸）の誤差を考慮します。

Pythonのバージョンは`3.11`以上を対象としています。

## インストール

```bash
pip install rustgression
```

## 使用方法

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
    # サンプルデータの生成
    x, y = generate_sample_data()
    
    # 通常最小二乗法 (OLS) 回帰
    print("=== Ordinary Least Squares (OLS) Results ===")
    ols_model = OlsRegressor(x, y)
    ols_params: OlsRegressionParams = ols_model.get_params()
    print(f"Slope: {ols_params.slope:.4f}")
    print(f"Intercept: {ols_params.intercept:.4f}")
    print(f"R-value: {ols_params.r_value:.4f}")
    print(f"P-value: {ols_params.p_value:.4e}")
    print(f"Standard Error: {ols_params.stderr:.4f}")
    print(f"Intercept Standard Error: {ols_params.intercept_stderr:.4f}\n")
    
    # 全最小二乗法 (TLS) 回帰
    print("=== Total Least Squares (TLS) Results ===")
    tls_model = TlsRegressor(x, y)
    tls_params: TlsRegressionParams = tls_model.get_params()
    print(f"Slope: {tls_params.slope:.4f}")
    print(f"Intercept: {tls_params.intercept:.4f}")
    print(f"R-value: {tls_params.r_value:.4f}")

if __name__ == "__main__":
    main()
```

## 著者

[connect0459](https://github.com/connect0459)
