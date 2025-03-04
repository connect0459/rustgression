# regression_tools

高速な回帰分析ツール（TLSとOLSの実装）

## 概要

`regression_tools`はPythonでRustを利用した高速な回帰分析ツールを提供します。
以下の機能を含みます：

- **Ordinary Least Squares (OLS)**: 通常の最小二乗法。y方向の誤差のみを最小化します。
- **Total Least Squares (TLS)**: 直交回帰。両方の変数（x軸とy軸）の誤差を考慮します。

## インストール

```bash
pip install regression_tools
```

## 使用方法

```python
import numpy as np
from regression_tools import TlsRegressor, OlsRegressor

# データの準備
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# OLSモデル
ols_model = OlsRegressor()
ols_model.fit(x, y)
ols_slope, ols_intercept, ols_r_value = ols_model.get_params()

# TLSモデル
tls_model = TlsRegressor()
tls_model.fit(x, y)
tls_slope, tls_intercept, tls_r_value = tls_model.get_params()
```
