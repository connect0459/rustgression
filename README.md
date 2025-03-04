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
    RegressionParams,
    TlsRegressor,
)

# データの準備
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# OLSモデル
ols_model = OlsRegressor(x, y)
ols_params: OlsRegressionParams = ols_model.get_params()
ols_slope = ols_params.slope
ols_intercept = ols_params.intercept
r_value = ols_params.r_value

# TLSモデル
tls_model = TlsRegressor(x, y)
tls_params: RegressionParams = tls_model.get_params()
tls_slope = tls_params.slope
tls_intercept = tls_params.intercept
```

## 著者

[connect0459](https://github.com/connect0459)
