# rustgression

[![PyPI Downloads](https://static.pepy.tech/badge/rustgression)](https://pepy.tech/projects/rustgression)

このプロジェクトは、高速な回帰分析（OLS、TLS）をPythonパッケージとして提供します。

- **WebSite**: <https://pypi.org/project/rustgression/>
- **Documentation**: <https://github.com/connect0459/rustgression/blob/main/README.md>
- **Source code**: <https://github.com/connect0459/rustgression>
- **Bug reports and security issues**: <https://github.com/connect0459/rustgression/issues>

## 概要

`rustgression`は、RustでPythonパッケージとして実装された高性能な回帰分析ツールを提供します。
以下の機能が含まれています：

- **最小二乗法（OLS）**: 従来の最小二乗法。y方向のみの誤差を最小化します。
- **全最小二乗法（TLS）**: 直交回帰。両方の変数（x軸とy軸）の誤差を考慮します。

このパッケージは Python `3.11` 以上をサポートしています。

## インストール

PyPIからインストール:

```bash
pip install rustgression
```

## クイックスタート

簡単な例から始めましょう：

```python
import numpy as np
from rustgression import OlsRegressor, TlsRegressor

# サンプルデータの生成
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# 最小二乗法（OLS）
ols_model = OlsRegressor(x, y)
print(f"OLS - 傾き: {ols_model.slope():.4f}, 切片: {ols_model.intercept():.4f}")

# 全最小二乗法（TLS）
tls_model = TlsRegressor(x, y)
print(f"TLS - 傾き: {tls_model.slope():.4f}, 切片: {tls_model.intercept():.4f}")
```

## APIリファレンス

### OlsRegressor

最小二乗法回帰の実装。

- コンストラクタ

```python
OlsRegressor(x: np.ndarray, y: np.ndarray)
```

- メソッド

- `slope() -> float`: 回帰直線の傾きを返します
- `intercept() -> float`: 回帰直線のy切片を返します
- `r_value() -> float`: 相関係数を返します
- `p_value() -> float`: 傾きのp値を返します
- `stderr() -> float`: 傾きの標準誤差を返します
- `intercept_stderr() -> float`: 切片の標準誤差を返します

### TlsRegressor

全最小二乗法（直交）回帰の実装。

- コンストラクタ

```python
TlsRegressor(x: np.ndarray, y: np.ndarray)
```

- メソッド

- `slope() -> float`: 回帰直線の傾きを返します
- `intercept() -> float`: 回帰直線のy切片を返します
- `r_value() -> float`: 相関係数を返します

## 使用例

より詳細な例については、以下をご覧ください：

- [シンプルな例](../../examples/simple_example.py)
- [科学的な例](../../examples/scientific_example.py)

## パフォーマンス

`rustgression`は最適なパフォーマンスのためにRustで実装されています。以下を提供します：

- 大規模データセットの高速計算
- メモリ効率的なアルゴリズム
- 信頼性の高い数値安定性

## 比較：OLS vs TLS

- **OLS（最小二乗法）**: 垂直距離を最小化。y値のみに誤差があると仮定します。
- **TLS（全最小二乗法）**: 直交距離を最小化。xとy両方の値に誤差があることを考慮します。

両方の変数に測定誤差がある場合はTLSを、従属変数のみに誤差がある場合はOLSを使用してください。

## コントリビューション

コントリビューションを歓迎します！開発環境のセットアップとガイドラインについては、以下をご覧ください：

**🔗 [開発者ドキュメント](development.md)**

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。

## 作者

[connect0459](https://github.com/connect0459)
