#!/bin/bash
# テスト環境確認スクリプト

echo "環境情報:"
uname -a
uv run python --version
rustc --version

echo "Rustgressionをビルドしています..."
uv run maturin develop

echo "インポートテスト:"
uv run python -c "import rustgression; print(f'Successfully imported rustgression {rustgression.__version__}')"

echo "基本的なテストを実行しています..."
uv run python -c "
import numpy as np
from rustgression import OlsRegressor, TlsRegressor

# テストデータ生成
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, 100)

# OLS回帰テスト
print('OLS回帰テスト:')
ols = OlsRegressor(x, y)
ols_params = ols.get_params()
print(f'傾き: {ols_params.slope:.4f}')
print(f'切片: {ols_params.intercept:.4f}')
print(f'R値: {ols_params.r_value:.4f}')

# TLS回帰テスト
print('\\nTLS回帰テスト:')
tls = TlsRegressor(x, y)
tls_params = tls.get_params()
print(f'傾き: {tls_params.slope:.4f}')
print(f'切片: {tls_params.intercept:.4f}')
print(f'R値: {tls_params.r_value:.4f}')
"

echo "テスト完了"
