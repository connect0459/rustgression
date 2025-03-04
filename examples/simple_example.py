import numpy as np

from rustgression import TlsRegressor


def main():
    # テストデータの生成
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_slope = 2.0
    true_intercept = 1.0
    y = true_slope * x + true_intercept + np.random.normal(0, 0.5, 100)

    # TlsRegressorのインスタンス化とフィッティング
    model = TlsRegressor(x, y)

    # パラメータの取得と表示
    params = model.get_params()
    print(f"推定された傾き: {params.slope:.4f} (真の値: {true_slope})")
    print(f"推定された切片: {params.intercept:.4f} (真の値: {true_intercept})")
    print(f"相関係数: {params.correlation:.4f}")

    # 予測の実行
    x_test = np.array([0, 5, 10])
    y_pred = model.predict(x_test)
    print("\n予測結果:")
    for x_val, y_val in zip(x_test, y_pred, strict=True):
        print(f"x = {x_val:.1f} -> y = {y_val:.4f}")


if __name__ == "__main__":
    main()
