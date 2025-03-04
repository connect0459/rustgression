import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from tls_regressor import TlsRegressor


@dataclass
class RegressionData:
    """回帰分析用のデータクラス"""

    x: np.ndarray  # 真のx値
    y: np.ndarray  # 真のy値
    x_noisy: np.ndarray  # ノイズ入りx値
    y_noisy: np.ndarray  # ノイズ入りy値
    true_slope: float
    true_intercept: float
    description: str


def generate_linear_measurement_data(n_points: int = 100) -> RegressionData:
    """単純な線形測定データの生成"""
    np.random.seed(42)
    x = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0

    # x, yともに同程度のノイズ
    noise_scale = 0.3
    x_noisy = x + np.random.normal(0, noise_scale, n_points)
    y_noisy = (
        true_slope * x + true_intercept + np.random.normal(0, noise_scale, n_points)
    )

    return RegressionData(
        x=x,
        y=true_slope * x + true_intercept,
        x_noisy=x_noisy,
        y_noisy=y_noisy,
        true_slope=true_slope,
        true_intercept=true_intercept,
        description="Linear Measurement with Equal Noise",
    )


def generate_spectroscopy_data(n_points: int = 100) -> RegressionData:
    """分光分析のような、x軸の誤差が小さく、y軸の誤差が大きいデータ"""
    np.random.seed(43)
    x = np.linspace(0, 5, n_points)
    true_slope = 0.5
    true_intercept = 2.0

    # x軸の誤差は小さく、y軸の誤差は大きい
    x_noisy = x + np.random.normal(0, 0.1, n_points)
    y_noisy = true_slope * x + true_intercept + np.random.normal(0, 0.4, n_points)

    return RegressionData(
        x=x,
        y=true_slope * x + true_intercept,
        x_noisy=x_noisy,
        y_noisy=y_noisy,
        true_slope=true_slope,
        true_intercept=true_intercept,
        description="Spectroscopy-like Data (Small X Error, Large Y Error)",
    )


def generate_particle_tracking_data(n_points: int = 100) -> RegressionData:
    """粒子追跡のような、両軸とも大きな誤差を持つデータ"""
    np.random.seed(44)
    x = np.linspace(0, 8, n_points)
    true_slope = 1.5
    true_intercept = 0.5

    # 両軸とも大きな誤差
    x_noisy = x + np.random.normal(0, 0.5, n_points)
    y_noisy = true_slope * x + true_intercept + np.random.normal(0, 0.5, n_points)

    return RegressionData(
        x=x,
        y=true_slope * x + true_intercept,
        x_noisy=x_noisy,
        y_noisy=y_noisy,
        true_slope=true_slope,
        true_intercept=true_intercept,
        description="Particle Tracking Data (Large Errors in Both Axes)",
    )


def plot_regression_comparison(
    data: RegressionData,
    output_dirpath: str | Path | None,
    output_filename: str = "regression_comparison.png",
    save_fig: bool = True,
    show_fig: bool = True,
) -> tuple[float, float, float, float]:
    """回帰分析の比較プロット"""
    # TLS回帰の実行
    tls_model = TlsRegressor()
    tls_model.fit(data.x_noisy, data.y_noisy)
    tls_slope, tls_intercept, tls_correlation = tls_model.get_params()

    # OLS回帰の実行
    ols_slope, ols_intercept, ols_correlation, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # プロット用のデータ準備
    x_plot = np.linspace(min(data.x_noisy), max(data.x_noisy), 100)
    tls_y_plot = tls_model.predict(x_plot)
    ols_y_plot = ols_slope * x_plot + ols_intercept

    # プロット
    plt.figure(figsize=(12, 8))
    plt.scatter(
        data.x_noisy, data.y_noisy, color="blue", alpha=0.5, label="Observed Data"
    )

    # 真の関係
    plt.plot(
        data.x,
        data.y,
        "--",
        color="green",
        linewidth=2,
        label=f"True Relationship (Slope={data.true_slope:.3f})",
    )

    # 回帰線
    plt.plot(
        x_plot,
        tls_y_plot,
        color="red",
        linewidth=2,
        label=f"TLS Regression (Slope={tls_slope:.3f}, r={tls_correlation:.3f})",
    )
    plt.plot(
        x_plot,
        ols_y_plot,
        color="purple",
        linewidth=2,
        label=f"OLS Regression (Slope={ols_slope:.3f}, r={ols_correlation:.3f})",
    )

    # MSEの計算と表示
    tls_mse = np.mean((data.y_noisy - tls_model.predict(data.x_noisy)) ** 2)
    ols_mse = np.mean((data.y_noisy - (ols_slope * data.x_noisy + ols_intercept)) ** 2)

    plt.text(
        0.02,
        0.98,
        f"TLS MSE: {tls_mse:.4f}\nOLS MSE: {ols_mse:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Regression Comparison\n{data.description}")
    plt.legend()
    plt.grid(True)

    if save_fig:
        if output_dirpath is None:
            raise ValueError("When save_fig=True, please specify output_dirpath.")
        os.makedirs(output_dirpath, exist_ok=True)
        output_filepath = os.path.join(output_dirpath, output_filename)
        plt.savefig(output_filepath)
    if show_fig:
        plt.show()

    return tls_slope, tls_intercept, ols_slope, ols_intercept


if __name__ == "__main__":
    output_dir = "examples/outputs"

    # 3つの異なるケースでテスト
    datasets = [
        (generate_linear_measurement_data(), "linear_measurement.png"),
        (generate_spectroscopy_data(), "spectroscopy.png"),
        (generate_particle_tracking_data(), "particle_tracking.png"),
    ]

    for data, filename in datasets:
        plot_regression_comparison(
            data, output_dirpath=output_dir, output_filename=filename, show_fig=False
        )
