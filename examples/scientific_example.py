import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from regression_tools import (
    OlsRegressionParams,
    OlsRegressor,
    RegressionParams,
    TlsRegressor,
)


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


def plot_ols_comparison(
    data: RegressionData,
    output_dirpath: str | Path | None,
    output_filename: str = "ols_comparison.png",
    save_fig: bool = True,
    show_fig: bool = True,
) -> None:
    """OLS回帰の比較プロット(stats.linregressとOlsRegressor)"""
    # stats.linregress
    stats_slope, stats_intercept, stats_correlation, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # OlsRegressor
    ols_model = OlsRegressor(data.x_noisy, data.y_noisy)
    ols_params = ols_model.get_params()

    # プロット用のデータ準備
    x_plot = np.linspace(min(data.x_noisy), max(data.x_noisy), 100)
    ols_y_plot = ols_model.predict(x_plot)
    stats_y_plot = stats_slope * x_plot + stats_intercept

    plt.figure(figsize=(12, 8))
    plt.scatter(
        data.x_noisy, data.y_noisy, color="blue", alpha=0.5, label="Observed Data"
    )
    plt.plot(
        data.x,
        data.y,
        "--",
        color="green",
        linewidth=2,
        label=f"True Relationship (Slope={data.true_slope:.3f})",
    )
    plt.plot(
        x_plot,
        stats_y_plot,
        color="purple",
        linewidth=2,
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_correlation:.3f})",
    )
    plt.plot(
        x_plot,
        ols_y_plot,
        color="red",
        linewidth=2,
        label=f"OlsRegressor (Slope={ols_params.slope:.3f}, r={ols_params.correlation:.3f})",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"OLS Regression Comparison\n{data.description}")
    plt.legend()
    plt.grid(True)

    if save_fig:
        if output_dirpath is None:
            raise ValueError("When save_fig=True, please specify output_dirpath.")
        os.makedirs(output_dirpath, exist_ok=True)
        plt.savefig(os.path.join(output_dirpath, output_filename))
    if show_fig:
        plt.show()


def plot_tls_comparison(
    data: RegressionData,
    output_dirpath: str | Path | None,
    output_filename: str = "tls_comparison.png",
    save_fig: bool = True,
    show_fig: bool = True,
) -> None:
    """TLS回帰の比較プロット(stats.linregressとTlsRegressor)"""
    # stats.linregress
    stats_slope, stats_intercept, stats_correlation, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # TlsRegressor
    tls_model = TlsRegressor(data.x_noisy, data.y_noisy)
    tls_params = tls_model.get_params()

    # プロット用のデータ準備
    x_plot = np.linspace(min(data.x_noisy), max(data.x_noisy), 100)
    tls_y_plot = tls_model.predict(x_plot)
    stats_y_plot = stats_slope * x_plot + stats_intercept

    plt.figure(figsize=(12, 8))
    plt.scatter(
        data.x_noisy, data.y_noisy, color="blue", alpha=0.5, label="Observed Data"
    )
    plt.plot(
        data.x,
        data.y,
        "--",
        color="green",
        linewidth=2,
        label=f"True Relationship (Slope={data.true_slope:.3f})",
    )
    plt.plot(
        x_plot,
        stats_y_plot,
        color="purple",
        linewidth=2,
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_correlation:.3f})",
    )
    plt.plot(
        x_plot,
        tls_y_plot,
        color="red",
        linewidth=2,
        label=f"TlsRegressor (Slope={tls_params.slope:.3f}, r={tls_params.correlation:.3f})",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"TLS vs OLS Regression Comparison\n{data.description}")
    plt.legend()
    plt.grid(True)

    if save_fig:
        if output_dirpath is None:
            raise ValueError("When save_fig=True, please specify output_dirpath.")
        os.makedirs(output_dirpath, exist_ok=True)
        plt.savefig(os.path.join(output_dirpath, output_filename))
    if show_fig:
        plt.show()


def plot_all_comparison(
    data: RegressionData,
    output_dirpath: str | Path | None,
    output_filename: str = "all_comparison.png",
    save_fig: bool = True,
    show_fig: bool = True,
) -> None:
    """全ての回帰手法の比較プロット"""
    # stats.linregress
    stats_slope, stats_intercept, stats_correlation, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # OlsRegressor
    ols_model = OlsRegressor(data.x_noisy, data.y_noisy)
    ols_params: OlsRegressionParams = ols_model.get_params()

    # TlsRegressor
    tls_model = TlsRegressor(data.x_noisy, data.y_noisy)
    tls_params: RegressionParams = tls_model.get_params()

    # プロット用のデータ準備
    x_plot = np.linspace(min(data.x_noisy), max(data.x_noisy), 100)
    ols_y_plot = ols_model.predict(x_plot)
    tls_y_plot = tls_model.predict(x_plot)
    stats_y_plot = stats_slope * x_plot + stats_intercept

    plt.figure(figsize=(12, 8))
    plt.scatter(
        data.x_noisy, data.y_noisy, color="blue", alpha=0.5, label="Observed Data"
    )
    plt.plot(
        data.x,
        data.y,
        "--",
        color="green",
        linewidth=2,
        label=f"True Relationship (Slope={data.true_slope:.3f})",
    )
    plt.plot(
        x_plot,
        stats_y_plot,
        color="purple",
        linewidth=2,
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_correlation:.3f})",
    )
    plt.plot(
        x_plot,
        ols_y_plot,
        color="orange",
        linewidth=2,
        label=f"OlsRegressor (Slope={ols_params.slope:.3f}, r={ols_params.correlation:.3f})",
    )
    plt.plot(
        x_plot,
        tls_y_plot,
        color="red",
        linewidth=2,
        label=f"TlsRegressor (Slope={tls_params.slope:.3f}, r={tls_params.correlation:.3f})",
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"All Regression Methods Comparison\n{data.description}")
    plt.legend()
    plt.grid(True)

    if save_fig:
        if output_dirpath is None:
            raise ValueError("When save_fig=True, please specify output_dirpath.")
        os.makedirs(output_dirpath, exist_ok=True)
        plt.savefig(os.path.join(output_dirpath, output_filename))
    if show_fig:
        plt.show()


if __name__ == "__main__":
    output_dir = "examples/outputs"

    # 3つの異なるケースでテスト
    datasets = [
        (generate_linear_measurement_data(), "linear_measurement"),
        (generate_spectroscopy_data(), "spectroscopy"),
        (generate_particle_tracking_data(), "particle_tracking"),
    ]

    for data, base_filename in datasets:
        # OLS比較
        plot_ols_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_ols_comparison.png",
            show_fig=False,
        )

        # TLS比較
        plot_tls_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_tls_comparison.png",
            show_fig=False,
        )

        # 全手法比較
        plot_all_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_all_comparison.png",
            show_fig=False,
        )
