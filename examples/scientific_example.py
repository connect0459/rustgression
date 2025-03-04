import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from rustgression import (
    OlsRegressionParams,
    OlsRegressor,
    TlsRegressionParams,
    TlsRegressor,
)


@dataclass
class RegressionData:
    """Data class for regression analysis.

    Attributes
    ----------
    x : np.ndarray
        True x values.
    y : np.ndarray
        True y values.
    x_noisy : np.ndarray
        Noisy x values.
    y_noisy : np.ndarray
        Noisy y values.
    true_slope : float
        The true slope of the linear relationship.
    true_intercept : float
        The true intercept of the linear relationship.
    description : str
        A description of the dataset.
    """

    x: np.ndarray
    y: np.ndarray
    x_noisy: np.ndarray
    y_noisy: np.ndarray
    true_slope: float
    true_intercept: float
    description: str


def generate_linear_measurement_data(n_points: int = 100) -> RegressionData:
    """Generate simple linear measurement data.

    Parameters
    ----------
    n_points : int, optional
        The number of data points to generate (default is 100).

    Returns
    -------
    RegressionData
        An instance of RegressionData containing the generated data.
    """
    np.random.seed(42)
    x = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0

    # Both x and y have similar noise
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
    """Generate data resembling spectroscopy, with small x-axis error and large y-axis error.

    Parameters
    ----------
    n_points : int, optional
        The number of data points to generate (default is 100).

    Returns
    -------
    RegressionData
        An instance of RegressionData containing the generated data.
    """
    np.random.seed(43)
    x = np.linspace(0, 5, n_points)
    true_slope = 0.5
    true_intercept = 2.0

    # Small error in x-axis and large error in y-axis
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
    """Generate data resembling particle tracking, with large errors in both axes.

    Parameters
    ----------
    n_points : int, optional
        The number of data points to generate (default is 100).

    Returns
    -------
    RegressionData
        An instance of RegressionData containing the generated data.
    """
    np.random.seed(44)
    x = np.linspace(0, 8, n_points)
    true_slope = 1.5
    true_intercept = 0.5

    # Large errors in both axes
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
    """Plot comparison of OLS regression using stats.linregress and OlsRegressor.

    Parameters
    ----------
    data : RegressionData
        The regression data to be analyzed.
    output_dirpath : str | Path | None
        The directory path where the figure will be saved.
    output_filename : str, optional
        The filename for the saved figure (default is "ols_comparison.png").
    save_fig : bool, optional
        Whether to save the figure (default is True).
    show_fig : bool, optional
        Whether to display the figure (default is True).
    """
    # stats.linregress
    stats_slope, stats_intercept, stats_r_value, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # OlsRegressor
    ols_model = OlsRegressor(data.x_noisy, data.y_noisy)
    ols_params = ols_model.get_params()

    # Prepare data for plotting
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
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_r_value:.3f})",
    )
    plt.plot(
        x_plot,
        ols_y_plot,
        color="red",
        linewidth=2,
        label=f"OlsRegressor (Slope={ols_params.slope:.3f}, r={ols_params.r_value:.3f})",
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
    """Plot comparison of TLS regression using stats.linregress and TlsRegressor.

    Parameters
    ----------
    data : RegressionData
        The regression data to be analyzed.
    output_dirpath : str | Path | None
        The directory path where the figure will be saved.
    output_filename : str, optional
        The filename for the saved figure (default is "tls_comparison.png").
    save_fig : bool, optional
        Whether to save the figure (default is True).
    show_fig : bool, optional
        Whether to display the figure (default is True).
    """
    # stats.linregress
    stats_slope, stats_intercept, stats_r_value, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # TlsRegressor
    tls_model = TlsRegressor(data.x_noisy, data.y_noisy)
    tls_params = tls_model.get_params()

    # Prepare data for plotting
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
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_r_value:.3f})",
    )
    plt.plot(
        x_plot,
        tls_y_plot,
        color="red",
        linewidth=2,
        label=f"TlsRegressor (Slope={tls_params.slope:.3f}, r={tls_params.r_value:.3f})",
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
    """Plot comparison of all regression methods.

    Parameters
    ----------
    data : RegressionData
        The regression data to be analyzed.
    output_dirpath : str | Path | None
        The directory path where the figure will be saved.
    output_filename : str, optional
        The filename for the saved figure (default is "all_comparison.png").
    save_fig : bool, optional
        Whether to save the figure (default is True).
    show_fig : bool, optional
        Whether to display the figure (default is True).
    """
    # stats.linregress
    stats_slope, stats_intercept, stats_r_value, _, _ = stats.linregress(
        data.x_noisy, data.y_noisy
    )

    # OlsRegressor
    ols_model = OlsRegressor(data.x_noisy, data.y_noisy)
    ols_params: OlsRegressionParams = ols_model.get_params()

    # TlsRegressor
    tls_model = TlsRegressor(data.x_noisy, data.y_noisy)
    tls_params: TlsRegressionParams = tls_model.get_params()

    # Prepare data for plotting
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
        label=f"stats.linregress (Slope={stats_slope:.3f}, r={stats_r_value:.3f})",
    )
    plt.plot(
        x_plot,
        ols_y_plot,
        color="orange",
        linewidth=2,
        label=f"OlsRegressor (Slope={ols_params.slope:.3f}, r={ols_params.r_value:.3f})",
    )
    plt.plot(
        x_plot,
        tls_y_plot,
        color="red",
        linewidth=2,
        label=f"TlsRegressor (Slope={tls_params.slope:.3f}, r={tls_params.r_value:.3f})",
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

    # Test with three different cases
    datasets = [
        (generate_linear_measurement_data(), "linear_measurement"),
        (generate_spectroscopy_data(), "spectroscopy"),
        (generate_particle_tracking_data(), "particle_tracking"),
    ]

    for data, base_filename in datasets:
        # OLS comparison
        plot_ols_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_ols_comparison.png",
            show_fig=False,
        )

        # TLS comparison
        plot_tls_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_tls_comparison.png",
            show_fig=False,
        )

        # Comparison of all methods
        plot_all_comparison(
            data,
            output_dirpath=output_dir,
            output_filename=f"{base_filename}_all_comparison.png",
            show_fig=False,
        )
