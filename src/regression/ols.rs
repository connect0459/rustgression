use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

/// Rust implementation of Ordinary Least Squares regression (similar to stats.linregress).
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data for the x-axis.
/// y : numpy.ndarray
///     Data for the y-axis.
///
/// Returns
/// -------
/// tuple
///     A tuple containing (predicted values, slope, intercept, r_value, p_value, stderr, intercept_stderr).
#[pyfunction]
pub fn calculate_ols_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, f64, f64, f64, f64, f64, f64)> {
    // Convert numpy arrays to ndarray
    let x_array = x.as_array().to_owned();
    let y_array = y.as_array().to_owned();
    let n = x_array.len() as f64;

    if n < 2.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least 2 data points are required for regression",
        ));
    }

    // Calculate means
    let x_mean = x_array.mean().unwrap();
    let y_mean = y_array.mean().unwrap();

    // Calculate variance and covariance
    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..x_array.len() {
        let x_diff = x_array[i] - x_mean;
        let y_diff = y_array[i] - y_mean;

        ss_xx += x_diff * x_diff;
        ss_xy += x_diff * y_diff;
        ss_yy += y_diff * y_diff;
    }

    // Calculate slope
    let slope = ss_xy / ss_xx;

    // Calculate intercept
    let intercept = y_mean - slope * x_mean;

    // Calculate correlation coefficient
    let r_value = if ss_xx * ss_yy > 0.0 {
        ss_xy / (ss_xx.sqrt() * ss_yy.sqrt())
    } else {
        0.0
    };

    // Calculate residual sum of squares
    let mut ss_res = 0.0;
    for i in 0..x_array.len() {
        let y_pred = slope * x_array[i] + intercept;
        let diff = y_array[i] - y_pred;
        ss_res += diff * diff;
    }

    // Calculate standard error
    let stderr = if n > 2.0 && ss_xx > 0.0 {
        let sd_res = (ss_res / (n - 2.0)).sqrt();
        sd_res / ss_xx.sqrt()
    } else {
        f64::NAN
    };

    // Calculate t-statistic and p-value (two-tailed test)
    let p_value = if n > 2.0 && stderr != 0.0 {
        let t_stat = slope.abs() / stderr;
        // Calculate p-value from Student's t-distribution (approximation)
        calculate_p_value(t_stat, n - 2.0)
    } else {
        f64::NAN
    };

    // Calculate standard error of the intercept
    let intercept_stderr = if n > 2.0 && ss_xx > 0.0 {
        let sd_res = (ss_res / (n - 2.0)).sqrt();
        sd_res * ((1.0 / n) + (x_mean * x_mean) / ss_xx).sqrt()
    } else {
        f64::NAN
    };

    // Calculate predicted values
    let y_pred = x_array.mapv(|v| slope * v + intercept);

    // Return results to Python
    Ok((
        y_pred.into_pyarray(py),
        slope,
        intercept,
        r_value,
        p_value,
        stderr,
        intercept_stderr,
    ))
}

// Function to calculate p-value from t-statistic (simple implementation)
fn calculate_p_value(t_value: f64, df: f64) -> f64 {
    // Use normal distribution approximation for large degrees of freedom
    if df > 30.0 {
        let x = t_value / (df.sqrt());
        2.0 * (1.0 - normal_cdf(x.abs()))
    } else {
        // Use simple approximation for small degrees of freedom
        // A more accurate implementation of the t-distribution CDF is needed in practice
        let x = df / (df + t_value * t_value);
        // Incomplete p-value due to beta function approximation
        let p = 1.0 - x.powf(df / 2.0);
        2.0 * p
    }
}

// CDF of the standard normal distribution
// Low accuracy due to simple implementation
fn normal_cdf(x: f64) -> f64 {
    // Approximation of the error function
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 {
        1.0 - prob
    } else {
        prob
    }
}
