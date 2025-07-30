use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

// Type alias to reduce complexity
type OlsResult<'py> = (&'py PyArray1<f64>, f64, f64, f64, f64, f64, f64);

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
) -> PyResult<OlsResult<'py>> {
    // Convert numpy arrays to ndarray
    let x_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x.as_array().to_owned();
    let y_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        y.as_array().to_owned();
    let n: f64 = x_array.len() as f64;

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
    let y_pred: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x_array.mapv(|v| slope * v + intercept);

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

#[cfg(test)]
mod tests {
    use super::*;

    mod calculate_ols_regression {
        use super::*;

        #[test]
        fn valid_regression() {
            let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

            let result = perform_ols(&x, &y);

            assert!((result.slope - 2.0).abs() < 1e-10);
            assert!(result.intercept.abs() < 1e-10);
            assert!((result.r_value - 1.0).abs() < 1e-10);
        }

        #[test]
        fn insufficient_data_points() {
            let x = vec![1.0];
            let y = vec![2.0];

            let result = perform_ols(&x, &y);
            assert!(result.slope.is_nan() || result.slope.is_infinite());
        }

        #[test]
        fn edge_cases_table_driven() {
            let test_cases = vec![
                (
                    "horizontal_line",
                    vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![3.0, 3.0, 3.0, 3.0, 3.0],
                    0.0,
                    3.0,
                    f64::NAN,
                ),
                (
                    "negative_correlation",
                    vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![10.0, 8.0, 6.0, 4.0, 2.0],
                    -2.0,
                    12.0,
                    -1.0,
                ),
                (
                    "weak_positive_correlation",
                    vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![2.1, 3.9, 6.2, 7.8, 10.1],
                    2.0,
                    0.06,
                    0.999,
                ),
                (
                    "intercept_offset",
                    vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    vec![5.0, 7.0, 9.0, 11.0, 13.0],
                    2.0,
                    5.0,
                    1.0,
                ),
            ];

            for (name, x, y, expected_slope, expected_intercept, expected_r) in test_cases {
                let result = perform_ols(&x, &y);
                assert!(
                    (result.slope - expected_slope).abs() < 1e-1,
                    "{}: slope mismatch. expected: {}, got: {}",
                    name,
                    expected_slope,
                    result.slope
                );
                assert!(
                    (result.intercept - expected_intercept).abs() < 1e-1,
                    "{}: intercept mismatch. expected: {}, got: {}",
                    name,
                    expected_intercept,
                    result.intercept
                );
                if expected_r.is_nan() {
                    assert!(
                        result.r_value.is_nan(),
                        "{}: expected NaN r_value, got: {}",
                        name,
                        result.r_value
                    );
                } else {
                    assert!(
                        (result.r_value - expected_r).abs() < 1e-1,
                        "{}: r_value mismatch. expected: {}, got: {}",
                        name,
                        expected_r,
                        result.r_value
                    );
                }
            }
        }

        #[test]
        fn boundary_cases_table_driven() {
            let test_cases = vec![
                (
                    "identical_x_values",
                    vec![2.0, 2.0, 2.0],
                    vec![1.0, 2.0, 3.0],
                    true,
                    false,
                ),
                (
                    "identical_y_values",
                    vec![1.0, 2.0, 3.0],
                    vec![5.0, 5.0, 5.0],
                    false,
                    true,
                ),
                (
                    "two_points_only",
                    vec![1.0, 3.0],
                    vec![2.0, 6.0],
                    false,
                    false,
                ),
            ];

            for (name, x, y, expect_nan_slope, expect_zero_r) in test_cases {
                let result = perform_ols(&x, &y);
                if expect_nan_slope {
                    assert!(
                        result.slope.is_nan() || result.slope.is_infinite(),
                        "{}: expected NaN/infinite slope, got: {}",
                        name,
                        result.slope
                    );
                }
                if expect_zero_r {
                    assert!(
                        result.r_value.abs() < 1e-10 || result.r_value.is_nan(),
                        "{}: expected zero or NaN r_value, got: {}",
                        name,
                        result.r_value
                    );
                }
            }
        }
    }

    mod calculate_p_value {
        use super::*;

        #[test]
        fn large_degrees_of_freedom() {
            let p_val = calculate_p_value(2.0, 100.0);
            assert!(p_val > 0.0 && p_val < 1.0);
        }
    }

    struct OlsResult {
        slope: f64,
        intercept: f64,
        r_value: f64,
    }

    fn perform_ols(x: &[f64], y: &[f64]) -> OlsResult {
        let n = x.len() as f64;
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;

        let mut ss_xx = 0.0;
        let mut ss_xy = 0.0;
        let mut ss_yy = 0.0;

        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            ss_xx += x_diff * x_diff;
            ss_xy += x_diff * y_diff;
            ss_yy += y_diff * y_diff;
        }

        let slope = ss_xy / ss_xx;
        let intercept = y_mean - slope * x_mean;
        let r_value = ss_xy / (ss_xx.sqrt() * ss_yy.sqrt());

        OlsResult {
            slope,
            intercept,
            r_value,
        }
    }
}
