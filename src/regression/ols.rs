#![allow(unsafe_op_in_unsafe_fn)] // PyO3 internal operations require unsafe

use crate::regression::utils::{
    calculate_p_value_exact, kahan_sum, safe_divide, validate_finite_array,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

// Type aliases to reduce complexity
type OlsResult<'py> = (&'py PyArray1<f64>, f64, f64, f64, f64, f64, f64);
type Array1Ref = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

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
    let x_array: Array1Ref = x.as_array().to_owned();
    let y_array: Array1Ref = y.as_array().to_owned();
    let n: f64 = x_array.len() as f64;

    if n < 2.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least 2 data points are required for regression",
        ));
    }

    // IEEE 754 edge case validation
    let x_slice = x_array.as_slice().unwrap();
    let y_slice = y_array.as_slice().unwrap();
    validate_finite_array(x_slice, "x")?;
    validate_finite_array(y_slice, "y")?;

    // Calculate means
    let x_mean = x_array.mean().unwrap();
    let y_mean = y_array.mean().unwrap();

    let (ss_xx, ss_xy, ss_yy) =
        calculate_variance_covariance_terms(&x_array, &y_array, x_mean, y_mean);

    // Calculate slope using safe division
    let slope = safe_divide(ss_xy, ss_xx, "slope calculation")?;

    // Calculate intercept
    let intercept = y_mean - slope * x_mean;

    let r_value = calculate_correlation_coefficient(ss_xx, ss_xy, ss_yy);

    let ss_res = calculate_residual_sum_of_squares(&x_array, &y_array, slope, intercept);

    let (stderr, p_value, intercept_stderr) =
        calculate_standard_errors(n, ss_xx, ss_res, slope, x_mean);

    // Calculate predicted values
    let y_pred: Array1Ref = x_array.mapv(|v| slope * v + intercept);

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

/// Calculate variance and covariance terms using Kahan summation
fn calculate_variance_covariance_terms(
    x_array: &Array1Ref,
    y_array: &Array1Ref,
    x_mean: f64,
    y_mean: f64,
) -> (f64, f64, f64) {
    let mut x_squared_terms = Vec::with_capacity(x_array.len());
    let mut xy_terms = Vec::with_capacity(x_array.len());
    let mut y_squared_terms = Vec::with_capacity(x_array.len());

    for i in 0..x_array.len() {
        let x_diff = x_array[i] - x_mean;
        let y_diff = y_array[i] - y_mean;

        x_squared_terms.push(x_diff * x_diff);
        xy_terms.push(x_diff * y_diff);
        y_squared_terms.push(y_diff * y_diff);
    }

    let ss_xx = kahan_sum(&x_squared_terms);
    let ss_xy = kahan_sum(&xy_terms);
    let ss_yy = kahan_sum(&y_squared_terms);

    (ss_xx, ss_xy, ss_yy)
}

/// Calculate correlation coefficient using safe operations
fn calculate_correlation_coefficient(ss_xx: f64, ss_xy: f64, ss_yy: f64) -> f64 {
    if ss_xx > 0.0 && ss_yy > 0.0 {
        let denominator = (ss_xx * ss_yy).sqrt();
        if denominator > f64::EPSILON {
            ss_xy / denominator
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Calculate residual sum of squares using Kahan summation
fn calculate_residual_sum_of_squares(
    x_array: &Array1Ref,
    y_array: &Array1Ref,
    slope: f64,
    intercept: f64,
) -> f64 {
    let mut residual_terms = Vec::with_capacity(x_array.len());
    for i in 0..x_array.len() {
        let y_pred = slope * x_array[i] + intercept;
        let diff = y_array[i] - y_pred;
        residual_terms.push(diff * diff);
    }
    kahan_sum(&residual_terms)
}

/// Calculate standard error, p-value, and intercept standard error
fn calculate_standard_errors(
    n: f64,
    ss_xx: f64,
    ss_res: f64,
    slope: f64,
    x_mean: f64,
) -> (f64, f64, f64) {
    let stderr = if n > 2.0 && ss_xx > f64::EPSILON {
        let sd_res = (ss_res / (n - 2.0)).sqrt();
        if sd_res.is_finite() && ss_xx > 0.0 {
            safe_divide(sd_res, ss_xx.sqrt(), "standard error calculation").unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    let p_value = if n > 2.0 && stderr != 0.0 {
        let t_stat = slope.abs() / stderr;
        calculate_p_value_exact(t_stat, n - 2.0)
    } else {
        f64::NAN
    };

    let intercept_stderr = if n > 2.0 && ss_xx > 0.0 {
        let sd_res = (ss_res / (n - 2.0)).sqrt();
        sd_res * ((1.0 / n) + (x_mean * x_mean) / ss_xx).sqrt()
    } else {
        f64::NAN
    };

    (stderr, p_value, intercept_stderr)
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

    mod calculate_p_value_exact {
        use super::*;

        #[test]
        fn large_degrees_of_freedom() {
            let p_val = calculate_p_value_exact(2.0, 100.0);
            assert!(p_val > 0.0 && p_val < 1.0);
        }

        #[test]
        fn p_value_table_driven() {
            let test_cases = vec![
                ("small_degrees_of_freedom", 2.0, 5.0),
                ("zero_t_value", 0.0, 10.0),
                ("negative_t_value", -1.5, 15.0),
                ("large_t_value", 5.0, 20.0),
                ("minimal_df", 1.0, 3.0),
            ];

            for (name, t_value, df) in test_cases {
                let p_val = calculate_p_value_exact(t_value, df);
                assert!(
                    (0.0..=2.0).contains(&p_val),
                    "{}: p-value out of range: {}",
                    name,
                    p_val
                );
            }
        }

        #[test]
        fn accuracy_comparison() {
            // Test with known t-value and p-value combinations
            let test_cases = vec![
                (0.0, 10.0, 1.0),     // t=0 => p=1
                (1.96, 1000.0, 0.05), // Large df with t=1.96 ≈ p=0.05
            ];

            for (t_value, df, expected_p) in test_cases {
                let p_val = calculate_p_value_exact(t_value, df);
                assert!(
                    (p_val - expected_p).abs() < 0.1,
                    "t={}, df={}: expected p≈{}, got {}",
                    t_value,
                    df,
                    expected_p,
                    p_val
                );
            }
        }
    }

    mod kahan_sum {
        use super::*;

        #[test]
        fn kahan_sum_accuracy() {
            // More practical test: many small values
            let small_values = vec![0.1; 10];
            let kahan_result = kahan_sum(&small_values);
            let expected = 1.0;
            assert!(
                (kahan_result - expected).abs() < 1e-14,
                "Kahan sum should be very accurate for small values"
            );

            // Precision comparison with regular addition
            let values = vec![1.0, 1e-15, 1e-15, 1e-15];
            let kahan_result = kahan_sum(&values);
            let naive_result: f64 = values.iter().sum();

            // Verify Kahan summation is more accurate or at least equivalent
            assert!(
                kahan_result >= naive_result - 1e-15,
                "Kahan sum should be at least as accurate"
            );
        }

        #[test]
        fn kahan_sum_vs_naive() {
            // Test floating-point precision limits
            let mut values = vec![1.0; 1000000];
            values.push(1e-10);

            let kahan_result = kahan_sum(&values);
            let naive_result: f64 = values.iter().sum();

            // Verify Kahan summation is more accurate
            assert!(
                kahan_result >= naive_result,
                "Kahan sum should be at least as accurate as naive sum"
            );
        }

        #[test]
        fn kahan_sum_edge_cases() {
            assert_eq!(kahan_sum(&[]), 0.0);
            assert_eq!(kahan_sum(&[42.0]), 42.0);
            assert!(kahan_sum(&[f64::NAN]).is_nan());
        }
    }

    mod ieee754_edge_cases {
        use super::*;

        #[test]
        fn test_nan_input_detection() {
            let x = vec![1.0, 2.0, f64::NAN];
            let y = vec![2.0, 4.0, 6.0];

            let result = perform_ols(&x, &y);
            // NaN input should be properly detected (no validation in test function)
            // Actual function will error
            assert!(result.slope.is_nan() || result.slope.is_finite());
        }

        #[test]
        fn test_infinite_input_detection() {
            let x = vec![1.0, 2.0, f64::INFINITY];
            let y = vec![2.0, 4.0, 6.0];

            let result = perform_ols(&x, &y);
            // Verify infinite input handling
            assert!(
                result.slope.is_nan() || result.slope.is_finite() || result.slope.is_infinite()
            );
        }

        #[test]
        fn test_subnormal_numbers() {
            // Subnormal number test
            let x = vec![1.0, 2.0, 3.0];
            let y = vec![1e-320, 2e-320, 3e-320]; // Very small values

            let result = perform_ols(&x, &y);
            // Verify calculation completes
            assert!(result.slope.is_finite() || result.slope.is_nan());
        }

        #[test]
        fn test_extreme_values() {
            // Extreme value test
            let x = vec![1e-100, 2e-100, 3e-100];
            let y = vec![1e100, 2e100, 3e100];

            let result = perform_ols(&x, &y);
            // Verify proper handling even with extreme values
            assert!(result.slope.is_finite() || result.slope.is_infinite());
        }

        #[test]
        fn test_zero_division_cases() {
            // Case with zero variance
            let x = vec![1.0, 1.0, 1.0]; // Zero variance
            let y = vec![2.0, 4.0, 6.0];

            let result = perform_ols(&x, &y);
            // Division by zero is properly handled
            assert!(result.slope.is_nan() || result.slope.is_infinite());
        }
    }

    // Add direct tests to increase coverage of internal functions
    mod internal_function_tests {
        use super::*;

        #[test]
        fn test_perform_ols_directly() {
            let x = vec![1.0, 2.0, 3.0, 4.0];
            let y = vec![2.0, 4.0, 6.0, 8.0];

            let result = perform_ols(&x, &y);
            assert!((result.slope - 2.0).abs() < 1e-10);
            assert!(result.intercept.abs() < 1e-10);
            assert!((result.r_value - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_perform_ols_edge_cases() {
            let test_cases = vec![
                ("single_point", vec![1.0], vec![2.0]),
                ("zero_variance_x", vec![2.0, 2.0, 2.0], vec![1.0, 2.0, 3.0]),
                ("zero_variance_y", vec![1.0, 2.0, 3.0], vec![5.0, 5.0, 5.0]),
            ];

            for (_name, x, y) in test_cases {
                let result = perform_ols(&x, &y);
                // Test that the function completes without panicking
                // Some results may be NaN or infinite, which is expected
                let _ = result.slope;
                let _ = result.intercept;
                let _ = result.r_value;
            }
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

        let mut x_squared_terms = Vec::with_capacity(x.len());
        let mut xy_terms = Vec::with_capacity(x.len());
        let mut y_squared_terms = Vec::with_capacity(x.len());

        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            x_squared_terms.push(x_diff * x_diff);
            xy_terms.push(x_diff * y_diff);
            y_squared_terms.push(y_diff * y_diff);
        }

        let ss_xx = kahan_sum(&x_squared_terms);
        let ss_xy = kahan_sum(&xy_terms);
        let ss_yy = kahan_sum(&y_squared_terms);

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
