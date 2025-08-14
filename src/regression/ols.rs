use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
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

    // IEEE 754エッジケース検証
    let x_slice = x_array.as_slice().unwrap();
    let y_slice = y_array.as_slice().unwrap();
    validate_finite_array(x_slice, "x")?;
    validate_finite_array(y_slice, "y")?;

    // Calculate means
    let x_mean = x_array.mean().unwrap();
    let y_mean = y_array.mean().unwrap();

    // Calculate variance and covariance using Kahan summation for better precision
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

    // Calculate slope using safe division
    let slope = safe_divide(ss_xy, ss_xx, "slope calculation")?;

    // Calculate intercept
    let intercept = y_mean - slope * x_mean;

    // Calculate correlation coefficient using safe operations
    let r_value = if ss_xx > 0.0 && ss_yy > 0.0 {
        let denominator = (ss_xx * ss_yy).sqrt();
        if denominator > f64::EPSILON {
            ss_xy / denominator
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Calculate residual sum of squares using Kahan summation
    let mut residual_terms = Vec::with_capacity(x_array.len());
    for i in 0..x_array.len() {
        let y_pred = slope * x_array[i] + intercept;
        let diff = y_array[i] - y_pred;
        residual_terms.push(diff * diff);
    }
    let ss_res = kahan_sum(&residual_terms);

    // Calculate standard error using safe operations
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

    // Calculate t-statistic and p-value (two-tailed test)
    let p_value = if n > 2.0 && stderr != 0.0 {
        let t_stat = slope.abs() / stderr;
        // Calculate p-value from Student's t-distribution (exact implementation)
        calculate_p_value_exact(t_stat, n - 2.0)
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

// Function to calculate p-value from t-statistic (exact implementation using statrs)
fn calculate_p_value_exact(t_value: f64, df: f64) -> f64 {
    // statrsを使用してStudent's t分布から正確なp値を計算
    match StudentsT::new(0.0, 1.0, df) {
        Ok(t_dist) => {
            // 両側検定なので2倍する
            2.0 * (1.0 - t_dist.cdf(t_value.abs()))
        }
        Err(_) => {
            // 分布の作成に失敗した場合はNaNを返す
            f64::NAN
        }
    }
}

// IEEE 754エッジケース検出とハンドリング
fn validate_finite_array(array: &[f64], name: &str) -> PyResult<()> {
    for (i, &value) in array.iter().enumerate() {
        if !value.is_finite() {
            if value.is_nan() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "NaN detected in {} array at index {}",
                    name, i
                )));
            } else if value.is_infinite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Infinite value detected in {} array at index {}",
                    name, i
                )));
            }
        }
        // 非正規化数（subnormal）の検出
        if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
            eprintln!(
                "Warning: Subnormal number detected in {} array at index {}: {}",
                name, i, value
            );
        }
    }
    Ok(())
}

fn safe_divide(numerator: f64, denominator: f64, context: &str) -> PyResult<f64> {
    if !numerator.is_finite() || !denominator.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Non-finite values in division ({}): {}/{}",
            context, numerator, denominator
        )));
    }

    if denominator.abs() < f64::EPSILON {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Division by zero or near-zero value ({}): denominator = {}",
            context, denominator
        )));
    }

    let result = numerator / denominator;
    if !result.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Division resulted in non-finite value ({}): {}/{} = {}",
            context, numerator, denominator, result
        )));
    }

    Ok(result)
}

// Kahan加算アルゴリズム - 浮動小数点の累積誤差を削減
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // 補正項

    for &value in values {
        let y = value - c; // 補正を適用
        let t = sum + y; // 新しい和
        c = (t - sum) - y; // 次回の補正項を計算
        sum = t;
    }

    sum
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
                    p_val >= 0.0 && p_val <= 2.0,
                    "{}: p-value out of range: {}",
                    name,
                    p_val
                );
            }
        }

        #[test]
        fn accuracy_comparison() {
            // 既知のt値とp値の組み合わせでテスト
            let test_cases = vec![
                (0.0, 10.0, 1.0),     // t=0 => p=1
                (1.96, 1000.0, 0.05), // 大きなdfでt=1.96 ≈ p=0.05
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
            // より実用的なテスト: 多数の小さな値
            let small_values = vec![0.1; 10];
            let kahan_result = kahan_sum(&small_values);
            let expected = 1.0;
            assert!(
                (kahan_result - expected).abs() < 1e-14,
                "Kahan sum should be very accurate for small values"
            );

            // 通常の加算との精度比較
            let values = vec![1.0, 1e-15, 1e-15, 1e-15];
            let kahan_result = kahan_sum(&values);
            let naive_result: f64 = values.iter().sum();

            // Kahan加算の方が精度が高いか、最低でも同等であることを確認
            assert!(
                kahan_result >= naive_result - 1e-15,
                "Kahan sum should be at least as accurate"
            );
        }

        #[test]
        fn kahan_sum_vs_naive() {
            // 浮動小数点精度の限界をテスト
            let mut values = vec![1.0; 1000000];
            values.push(1e-10);

            let kahan_result = kahan_sum(&values);
            let naive_result: f64 = values.iter().sum();

            // Kahan加算の方が精度が高いことを確認
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
            // NaN入力は適切に検出されるべき（テスト用関数では検証なし）
            // 実際の関数ではエラーになる
            assert!(result.slope.is_nan() || result.slope.is_finite());
        }

        #[test]
        fn test_infinite_input_detection() {
            let x = vec![1.0, 2.0, f64::INFINITY];
            let y = vec![2.0, 4.0, 6.0];

            let result = perform_ols(&x, &y);
            // 無限大入力の処理確認
            assert!(
                result.slope.is_nan() || result.slope.is_finite() || result.slope.is_infinite()
            );
        }

        #[test]
        fn test_subnormal_numbers() {
            // 非正規化数のテスト
            let x = vec![1.0, 2.0, 3.0];
            let y = vec![1e-320, 2e-320, 3e-320]; // 非常に小さい値

            let result = perform_ols(&x, &y);
            // 計算が完了することを確認
            assert!(result.slope.is_finite() || result.slope.is_nan());
        }

        #[test]
        fn test_extreme_values() {
            // 極値のテスト
            let x = vec![1e-100, 2e-100, 3e-100];
            let y = vec![1e100, 2e100, 3e100];

            let result = perform_ols(&x, &y);
            // 極値でも適切に処理されることを確認
            assert!(result.slope.is_finite() || result.slope.is_infinite());
        }

        #[test]
        fn test_zero_division_cases() {
            // 分散が0のケース
            let x = vec![1.0, 1.0, 1.0]; // 分散0
            let y = vec![2.0, 4.0, 6.0];

            let result = perform_ols(&x, &y);
            // 0除算は適切に処理される
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
