use crate::regression::utils::{kahan_sum, validation::validate_finite_array};
use nalgebra::{DMatrix, DVector};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

type OlsMultiResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
    f64,
    f64,
);

/// Rust implementation of multiple Ordinary Least Squares regression.
///
/// Parameters
/// ----------
/// x : numpy.ndarray, shape (n, p)
///     Predictor matrix.
/// y : numpy.ndarray, shape (n,)
///     Response vector.
///
/// Returns
/// -------
/// tuple
///     (y_pred, coefficients, r_squared, f_statistic, p_value)
///     coefficients is shape (p+1,) with intercept first.
#[pyfunction]
pub fn calculate_ols_multi_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<OlsMultiResult<'py>> {
    let x_view = x.as_array();
    let y_array = y.as_array().to_owned();

    let (n, p) = x_view.dim();

    let y_slice = y_array.as_slice().unwrap();
    validate_finite_array(py, y_slice, "y")?;

    let x_standard = x_view.as_standard_layout();
    let x_flat = x_standard.as_slice().unwrap();
    validate_finite_array(py, x_flat, "x")?;

    let result = compute_ols_multi_coefficients(x_flat, y_slice, n, p)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((
        result.y_pred.into_pyarray(py),
        result.coefficients.into_pyarray(py),
        result.r_squared,
        result.f_statistic,
        result.p_value,
    ))
}

struct OlsMultiOutput {
    y_pred: Vec<f64>,
    coefficients: Vec<f64>,
    r_squared: f64,
    f_statistic: f64,
    p_value: f64,
}

fn compute_ols_multi_coefficients(
    x_flat: &[f64],
    y_slice: &[f64],
    n: usize,
    p: usize,
) -> Result<OlsMultiOutput, String> {
    if n <= p {
        return Err("Number of observations n must exceed number of predictors p".to_string());
    }

    let x_mat = DMatrix::from_row_slice(n, p, x_flat);
    let mut x_aug = DMatrix::zeros(n, p + 1);
    for i in 0..n {
        x_aug[(i, 0)] = 1.0;
        for j in 0..p {
            x_aug[(i, j + 1)] = x_mat[(i, j)];
        }
    }

    let y_vec = DVector::from_column_slice(y_slice);

    let svd = x_aug.clone().svd(true, true);
    let max_sv = svd.singular_values.max();
    let rank_threshold = max_sv * (n.max(p + 1)) as f64 * f64::EPSILON;
    let min_sv = svd.singular_values.min();
    if max_sv == 0.0 || min_sv < rank_threshold {
        return Err("Collinear predictors: X matrix is rank-deficient".to_string());
    }

    let beta = svd
        .solve(&y_vec, rank_threshold)
        .map_err(|e| format!("Failed to compute coefficients: {}", e))?;

    let y_pred_vec = &x_aug * &beta;
    let y_pred: Vec<f64> = y_pred_vec.iter().cloned().collect();

    let y_mean = y_slice.iter().sum::<f64>() / n as f64;
    let ss_tot_terms: Vec<f64> = y_slice.iter().map(|&yi| (yi - y_mean).powi(2)).collect();
    let ss_tot = kahan_sum(&ss_tot_terms);
    let ss_res_terms: Vec<f64> = y_slice
        .iter()
        .zip(y_pred.iter())
        .map(|(&yi, &yp)| (yi - yp).powi(2))
        .collect();
    let ss_res = kahan_sum(&ss_res_terms);

    let r_squared = if ss_tot > f64::EPSILON {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    };

    let df1 = p as f64;
    let df2 = (n - p - 1) as f64;
    let ss_mod = ss_tot - ss_res;

    let f_statistic = if df1 <= 0.0 || df2 <= 0.0 || ss_tot <= f64::EPSILON {
        f64::NAN
    } else if ss_res <= f64::EPSILON {
        f64::INFINITY
    } else {
        (ss_mod / df1) / (ss_res / df2)
    };

    let p_value = if f_statistic.is_nan() {
        f64::NAN
    } else if f_statistic.is_infinite() {
        0.0
    } else if f_statistic >= 0.0 && df1 > 0.0 && df2 > 0.0 {
        match FisherSnedecor::new(df1, df2) {
            Ok(f_dist) => 1.0 - f_dist.cdf(f_statistic),
            Err(_) => f64::NAN,
        }
    } else {
        f64::NAN
    };

    Ok(OlsMultiOutput {
        y_pred,
        coefficients: beta.iter().cloned().collect(),
        r_squared,
        f_statistic,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OlsMultiTestResult {
        coefficients: Vec<f64>,
        r_squared: f64,
        f_statistic: f64,
        p_value: f64,
    }

    fn perform_ols_multi(x_rows: &[Vec<f64>], y: &[f64]) -> Result<OlsMultiTestResult, String> {
        let n = x_rows.len();
        let p = if n > 0 { x_rows[0].len() } else { 0 };

        let mut x_flat = Vec::with_capacity(n * p);
        for row in x_rows {
            x_flat.extend_from_slice(row);
        }

        let result = compute_ols_multi_coefficients(&x_flat, y, n, p)?;
        Ok(OlsMultiTestResult {
            coefficients: result.coefficients,
            r_squared: result.r_squared,
            f_statistic: result.f_statistic,
            p_value: result.p_value,
        })
    }

    mod calculate_ols_multi_regression {
        use super::*;

        #[test]
        fn returns_correct_coefficients_for_two_predictor_synthetic_data() {
            let x_rows = vec![
                vec![1.0, 1.0],
                vec![1.0, 2.0],
                vec![2.0, 1.0],
                vec![2.0, 2.0],
                vec![3.0, 1.0],
                vec![3.0, 2.0],
            ];
            let y: Vec<f64> = x_rows
                .iter()
                .map(|row| 5.0 + 2.0 * row[0] + 3.0 * row[1])
                .collect();

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert!(
                (result.coefficients[0] - 5.0).abs() < 1e-8,
                "intercept mismatch: {}",
                result.coefficients[0]
            );
            assert!(
                (result.coefficients[1] - 2.0).abs() < 1e-8,
                "coeff x1 mismatch: {}",
                result.coefficients[1]
            );
            assert!(
                (result.coefficients[2] - 3.0).abs() < 1e-8,
                "coeff x2 mismatch: {}",
                result.coefficients[2]
            );
        }

        #[test]
        fn returns_r_squared_of_one_for_perfect_linear_relationship() {
            let x_rows = vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![2.0, 1.0],
                vec![1.0, 2.0],
                vec![3.0, 2.0],
            ];
            let y: Vec<f64> = x_rows
                .iter()
                .map(|row| 1.0 + 2.0 * row[0] + 3.0 * row[1])
                .collect();

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert!(
                (result.r_squared - 1.0).abs() < 1e-8,
                "r_squared should be 1.0 for perfect fit, got {}",
                result.r_squared
            );
        }

        #[test]
        fn returns_error_for_collinear_predictors() {
            let x_rows = vec![
                vec![1.0, 2.0],
                vec![2.0, 4.0],
                vec![3.0, 6.0],
                vec![4.0, 8.0],
                vec![5.0, 10.0],
            ];
            let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

            let result = perform_ols_multi(&x_rows, &y);

            assert!(
                result.is_err(),
                "should return error for collinear predictors"
            );
        }

        #[test]
        fn returns_error_when_observations_do_not_exceed_predictors() {
            let x_flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let y = vec![1.0, 2.0];

            let result = compute_ols_multi_coefficients(&x_flat, &y, 2, 3);

            assert!(result.is_err(), "should return error when n <= p");
        }

        #[test]
        fn returns_f_statistic_greater_than_zero_for_significant_predictors() {
            let x_rows: Vec<Vec<f64>> = [1.0_f64, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|&x1| {
                    [1.0_f64, 2.0, 3.0, 4.0, 5.0]
                        .iter()
                        .map(move |&x2| vec![x1, x2])
                })
                .collect();
            let y: Vec<f64> = x_rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    3.0 + 1.5 * row[0] + 0.8 * row[1] + if i % 2 == 0 { 0.1 } else { -0.1 }
                })
                .collect();

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert!(
                result.f_statistic > 0.0,
                "F-statistic should be positive, got {}",
                result.f_statistic
            );
        }

        #[test]
        fn returns_p_value_in_valid_range() {
            let x_rows: Vec<Vec<f64>> = [1.0_f64, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|&x1| {
                    [1.0_f64, 2.0, 3.0, 4.0, 5.0]
                        .iter()
                        .map(move |&x2| vec![x1, x2])
                })
                .collect();
            let y: Vec<f64> = x_rows
                .iter()
                .enumerate()
                .map(|(i, row)| 2.0 + row[0] + 0.5 * row[1] + if i % 2 == 0 { 0.1 } else { -0.1 })
                .collect();

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert!(
                result.p_value >= 0.0 && result.p_value <= 1.0,
                "p_value should be in [0, 1], got {}",
                result.p_value
            );
        }

        #[test]
        fn coefficients_vector_has_length_equal_to_predictors_plus_one() {
            let x_rows = vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.5],
                vec![1.5, 3.0, 4.5],
                vec![2.0, 4.0, 6.5],
            ];
            let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert_eq!(
                result.coefficients.len(),
                4,
                "coefficients length should be p+1 = 4"
            );
        }

        #[test]
        fn returns_low_p_value_for_data_with_strong_linear_signal() {
            let x_rows: Vec<Vec<f64>> = [1.0_f64, 2.0, 3.0, 4.0, 5.0]
                .iter()
                .flat_map(|&x1| {
                    [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]
                        .iter()
                        .map(move |&x2| vec![x1, x2])
                })
                .collect();
            let y: Vec<f64> = x_rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    10.0 + 2.5 * row[0] + 1.5 * row[1] + if i % 2 == 0 { 0.1 } else { -0.1 }
                })
                .collect();

            let result = perform_ols_multi(&x_rows, &y).unwrap();

            assert!(
                result.p_value < 0.05,
                "p_value should be small for strong signal, got {}",
                result.p_value
            );
        }
    }
}
