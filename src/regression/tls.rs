use ndarray::{Array1, Axis};
use ndarray_linalg::SVD;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

/// Total Least Squares regression implemented in Rust.
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
///     A tuple containing (slope, intercept, r_value).
#[pyfunction]
pub fn calculate_tls_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, f64, f64, f64)> {
    // Convert numpy arrays to ndarray::Array1
    let x_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x.as_array().to_owned();
    let y_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        y.as_array().to_owned();

    // Center the data (subtract the mean)
    let x_mean: f64 = x_array.mean().unwrap();
    let y_mean: f64 = y_array.mean().unwrap();

    let x_centered: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x_array.mapv(|v: f64| v - x_mean);
    let y_centered: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        y_array.mapv(|v: f64| v - y_mean);

    // Create data matrix - stack centered data into a matrix
    let data_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
        ndarray::stack![Axis(1), x_centered.view(), y_centered.view()];

    // SVDの実行とエラーハンドリングの改善
    let svd_result = data_matrix.svd(true, true);
    
    match svd_result {
        Ok((_, s, v_t_opt)) => {
            if let Some(v_t) = v_t_opt {
                // 最小特異値に対応する特異ベクトルを見つける
                let min_singular_idx = s
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(s.len() - 1);

                let v = v_t.row(min_singular_idx);

                // Ensure we don't divide by zero
                if v[1].abs() < 1e-10 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Division by zero in TLS calculation: v[1] is too close to zero",
                    ));
                }

                // Calculate slope and intercept
                let slope = -v[0] / v[1];
                let intercept = y_mean - slope * x_mean;

                // Calculate correlation
                let r_value = compute_r_value(&x_array, &y_array);

                // Calculate predicted values
                let y_pred = x_array.mapv(|v| slope * v + intercept);

                Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "SVD computation failed to return V matrix",
                ))
            }
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "SVD computation failed: {}",
            e
        ))),
    }
}

// Helper function to compute the correlation coefficient
fn compute_r_value(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        sum_xy += x_diff * y_diff;
        sum_x2 += x_diff * x_diff;
        sum_y2 += y_diff * y_diff;
    }

    if sum_x2 == 0.0 || sum_y2 == 0.0 {
        return 0.0;
    }

    sum_xy / (sum_x2.sqrt() * sum_y2.sqrt())
}
