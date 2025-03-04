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

    // Perform SVD
    let (_, s, v_t) = data_matrix.svd(true, true).unwrap();
    let v_t: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
        v_t.expect("SVD failed to compute V^T matrix");

    // Find the singular vector corresponding to the smallest singular value
    // Fix: Find the index of the smallest singular value
    let min_singular_idx: usize = s
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    let v: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>> =
        v_t.row(min_singular_idx);

    // Calculate the slope of TLS: -v_x / v_y (from the elements of the singular vector corresponding to the smallest singular value)
    let slope: f64 = -v[0] / v[1];

    // Calculate the intercept
    let intercept: f64 = y_mean - slope * x_mean;

    // Calculate the correlation coefficient
    let r_value: f64 = compute_r_value(&x_array, &y_array);

    // Calculate predicted values from slope and intercept
    let y_pred: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x_array.mapv(|v| slope * v + intercept);

    // Return results to Python
    Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
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
