use nalgebra::{DMatrix, SVD};
use ndarray::Array1;
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
///     A tuple containing (predicted values, slope, intercept, r_value).
#[pyfunction]
pub fn calculate_tls_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, f64, f64, f64)> {
    // Convert NumPy array to ndarray
    let x_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x.as_array().to_owned();
    let y_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        y.as_array().to_owned();

    // Center the data (subtract the mean)
    let x_mean = x_array.mean().unwrap_or(0.0);
    let y_mean = y_array.mean().unwrap_or(0.0);

    let x_centered: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 1]>> =
        x_array.mapv(|v| v - x_mean);
    let y_centered: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 1]>> =
        y_array.mapv(|v| v - y_mean);

    // Convert data to nalgebra matrix
    let mut data_matrix: nalgebra::Matrix<
        _,
        nalgebra::Dyn,
        nalgebra::Dyn,
        nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Dyn>,
    > = DMatrix::zeros(x_centered.len(), 2);
    for i in 0..x_centered.len() {
        data_matrix[(i, 0)] = x_centered[i];
        data_matrix[(i, 1)] = y_centered[i];
    }

    // Perform SVD
    let svd: SVD<_, nalgebra::Dyn, nalgebra::Dyn> = SVD::new(data_matrix, true, true);

    // Get the right singular vector V (corresponding to the last singular value)
    let v = if let Some(v) = svd.v_t {
        v
    } else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SVD computation failed to return V matrix",
        ));
    };

    // Check singular values
    let singular_values: nalgebra::Matrix<
        _,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::VecStorage<_, nalgebra::Dyn, nalgebra::Const<1>>,
    > = svd.singular_values;

    // Find the index of the minimum singular value
    let min_singular_idx = (0..singular_values.len())
        .min_by(|&a, &b| {
            singular_values[a]
                .partial_cmp(&singular_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(singular_values.len() - 1);

    // Get the right singular vector corresponding to the minimum singular value
    let v_col = v.row(min_singular_idx);

    // Prevent division by zero
    if v_col[1].abs() < 1e-10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Division by zero in TLS calculation: v[1] is too close to zero",
        ));
    }

    // Calculate the correlation coefficient and its sign
    let r_value = compute_r_value(&x_array, &y_array);
    let r_sign = r_value.signum();

    // Calculate slope and intercept
    // Adjust the slope direction based on the sign of the correlation coefficient
    let slope = -v_col[0] / v_col[1] * r_sign;
    let intercept = y_mean - slope * x_mean;

    // Calculate predicted values
    let y_pred: ndarray::ArrayBase<ndarray::OwnedRepr<_>, ndarray::Dim<[usize; 1]>> =
        x_array.mapv(|v| slope * v + intercept);

    // Return results to Python
    Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
}

// Helper function to calculate the correlation coefficient
fn compute_r_value(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

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
