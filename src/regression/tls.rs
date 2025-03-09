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
    // NumPy配列をndarrayに変換
    let x_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        x.as_array().to_owned();
    let y_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        y.as_array().to_owned();

    // データの中心化（平均の減算）
    let x_mean = x_array.mean().unwrap_or(0.0);
    let y_mean = y_array.mean().unwrap_or(0.0);

    let x_centered = x_array.mapv(|v| v - x_mean);
    let y_centered = y_array.mapv(|v| v - y_mean);

    // データをnalgebra行列に変換
    let mut data_matrix = DMatrix::zeros(x_centered.len(), 2);
    for i in 0..x_centered.len() {
        data_matrix[(i, 0)] = x_centered[i];
        data_matrix[(i, 1)] = y_centered[i];
    }

    // SVDを実行
    let svd = SVD::new(data_matrix, true, true);

    // 右特異ベクトルVの取得（最後の特異値に対応する列）
    let v = if let Some(v) = svd.v_t {
        v
    } else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "SVD computation failed to return V matrix",
        ));
    };

    // 特異値の確認
    let singular_values = svd.singular_values;

    // 最小特異値のインデックスを見つける
    let min_singular_idx = (0..singular_values.len())
        .min_by(|&a, &b| {
            singular_values[a]
                .partial_cmp(&singular_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(singular_values.len() - 1);

    // 最小特異値に対応する右特異ベクトルを取得
    let v_col = v.row(min_singular_idx);

    // 0除算防止
    if v_col[1].abs() < 1e-10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Division by zero in TLS calculation: v[1] is too close to zero",
        ));
    }

    // 傾きと切片の計算
    let slope = -v_col[0] / v_col[1];
    let intercept = y_mean - slope * x_mean;

    // 相関係数の計算
    let r_value = compute_r_value(&x_array, &y_array);

    // 予測値の計算
    let y_pred = x_array.mapv(|v| slope * v + intercept);

    // 結果をPythonに返す
    Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
}

// 相関係数を計算するヘルパー関数
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
