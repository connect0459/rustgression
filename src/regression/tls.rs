use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;
use svdrs::SVD;

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
    let x_array: Array1<f64> = x.as_array().to_owned();
    let y_array: Array1<f64> = y.as_array().to_owned();

    // データの中心化（平均の減算）
    let x_mean = x_array.mean().unwrap_or(0.0);
    let y_mean = y_array.mean().unwrap_or(0.0);

    let x_centered = x_array.mapv(|v| v - x_mean);
    let y_centered = y_array.mapv(|v| v - y_mean);

    // データ行列の作成 - 中心化したデータを行列としてスタック
    let mut data_matrix = Array2::zeros((x_centered.len(), 2));
    for i in 0..x_centered.len() {
        data_matrix[[i, 0]] = x_centered[i];
        data_matrix[[i, 1]] = y_centered[i];
    }

    // svdrsを使用したSVD計算
    let svd = SVD::new();

    // SVD計算を実行
    // data_matrixをフラット化して計算
    let rows = data_matrix.shape()[0];
    let cols = data_matrix.shape()[1];
    let flat_data: Vec<f64> = data_matrix.iter().copied().collect();

    let (u, s, v) = match svd.svd_flat(flat_data.as_slice(), rows, cols) {
        Ok(result) => result,
        Err(e) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "SVD calculation failed: {}",
                e
            )));
        }
    };

    // 最小特異値のインデックスを見つける
    let min_singular_idx = s
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    // 最小特異値に対応する右特異ベクトルを取得
    // V行列は転置されていないので、列ベクトルとして取得
    let v_col1 = v[min_singular_idx];
    let v_col2 = v[min_singular_idx + v.len() / 2]; // V行列は横に並んでいるので、次の列の値はv.len()/2離れている

    // 0除算防止
    if v_col2.abs() < 1e-10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Division by zero in TLS calculation",
        ));
    }

    // 傾きと切片の計算
    let slope = -v_col1 / v_col2;
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
