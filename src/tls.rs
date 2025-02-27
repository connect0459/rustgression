use ndarray::{Array1, Axis}; // 未使用の`s`を削除
use ndarray_linalg::SVD;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1}; // IntoPyArrayを追加
use pyo3::prelude::*;
use std::f64;

/// Rust実装のTotal Least Squares回帰
///
/// Parameters
/// ----------
/// x: numpy.ndarray
///     x軸データ
/// y: numpy.ndarray
///     y軸データ
///
/// Returns
/// -------
/// tuple
///     (傾き, 切片, 相関係数)のタプル
#[pyfunction]
pub fn calculate_tls_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, f64, f64, f64)> {
    // numpy配列をndarray::Array1に変換
    let x_array = x.as_array().to_owned();
    let y_array = y.as_array().to_owned();

    // データの標準化
    let x_mean = x_array.mean().unwrap();
    let y_mean = y_array.mean().unwrap();
    let x_std = x_array.std(0.0);
    let y_std = y_array.std(0.0);

    let x_standardized = x_array.mapv(|v| (v - x_mean) / x_std);
    let y_standardized = y_array.mapv(|v| (v - y_mean) / y_std);

    // データ行列を作成
    let data_matrix = ndarray::stack![Axis(0), x_standardized.view(), y_standardized.view()];

    // SVD計算
    let (_, _, v_t) = data_matrix.svd(false, true).unwrap();
    let v_t = v_t.unwrap();
    let v = v_t.row(v_t.nrows() - 1);

    // 相関係数を計算
    let r_value = compute_correlation(&x_array, &y_array);

    // 傾きを計算
    let mut slope_standardized = -v[0] / v[1];

    // 相関係数の符号と一致するように傾きの符号を調整
    if (slope_standardized.signum() != r_value.signum()) && (r_value != 0.0) {
        slope_standardized = -slope_standardized;
    }

    // 元のスケールに戻す
    let slope = slope_standardized * (y_std / x_std);
    let intercept = y_mean - slope * x_mean;

    // 傾きと切片から予測値を計算
    let y_pred = x_array.mapv(|v| slope * v + intercept);

    // 結果をPythonに返す
    Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
}

// 相関係数を計算する補助関数
fn compute_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let _n = x.len() as f64;
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
