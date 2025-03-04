use ndarray::{Array1, Axis};
use ndarray_linalg::SVD;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
    let x_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = x.as_array().to_owned();
    let y_array: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = y.as_array().to_owned();

    // データの中心化（平均を引く）
    let x_mean: f64 = x_array.mean().unwrap();
    let y_mean: f64 = y_array.mean().unwrap();
    
    let x_centered: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = 
        x_array.mapv(|v: f64| v - x_mean);
    let y_centered: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = 
        y_array.mapv(|v: f64| v - y_mean);

    // データ行列を作成 - 中心化したデータで行列を作成
    let data_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> = 
        ndarray::stack![Axis(1), x_centered.view(), y_centered.view()];
    
    // SVD計算
    let (_, s, v_t) = data_matrix.svd(true, true).unwrap();
    let v_t = v_t.expect("SVD failed to compute V^T matrix");
    
    // 最小特異値に対応する特異ベクトルを見つける
    // 修正: 最小特異値のインデックスを見つける 
    let min_singular_idx = s.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    let v: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 1]>> = 
        v_t.row(min_singular_idx);

    // TLSの傾きを計算: -v_x / v_y (最小特異値に対応する特異ベクトルの要素から)
    let slope: f64 = -v[0] / v[1];
    
    // 切片を計算
    let intercept: f64 = y_mean - slope * x_mean;
    
    // 相関係数を計算
    let r_value: f64 = compute_correlation(&x_array, &y_array);
    
    // 傾きと切片から予測値を計算
    let y_pred: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> = 
        x_array.mapv(|v| slope * v + intercept);
    
    // 結果をPythonに返す
    Ok((y_pred.into_pyarray(py), slope, intercept, r_value))
}

// 相関係数を計算する補助関数
fn compute_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
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