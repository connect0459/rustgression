use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::f64;

/// Rust実装のOrdinary Least Squares回帰（statsのlinregressと同様）
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
///     (予測値, 傾き, 切片, 相関係数, p値, 標準誤差)のタプル
#[pyfunction]
pub fn calculate_ols_regression<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, f64, f64, f64, f64, f64)> {
    // numpy配列をndarrayに変換
    let x_array = x.as_array().to_owned();
    let y_array = y.as_array().to_owned();
    let n = x_array.len() as f64;

    if n < 2.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "At least 2 data points are required for regression",
        ));
    }

    // 平均を計算
    let x_mean = x_array.mean().unwrap();
    let y_mean = y_array.mean().unwrap();

    // 分散と共分散を計算
    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..x_array.len() {
        let x_diff = x_array[i] - x_mean;
        let y_diff = y_array[i] - y_mean;

        ss_xx += x_diff * x_diff;
        ss_xy += x_diff * y_diff;
        ss_yy += y_diff * y_diff;
    }

    // 傾きを計算
    let slope = ss_xy / ss_xx;

    // 切片を計算
    let intercept = y_mean - slope * x_mean;

    // 相関係数を計算
    let r_value = if ss_xx * ss_yy > 0.0 {
        ss_xy / (ss_xx.sqrt() * ss_yy.sqrt())
    } else {
        0.0
    };

    // 残差平方和を計算
    let mut ss_res = 0.0;
    for i in 0..x_array.len() {
        let y_pred = slope * x_array[i] + intercept;
        let diff = y_array[i] - y_pred;
        ss_res += diff * diff;
    }

    // 標準誤差を計算
    let stderr = if n > 2.0 && ss_xx > 0.0 {
        let sd_res = (ss_res / (n - 2.0)).sqrt();
        sd_res / ss_xx.sqrt()
    } else {
        f64::NAN
    };

    // t統計量とp値の計算（二側検定）
    let p_value = if n > 2.0 && stderr != 0.0 {
        let t_stat = slope.abs() / stderr;
        // Student's t分布のp値計算（近似値）
        calculate_p_value(t_stat, n - 2.0)
    } else {
        f64::NAN
    };

    // 予測値の計算
    let y_pred = x_array.mapv(|v| slope * v + intercept);

    // 結果をPythonに返す
    Ok((
        y_pred.into_pyarray(py),
        slope,
        intercept,
        r_value,
        p_value,
        stderr,
    ))
}

// t統計量からp値を計算する関数（簡易実装）
fn calculate_p_value(t_value: f64, df: f64) -> f64 {
    // 自由度が大きい場合は正規分布を使用した近似
    if df > 30.0 {
        let x = t_value / (df.sqrt());
        2.0 * (1.0 - normal_cdf(x.abs()))
    } else {
        // 小さい自由度の場合は簡易近似を使用
        // 実際の実装ではより正確なt分布のCDFが必要
        let x = df / (df + t_value * t_value);
        // ベータ関数の近似のため、不完全なp値
        let p = 1.0 - x.powf(df / 2.0);
        2.0 * p
    }
}

// 標準正規分布のCDF（累積分布関数）
// 簡易実装のため精度は低い
fn normal_cdf(x: f64) -> f64 {
    // 誤差関数の近似
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 {
        1.0 - prob
    } else {
        prob
    }
}
