/// Kahan summation algorithm - 浮動小数点数の累積誤差を軽減
///
/// Parameters
/// ----------
/// values : &[f64]
///     合計したい値の配列
///
/// Returns
/// -------
/// f64
///     高精度で計算された合計値
pub fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation term

    for &value in values {
        let y = value - c; // Apply compensation
        let t = sum + y; // New sum
        c = (t - sum) - y; // Calculate next compensation term
        sum = t;
    }

    sum
}
