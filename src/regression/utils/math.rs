/// Kahan summation algorithm - reduces floating-point accumulation errors
///
/// Parameters
/// ----------
/// values : &[f64]
///     Array of values to sum
///
/// Returns
/// -------
/// f64
///     High-precision computed sum
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
