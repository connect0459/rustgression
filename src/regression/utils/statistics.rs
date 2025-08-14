use crate::regression::utils::math::kahan_sum;
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::f64;

/// Calculate correlation coefficient using Kahan summation
///
/// Parameters
/// ----------
/// x : &Array1<f64>
///     Data for the x-axis
/// y : &Array1<f64>
///     Data for the y-axis
///
/// Returns
/// -------
/// f64
///     Correlation coefficient (-1.0 to 1.0 range)
pub fn compute_r_value(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut xy_terms = Vec::with_capacity(x.len());
    let mut x2_terms = Vec::with_capacity(x.len());
    let mut y2_terms = Vec::with_capacity(x.len());

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        xy_terms.push(x_diff * y_diff);
        x2_terms.push(x_diff * x_diff);
        y2_terms.push(y_diff * y_diff);
    }

    let sum_xy = kahan_sum(&xy_terms);
    let sum_x2 = kahan_sum(&x2_terms);
    let sum_y2 = kahan_sum(&y2_terms);

    if sum_x2 == 0.0 || sum_y2 == 0.0 {
        return 0.0;
    }

    sum_xy / (sum_x2.sqrt() * sum_y2.sqrt())
}

/// Calculate p-value from t-statistic using Student's t-distribution (exact implementation)
///
/// Parameters
/// ----------
/// t_value : f64
///     t-statistic
/// df : f64
///     Degrees of freedom
///
/// Returns
/// -------
/// f64
///     Two-tailed test p-value
pub fn calculate_p_value_exact(t_value: f64, df: f64) -> f64 {
    // Calculate exact p-value from Student's t-distribution using statrs
    match StudentsT::new(0.0, 1.0, df) {
        Ok(t_dist) => {
            // Double for two-sided test
            2.0 * (1.0 - t_dist.cdf(t_value.abs()))
        }
        Err(_) => {
            // Return NaN if distribution creation failed
            f64::NAN
        }
    }
}
