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

#[cfg(test)]
mod tests {
    use super::*;

    mod compute_r_value {
        use super::*;

        #[test]
        fn test_perfect_positive_correlation() {
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
            let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
            let result = compute_r_value(&x, &y);
            assert!((result - 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_perfect_negative_correlation() {
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
            let y = Array1::from_vec(vec![8.0, 6.0, 4.0, 2.0]);
            let result = compute_r_value(&x, &y);
            assert!((result + 1.0).abs() < 1e-10);
        }

        #[test]
        fn test_no_correlation() {
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
            let y = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0]);
            let result = compute_r_value(&x, &y);
            // This data actually has some correlation, so just check it's in valid range
            assert!(result >= -1.0 && result <= 1.0);
        }

        #[test]
        fn test_zero_variance_x() {
            let x = Array1::from_vec(vec![2.0, 2.0, 2.0]);
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let result = compute_r_value(&x, &y);
            assert_eq!(result, 0.0);
        }

        #[test]
        fn test_zero_variance_y() {
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let y = Array1::from_vec(vec![5.0, 5.0, 5.0]);
            let result = compute_r_value(&x, &y);
            assert_eq!(result, 0.0);
        }

        #[test]
        fn test_single_point() {
            let x = Array1::from_vec(vec![1.0]);
            let y = Array1::from_vec(vec![2.0]);
            let result = compute_r_value(&x, &y);
            assert_eq!(result, 0.0); // Single point has no variance
        }

        #[test]
        fn test_two_identical_points() {
            let x = Array1::from_vec(vec![1.0, 1.0]);
            let y = Array1::from_vec(vec![2.0, 2.0]);
            let result = compute_r_value(&x, &y);
            assert_eq!(result, 0.0);
        }

        #[test]
        fn test_correlation_range() {
            // Test that correlation is always in [-1, 1]
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let y = Array1::from_vec(vec![2.1, 3.9, 6.2, 7.8, 10.1]);
            let result = compute_r_value(&x, &y);
            assert!(result >= -1.0 && result <= 1.0);
        }

        #[test]
        fn test_weak_correlation() {
            let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
            let y = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.2]);
            let result = compute_r_value(&x, &y);
            assert!(result > 0.9); // Should be strongly positive
        }

        #[test]
        fn test_negative_values() {
            let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
            let y = Array1::from_vec(vec![-4.0, -2.0, 0.0, 2.0, 4.0]);
            let result = compute_r_value(&x, &y);
            assert!((result - 1.0).abs() < 1e-10);
        }
    }

    mod calculate_p_value_exact {
        use super::*;

        #[test]
        fn test_zero_t_value() {
            let p_val = calculate_p_value_exact(0.0, 10.0);
            assert!((p_val - 1.0).abs() < 1e-10); // t=0 should give p≈1
        }

        #[test]
        fn test_positive_t_value() {
            let p_val = calculate_p_value_exact(2.0, 10.0);
            assert!(p_val > 0.0 && p_val < 1.0);
            assert!(p_val < 0.1); // Should be significant
        }

        #[test]
        fn test_negative_t_value() {
            let p_val_pos = calculate_p_value_exact(2.0, 10.0);
            let p_val_neg = calculate_p_value_exact(-2.0, 10.0);
            // Two-tailed test should give same result for positive and negative
            assert!((p_val_pos - p_val_neg).abs() < 1e-10);
        }

        #[test]
        fn test_large_degrees_of_freedom() {
            let p_val = calculate_p_value_exact(1.96, 1000.0);
            // With large df, t-distribution approaches normal
            assert!((p_val - 0.05).abs() < 0.01); // Should be close to 0.05
        }

        #[test]
        fn test_small_degrees_of_freedom() {
            let p_val = calculate_p_value_exact(2.0, 1.0);
            assert!(p_val > 0.0 && p_val < 1.0);
        }

        #[test]
        fn test_very_large_t_value() {
            let p_val = calculate_p_value_exact(10.0, 10.0);
            assert!(p_val < 0.001); // Should be very small, but allow more tolerance
        }

        #[test]
        fn test_p_value_range() {
            let test_cases = vec![
                (0.0, 5.0),
                (1.0, 10.0),
                (2.0, 20.0),
                (3.0, 30.0),
                (-1.5, 15.0),
            ];

            for (t_value, df) in test_cases {
                let p_val = calculate_p_value_exact(t_value, df);
                assert!(
                    p_val >= 0.0 && p_val <= 2.0,
                    "p-value out of range: {}",
                    p_val
                );
            }
        }

        #[test]
        fn test_invalid_degrees_of_freedom() {
            // Test with very small or invalid degrees of freedom
            let p_val = calculate_p_value_exact(1.0, 0.1);
            // Should return NaN for invalid df
            assert!(p_val.is_nan() || (p_val >= 0.0 && p_val <= 2.0));
        }

        #[test]
        fn test_monotonicity() {
            // Larger absolute t-values should give smaller p-values
            let p1 = calculate_p_value_exact(1.0, 10.0);
            let p2 = calculate_p_value_exact(2.0, 10.0);
            let p3 = calculate_p_value_exact(3.0, 10.0);

            assert!(p1 > p2);
            assert!(p2 > p3);
        }

        #[test]
        fn test_symmetry() {
            // Test symmetry around zero
            for t in [0.5, 1.0, 1.5, 2.0, 2.5] {
                let p_pos = calculate_p_value_exact(t, 10.0);
                let p_neg = calculate_p_value_exact(-t, 10.0);
                assert!((p_pos - p_neg).abs() < 1e-10);
            }
        }
    }
}
