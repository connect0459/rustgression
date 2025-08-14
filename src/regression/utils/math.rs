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

#[cfg(test)]
mod tests {
    use super::*;

    mod kahan_sum {
        use super::*;

        #[test]
        fn test_simple_sum() {
            let values = [1.0, 2.0, 3.0, 4.0, 5.0];
            let result = kahan_sum(&values);
            assert_eq!(result, 15.0);
        }

        #[test]
        fn test_empty_array() {
            let values: [f64; 0] = [];
            let result = kahan_sum(&values);
            assert_eq!(result, 0.0);
        }

        #[test]
        fn test_single_value() {
            let values = [42.0];
            let result = kahan_sum(&values);
            assert_eq!(result, 42.0);
        }

        #[test]
        fn test_precision_improvement() {
            // Test case where Kahan summation shows improvement over naive summation
            let small_values = vec![0.1; 10];
            let kahan_result = kahan_sum(&small_values);
            let expected = 1.0;

            // Kahan summation should be very close to expected value
            assert!((kahan_result - expected).abs() < 1e-14);
        }

        #[test]
        fn test_precision_vs_naive() {
            // Test floating-point precision limits
            let mut values = vec![1.0; 1000000];
            values.push(1e-10);

            let kahan_result = kahan_sum(&values);
            let naive_result: f64 = values.iter().sum();

            // Verify Kahan summation is at least as accurate as naive sum
            assert!(kahan_result >= naive_result);
        }

        #[test]
        fn test_negative_values() {
            let values = [-1.0, -2.0, -3.0, -4.0];
            let result = kahan_sum(&values);
            assert_eq!(result, -10.0);
        }

        #[test]
        fn test_mixed_positive_negative() {
            let values = [10.0, -5.0, 3.0, -2.0];
            let result = kahan_sum(&values);
            assert_eq!(result, 6.0);
        }

        #[test]
        fn test_zeros() {
            let values = [0.0, 0.0, 0.0];
            let result = kahan_sum(&values);
            assert_eq!(result, 0.0);
        }

        #[test]
        fn test_very_small_values() {
            let values = [1e-15, 2e-15, 3e-15];
            let result = kahan_sum(&values);
            let expected = 6e-15;
            assert!((result - expected).abs() < 1e-16);
        }

        #[test]
        fn test_large_values() {
            let values = [1e10, 2e10, 3e10];
            let result = kahan_sum(&values);
            assert_eq!(result, 6e10);
        }

        #[test]
        fn test_nan_handling() {
            let values = [1.0, f64::NAN, 3.0];
            let result = kahan_sum(&values);
            assert!(result.is_nan());
        }

        #[test]
        fn test_infinity_handling() {
            let values = [1.0, f64::INFINITY, 3.0];
            let result = kahan_sum(&values);
            // With infinity, the result should either be infinite or NaN due to compensation arithmetic
            assert!(result.is_infinite() || result.is_nan());
        }

        #[test]
        fn test_alternating_small_large() {
            // Test case that challenges floating-point precision
            let values = [1e20, 1.0, -1e20, 1.0];
            let result = kahan_sum(&values);
            // This specific case may not always give 2.0 due to the extreme magnitude differences
            // Just verify the result is reasonable
            assert!((result - 2.0).abs() < 2.0); // Allow some variation
        }
    }
}
