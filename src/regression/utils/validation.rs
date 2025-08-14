use pyo3::prelude::*;
use std::f64;

/// Internal implementation for finite array validation
fn validate_finite_array_impl<E, F>(array: &[f64], name: &str, error_fn: F) -> Result<(), E>
where
    F: Fn(String) -> E,
{
    for (i, &value) in array.iter().enumerate() {
        if !value.is_finite() {
            if value.is_nan() {
                return Err(error_fn(format!(
                    "NaN detected in {} array at index {}",
                    name, i
                )));
            } else if value.is_infinite() {
                return Err(error_fn(format!(
                    "Infinite value detected in {} array at index {}",
                    name, i
                )));
            }
        }
        // Subnormal number detection
        if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
            eprintln!(
                "Warning: Subnormal number detected in {} array at index {}: {}",
                name, i, value
            );
        }
    }
    Ok(())
}

/// Internal implementation for safe division
fn safe_divide_impl<E, F>(
    numerator: f64,
    denominator: f64,
    context: &str,
    error_fn: F,
) -> Result<f64, E>
where
    F: Fn(String) -> E,
{
    if !numerator.is_finite() || !denominator.is_finite() {
        return Err(error_fn(format!(
            "Non-finite values in division ({}): {}/{}",
            context, numerator, denominator
        )));
    }

    if denominator.abs() < f64::EPSILON {
        return Err(error_fn(format!(
            "Division by zero or near-zero value ({}): denominator = {}",
            context, denominator
        )));
    }

    let result = numerator / denominator;
    if !result.is_finite() {
        return Err(error_fn(format!(
            "Division resulted in non-finite value ({}): {}/{} = {}",
            context, numerator, denominator, result
        )));
    }

    Ok(result)
}

/// IEEE 754 edge case detection and handling
///
/// Parameters
/// ----------
/// array : &[f64]
///     Array reference
/// name : &str
///     Array name for error messages
///
/// Returns
/// -------
/// PyResult<()>
///     Ok(()) on success, error if problems found
pub fn validate_finite_array(array: &[f64], name: &str) -> PyResult<()> {
    validate_finite_array_impl(array, name, |msg| {
        pyo3::exceptions::PyValueError::new_err(msg)
    })
}

/// Safe division with zero division and non-finite value checks
///
/// Parameters
/// ----------
/// numerator : f64
///     Numerator
/// denominator : f64
///     Denominator
/// context : &str
///     Context for error messages
///
/// Returns
/// -------
/// PyResult<f64>
///     Division result, or error if problems occur
pub fn safe_divide(numerator: f64, denominator: f64, context: &str) -> PyResult<f64> {
    safe_divide_impl(numerator, denominator, context, |msg| {
        pyo3::exceptions::PyValueError::new_err(msg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    // Test-specific versions of functions that don't depend on PyO3
    fn validate_finite_array_test(array: &[f64], name: &str) -> TestResult<()> {
        validate_finite_array_impl(array, name, |msg| msg.into())
    }

    fn safe_divide_test(numerator: f64, denominator: f64, context: &str) -> TestResult<f64> {
        safe_divide_impl(numerator, denominator, context, |msg| msg.into())
    }

    mod validate_finite_array {
        use super::*;

        #[test]
        fn test_valid_array() {
            let array = [1.0, 2.0, 3.0, 4.0, 5.0];
            assert!(validate_finite_array_test(&array, "test").is_ok());
        }

        #[test]
        fn test_nan_detection() {
            let array = [1.0, 2.0, f64::NAN, 4.0];
            let result = validate_finite_array_test(&array, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("NaN detected"));
            assert!(error.to_string().contains("index 2"));
        }

        #[test]
        fn test_positive_infinity_detection() {
            let array = [1.0, f64::INFINITY, 3.0];
            let result = validate_finite_array_test(&array, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Infinite value detected"));
            assert!(error.to_string().contains("index 1"));
        }

        #[test]
        fn test_negative_infinity_detection() {
            let array = [1.0, 2.0, f64::NEG_INFINITY];
            let result = validate_finite_array_test(&array, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Infinite value detected"));
            assert!(error.to_string().contains("index 2"));
        }

        #[test]
        fn test_subnormal_number_warning() {
            // Test with very small numbers (subnormal)
            let array = [1.0, 1e-320, 3.0]; // 1e-320 is subnormal
            let result = validate_finite_array_test(&array, "test");
            assert!(result.is_ok()); // Should succeed but print warning
        }

        #[test]
        fn test_empty_array() {
            let array: [f64; 0] = [];
            assert!(validate_finite_array_test(&array, "test").is_ok());
        }

        #[test]
        fn test_zeros_and_negatives() {
            let array = [0.0, -1.0, -0.0, 42.0];
            assert!(validate_finite_array_test(&array, "test").is_ok());
        }
    }

    mod safe_divide {
        use super::*;

        #[test]
        fn test_normal_division() {
            let result = safe_divide_test(10.0, 2.0, "test");
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), 5.0);
        }

        #[test]
        fn test_zero_division() {
            let result = safe_divide_test(5.0, 0.0, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Division by zero"));
        }

        #[test]
        fn test_near_zero_division() {
            let result = safe_divide_test(5.0, f64::EPSILON / 2.0, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("near-zero value"));
        }

        #[test]
        fn test_nan_numerator() {
            let result = safe_divide_test(f64::NAN, 2.0, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Non-finite values"));
        }

        #[test]
        fn test_nan_denominator() {
            let result = safe_divide_test(5.0, f64::NAN, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Non-finite values"));
        }

        #[test]
        fn test_infinite_numerator() {
            let result = safe_divide_test(f64::INFINITY, 2.0, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Non-finite values"));
        }

        #[test]
        fn test_infinite_denominator() {
            let result = safe_divide_test(5.0, f64::INFINITY, "test");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Non-finite values"));
        }

        #[test]
        fn test_negative_division() {
            let result = safe_divide_test(-10.0, -2.0, "test");
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), 5.0);
        }

        #[test]
        fn test_very_large_result() {
            // Test that could potentially overflow
            let result = safe_divide_test(f64::MAX, 0.5, "test");
            assert!(result.is_err()); // Should fail due to non-finite result
        }

        #[test]
        fn test_context_in_error_message() {
            let result = safe_divide_test(5.0, 0.0, "custom context");
            assert!(result.is_err());
            let error = result.unwrap_err();
            assert!(error.to_string().contains("custom context"));
        }
    }
}
