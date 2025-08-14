use pyo3::prelude::*;
use std::f64;

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
    for (i, &value) in array.iter().enumerate() {
        if !value.is_finite() {
            if value.is_nan() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "NaN detected in {} array at index {}",
                    name, i
                )));
            } else if value.is_infinite() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
    if !numerator.is_finite() || !denominator.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Non-finite values in division ({}): {}/{}",
            context, numerator, denominator
        )));
    }

    if denominator.abs() < f64::EPSILON {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Division by zero or near-zero value ({}): denominator = {}",
            context, denominator
        )));
    }

    let result = numerator / denominator;
    if !result.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Division resulted in non-finite value ({}): {}/{} = {}",
            context, numerator, denominator, result
        )));
    }

    Ok(result)
}
