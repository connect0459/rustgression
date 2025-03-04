use pyo3::prelude::*;

mod regression;

/// Rust module for performing regression analysis.
///
/// This module provides Python bindings for regression functions implemented in Rust.
/// It includes methods for calculating Ordinary Least Squares (OLS) and Total Least Squares (TLS) regression.
///
/// Functions
/// ---------
/// calculate_ols_regression(x: &PyArray1<f64>, y: &PyArray1<f64>) -> (f64, f64, f64, f64)
///     Calculate the Ordinary Least Squares regression parameters.
///
/// calculate_tls_regression(x: &PyArray1<f64>, y: &PyArray1<f64>) -> (f64, f64, f64)
///     Calculate the Total Least Squares regression parameters.
///
/// Parameters
/// ----------
/// _py : Python
///     The Python interpreter instance.
/// m : &PyModule
///     The module to which the functions will be added.
///
/// Returns
/// -------
/// PyResult<()> 
///     A result indicating success or failure of the module initialization.
#[pymodule]
fn rustgression(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regression::calculate_ols_regression, m)?)?;
    m.add_function(wrap_pyfunction!(regression::calculate_tls_regression, m)?)?;
    Ok(())
}
