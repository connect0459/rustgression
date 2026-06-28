use pyo3::prelude::*;

pyo3::create_exception!(
    rustgression,
    NumericalWarning,
    pyo3::exceptions::PyUserWarning
);

pub fn emit_numerical_warning(py: Python<'_>, message: &str) -> PyResult<()> {
    let warnings = PyModule::import(py, "warnings")?;
    let warning_type = py.get_type::<NumericalWarning>();
    warnings.call_method1("warn", (message, warning_type))?;
    Ok(())
}
