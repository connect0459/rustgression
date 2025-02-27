use pyo3::prelude::*;

mod tls_regression;

/// TLS回帰を実行するRustモジュール
#[pymodule]
fn tls_regressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tls_regression::calculate_tls_regression, m)?)?;
    Ok(())
}
