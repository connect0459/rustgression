use pyo3::prelude::*;

mod tls;

/// TLS回帰を実行するRustモジュール
#[pymodule]
fn tls_regressor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tls::calculate_tls_regression, m)?)?;
    Ok(())
}
