use pyo3::prelude::*;

mod regression;

/// 回帰分析を実行するRustモジュール
#[pymodule]
fn rustgression(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regression::calculate_ols_regression, m)?)?;
    m.add_function(wrap_pyfunction!(regression::calculate_tls_regression, m)?)?;
    Ok(())
}
