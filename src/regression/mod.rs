mod ols;
mod ols_multi;
mod tls;
mod utils;

pub use ols::calculate_ols_regression;
pub use ols_multi::calculate_ols_multi_regression;
pub use tls::calculate_tls_regression;
