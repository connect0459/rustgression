// Utility modules for regression calculations

pub mod math;
pub mod statistics;
pub mod validation;

// Re-export commonly used functions
pub use math::kahan_sum;
pub use statistics::{calculate_p_value_exact, compute_r_value};
pub use validation::{safe_divide, validate_finite_array};
