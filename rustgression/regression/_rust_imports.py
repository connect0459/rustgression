"""
Rust module imports for regression calculations.
"""

import importlib.util
import sys

# Fix Rust module import
try:
    # Import Rust functions from parent module
    from ..rustgression import calculate_ols_regression, calculate_tls_regression
except ImportError:
    try:
        # Try alternative import method
        # Check existence of rustgression module
        spec = importlib.util.find_spec("rustgression.rustgression")
        if spec is not None:
            # Dynamically import module
            rust_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rust_module)
            calculate_ols_regression = rust_module.calculate_ols_regression
            calculate_tls_regression = rust_module.calculate_tls_regression
        else:
            raise ImportError("Could not find rustgression.rustgression module")
    except ImportError as e:
        print(f"Failed to import Rust functions: {e}", file=sys.stderr)
        raise

__all__ = ["calculate_ols_regression", "calculate_tls_regression"]
