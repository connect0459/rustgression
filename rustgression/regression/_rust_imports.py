"""
Rust module imports for regression calculations.
"""

import sys
import importlib.util

# Rustモジュールのインポートを修正
try:
    # 親モジュールからRust関数をインポート
    from ..rustgression import calculate_ols_regression, calculate_tls_regression
except ImportError:
    try:
        # 別の方法でインポートを試みる
        # rustgressionモジュールの存在を確認
        spec = importlib.util.find_spec("rustgression.rustgression")
        if spec is not None:
            # モジュールを動的にインポート
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