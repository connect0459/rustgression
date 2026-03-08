#!/usr/bin/env python3
"""
Import test and basic functionality verification script

This script performs the following:
1. Import test for the rustgression module
2. Basic class instantiation test
3. Basic regression analysis verification
"""

import sys
import traceback

import numpy as np


def test_import():
    """Import test for the rustgression module."""
    try:
        import rustgression

        print(f"Successfully imported rustgression {rustgression.__version__}")
        return True
    except ImportError as e:
        print(f"Failed to import rustgression: {e}")
        return False


def test_basic_functionality():
    """Basic class instantiation and functionality test."""
    try:
        import rustgression

        # Create simple test data
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        # Try OLS regressor (new property method API)
        print("Creating OLS regressor...")
        regressor = rustgression.OlsRegressor(x, y)
        print(f"OLS slope: {regressor.slope()}")
        print(f"OLS intercept: {regressor.intercept()}")
        print(f"OLS r_value: {regressor.r_value()}")

        # Try TLS regressor as well (new property method API)
        print("Creating TLS regressor...")
        tls = rustgression.TlsRegressor(x, y)
        print(f"TLS slope: {tls.slope()}")
        print(f"TLS intercept: {tls.intercept()}")
        print(f"TLS r_value: {tls.r_value()}")

        print("All classes imported and tested successfully!")
        return True

    except Exception as e:
        print(f"Error during class testing: {e}")
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=== Rustgression Import and Basic Functionality Test ===")

    # Run import test
    if not test_import():
        sys.exit(1)

    # Run basic functionality test
    if not test_basic_functionality():
        sys.exit(1)

    print("All tests passed successfully!")


if __name__ == "__main__":
    main()
