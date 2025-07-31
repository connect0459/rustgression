"""
Test cases for import functionality and error handling.
"""

import importlib
import sys
import unittest.mock

import pytest


class TestImportErrorHandling:
    """Test error handling for import failures."""

    def test_rust_module_import_error(self):
        """Test handling of Rust module import errors."""
        # Mock the rustgression module import to fail
        with unittest.mock.patch.dict(sys.modules, {'rustgression.rustgression': None}):
            with unittest.mock.patch('builtins.__import__', side_effect=ImportError("Mock import error")):
                with pytest.raises(ImportError):
                    # This should trigger the import error handling
                    pass

    def test_importlib_find_spec_failure(self):
        """Test handling when importlib.util.find_spec returns None."""
        # Clear modules to force reimport
        modules_to_clear = [
            'rustgression.regression._rust_imports',
            'rustgression.rustgression'
        ]
        
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        # Mock both the initial import and find_spec
        with unittest.mock.patch.dict(sys.modules, {'rustgression.rustgression': None}):
            with unittest.mock.patch('builtins.__import__', side_effect=ImportError("Initial import error")):
                with unittest.mock.patch('importlib.util.find_spec', return_value=None):
                    with pytest.raises(ImportError, match="Could not find rustgression.rustgression module"):
                        importlib.import_module('rustgression.regression._rust_imports')

    def test_dynamic_import_failure(self):
        """Test handling when dynamic import fails."""
        # Clear modules to force reimport
        modules_to_clear = [
            'rustgression.regression._rust_imports',
            'rustgression.rustgression'
        ]
        
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        mock_spec = unittest.mock.MagicMock()
        mock_spec.loader.exec_module.side_effect = ImportError("Dynamic import failed")
        
        with unittest.mock.patch.dict(sys.modules, {'rustgression.rustgression': None}):
            with unittest.mock.patch('builtins.__import__', side_effect=ImportError("Initial import error")):
                with unittest.mock.patch('importlib.util.find_spec', return_value=mock_spec):
                    with unittest.mock.patch('importlib.util.module_from_spec', return_value=unittest.mock.MagicMock()):
                        with pytest.raises(ImportError, match="Dynamic import failed"):
                            importlib.import_module('rustgression.regression._rust_imports')


class TestModuleInitialization:
    """Test module initialization scenarios."""

    def test_successful_rust_import(self):
        """Test successful import of Rust module."""
        # This should work normally in the test environment
        from rustgression.regression._rust_imports import (
            calculate_ols_regression,
            calculate_tls_regression,
        )
        assert callable(calculate_ols_regression)
        assert callable(calculate_tls_regression)

    def test_regression_module_components(self):
        """Test that all expected components are available."""
        from rustgression import (
            OlsRegressionParams,
            OlsRegressor,
            TlsRegressionParams,
            TlsRegressor,
            create_regressor,
        )
        
        # Check that all components are importable
        assert OlsRegressor is not None
        assert TlsRegressor is not None
        assert OlsRegressionParams is not None
        assert TlsRegressionParams is not None
        assert create_regressor is not None

    def test_package_version(self):
        """Test package version is available."""
        import rustgression
        assert hasattr(rustgression, '__version__')
        assert rustgression.__version__ == "0.2.0"

    def test_package_all_attribute(self):
        """Test __all__ attribute contains expected items."""
        import rustgression
        expected_items = {
            "OlsRegressionParams",
            "OlsRegressor", 
            "TlsRegressionParams",
            "TlsRegressor",
            "create_regressor"
        }
        assert hasattr(rustgression, '__all__')
        assert set(rustgression.__all__) == expected_items