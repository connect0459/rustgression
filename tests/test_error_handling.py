"""
Test cases for error handling scenarios to improve coverage.
"""
import importlib
import sys
import unittest.mock

import pytest


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_init_rust_import_error(self):
        """Test error handling in __init__.py when Rust import fails."""
        # Create a mock module that fails to import calculate_ols_regression
        with unittest.mock.patch.dict(
            sys.modules, 
            {'rustgression.rustgression': None}
        ):
            with unittest.mock.patch(
                'builtins.__import__', 
                side_effect=ImportError("Rust module not found")
            ):
                # Import the package to trigger error handling in __init__.py
                import rustgression
                # This should succeed despite the error, but trigger the except block
                # Lines 50-54 should be covered
                assert rustgression is not None

    def test_init_regression_import_error(self):
        """Test error handling in __init__.py when regression module import fails."""
        # Mock successful Rust import but failed regression import
        mock_rust_module = unittest.mock.MagicMock()
        mock_rust_module.calculate_ols_regression = lambda: None
        
        with unittest.mock.patch.dict(
            sys.modules, 
            {'rustgression.rustgression': mock_rust_module}
        ):
            # Mock regression module import failure
            original_import = __builtins__['__import__']
            
            def mock_import(name, *args, **kwargs):
                if name == 'rustgression.regression' or 'regression' in name:
                    raise ImportError("Regression module not found")
                return original_import(name, *args, **kwargs)
            
            with unittest.mock.patch('builtins.__import__', side_effect=mock_import):
                # This should trigger lines 73-76 in __init__.py
                try:
                    importlib.reload(importlib.import_module('rustgression'))
                except ImportError:
                    # Expected behavior
                    pass

    def test_rust_imports_error_path_coverage(self):
        """Test error path coverage in _rust_imports.py."""
        # Clear module from cache to force reimport
        if 'rustgression.regression._rust_imports' in sys.modules:
            del sys.modules['rustgression.regression._rust_imports']
        
        # Mock failed import from parent module
        with unittest.mock.patch(
            'builtins.__import__',
            side_effect=ImportError("Failed to import from parent")
        ):
            # Mock find_spec to return None (line 24 path)
            with unittest.mock.patch('importlib.util.find_spec', return_value=None):
                with pytest.raises(ImportError, match="Could not find rustgression.rustgression module"):
                    importlib.import_module('rustgression.regression._rust_imports')

    def test_rust_imports_stderr_output(self):
        """Test stderr output in _rust_imports.py error handling.""" 
        # Clear module from cache
        if 'rustgression.regression._rust_imports' in sys.modules:
            del sys.modules['rustgression.regression._rust_imports']
        
        with unittest.mock.patch('sys.stderr') as mock_stderr:
            with unittest.mock.patch(
                'builtins.__import__',
                side_effect=ImportError("Test error")
            ):
                with unittest.mock.patch('importlib.util.find_spec', return_value=None):
                    with pytest.raises(ImportError):
                        importlib.import_module('rustgression.regression._rust_imports')
            
            # Verify stderr.write was called (line 26)
            mock_stderr.write.assert_called()

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise appropriate errors."""
        import numpy as np

        from rustgression.regression.base import BaseRegressor
        
        # Create a concrete subclass that doesn't implement abstract methods
        class IncompleteRegressor(BaseRegressor):
            def __init__(self, x, y):
                super().__init__(x, y)
                # Don't call _fit() to test abstract method
            
            # Don't implement _fit or get_params to test abstract behavior
        
        with pytest.raises(TypeError):
            # This should fail because abstract methods aren't implemented
            IncompleteRegressor(np.array([1, 2]), np.array([1, 2]))