"""
Test cases for edge cases and error handling to improve coverage.
"""
import numpy as np
import pytest

from rustgression import OlsRegressor, TlsRegressor


class TestRegressorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_regressor_string_representation(self):
        """Test string representation of regressors."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        
        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)
        
        # Test __repr__ method (line 135 in base.py)
        ols_repr = repr(ols)
        tls_repr = repr(tls)
        
        assert "OlsRegressor" in ols_repr
        assert "slope=" in ols_repr
        assert "intercept=" in ols_repr
        assert "r_value=" in ols_repr
        
        assert "TlsRegressor" in tls_repr
        assert "slope=" in tls_repr
        assert "intercept=" in tls_repr
        assert "r_value=" in tls_repr

    def test_predict_method_coverage(self):
        """Test predict method for complete coverage."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])
        
        regressor = OlsRegressor(x, y)
        
        # Test predict method
        x_new = np.array([5.0, 6.0])
        predictions = regressor.predict(x_new)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(x_new)
        
        # For perfect linear relationship, predictions should be exact
        expected = regressor.slope() * x_new + regressor.intercept()
        np.testing.assert_array_almost_equal(predictions, expected, decimal=10)

    def test_abstract_method_coverage(self):
        """Test abstract method behaviors.""" 
        from rustgression.regression.base import BaseRegressor
        
        # BaseRegressor is abstract and cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseRegressor(np.array([1, 2]), np.array([1, 2]))

    def test_get_params_method_coverage(self):
        """Test get_params method for coverage."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        
        ols = OlsRegressor(x, y)
        tls = TlsRegressor(x, y)
        
        # Test get_params method exists and returns something
        ols_params = ols.get_params()
        tls_params = tls.get_params()
        
        # Basic validation that params are returned
        assert ols_params is not None
        assert tls_params is not None
        
        # Check params have expected attributes
        assert hasattr(ols_params, 'slope')
        assert hasattr(ols_params, 'intercept')
        assert hasattr(ols_params, 'r_value')
        
        assert hasattr(tls_params, 'slope')
        assert hasattr(tls_params, 'intercept')
        assert hasattr(tls_params, 'r_value')

    def test_regressor_properties_consistency(self):
        """Test that regressor properties are consistent with get_params."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 3.5, 5.5, 7.5])
        
        regressor = OlsRegressor(x, y)
        params = regressor.get_params()
        
        # Test that properties match params
        assert abs(regressor.slope() - params.slope) < 1e-10
        assert abs(regressor.intercept() - params.intercept) < 1e-10
        assert abs(regressor.r_value() - params.r_value) < 1e-10

    def test_large_numbers_handling(self):
        """Test handling of large numbers."""
        x = np.array([1e6, 2e6, 3e6])
        y = np.array([2e6, 4e6, 6e6])
        
        regressor = OlsRegressor(x, y)
        
        # Should handle large numbers without issues
        assert abs(regressor.slope() - 2.0) < 1e-10
        assert abs(regressor.intercept()) < 1e-6
        assert abs(regressor.r_value() - 1.0) < 1e-10

    def test_small_numbers_handling(self):
        """Test handling of very small numbers."""
        x = np.array([1e-6, 2e-6, 3e-6])
        y = np.array([2e-6, 4e-6, 6e-6])
        
        regressor = OlsRegressor(x, y)
        
        # Should handle small numbers without issues
        assert abs(regressor.slope() - 2.0) < 1e-10
        assert abs(regressor.intercept()) < 1e-12
        assert abs(regressor.r_value() - 1.0) < 1e-10