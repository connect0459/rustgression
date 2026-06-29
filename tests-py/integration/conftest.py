import numpy as np
import pytest


@pytest.fixture
def sample_data():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 100)
    return x, y
