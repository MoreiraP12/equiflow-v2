"""
Pytest configuration and shared fixtures for equiflow tests.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Create a basic sample dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'age': np.random.normal(50, 15, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.6, 0.2, 0.15, 0.05]),
        'bmi': np.random.normal(28, 5, n),
        'income': np.random.lognormal(10, 1, n),
        'los_days': np.random.exponential(5, n),
    })


@pytest.fixture
def sample_data_with_missing(sample_data):
    """Create a sample dataset with missing values."""
    data = sample_data.copy()
    np.random.seed(42)
    n = len(data)
    for col in data.columns:
        mask = np.random.rand(n) < 0.05  # 5% missing
        data.loc[mask, col] = None
    return data


@pytest.fixture
def large_sample_data():
    """Create a larger sample dataset for performance testing."""
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'age': np.random.normal(50, 15, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n),
        'bmi': np.random.normal(28, 5, n),
        'income': np.random.lognormal(10, 1, n),
        'los_days': np.random.exponential(5, n),
        'score': np.random.normal(100, 15, n),
    })
