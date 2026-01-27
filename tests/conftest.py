"""
Shared pytest fixtures for equiflow tests.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """
    Create a sample dataset for testing.
    
    Returns a DataFrame with 200 patients containing:
    - age: normally distributed continuous variable
    - sex: categorical (Male/Female)
    - race: categorical (White/Black/Asian/Other)
    - score: non-normally distributed continuous variable
    - outcome: binary outcome variable
    """
    np.random.seed(42)
    n = 200
    
    return pd.DataFrame({
        'age': np.random.normal(55, 15, n).clip(18, 95).astype(int),
        'sex': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'race': np.random.choice(
            ['White', 'Black', 'Asian', 'Other'], 
            n, 
            p=[0.65, 0.15, 0.10, 0.10]
        ),
        'score': np.random.exponential(10, n),
        'outcome': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def sample_data_with_missing(sample_data):
    """
    Create a sample dataset with missing values.
    
    Introduces ~5% missing values in age, score, and race columns.
    """
    df = sample_data.copy()
    np.random.seed(43)
    
    # Add missing values
    n = len(df)
    missing_indices_age = np.random.choice(n, size=int(n * 0.05), replace=False)
    missing_indices_score = np.random.choice(n, size=int(n * 0.05), replace=False)
    missing_indices_race = np.random.choice(n, size=int(n * 0.05), replace=False)
    
    df.loc[missing_indices_age, 'age'] = np.nan
    df.loc[missing_indices_score, 'score'] = np.nan
    df.loc[missing_indices_race, 'race'] = np.nan
    
    return df


@pytest.fixture
def large_sample_data():
    """
    Create a larger sample dataset for performance testing.
    
    Returns a DataFrame with 5000 patients.
    """
    np.random.seed(44)
    n = 5000
    
    return pd.DataFrame({
        'age': np.random.normal(55, 15, n).clip(18, 95).astype(int),
        'sex': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'race': np.random.choice(
            ['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
            n, 
            p=[0.60, 0.15, 0.08, 0.12, 0.05]
        ),
        'ethnicity': np.random.choice(
            ['Hispanic', 'Non-Hispanic'],
            n,
            p=[0.15, 0.85]
        ),
        'score': np.random.exponential(10, n),
        'los_days': np.random.lognormal(1.5, 0.8, n),
        'apache_score': np.random.gamma(3, 5, n),
        'outcome': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
