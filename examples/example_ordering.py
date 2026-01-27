"""
Example demonstrating variable and category ordering.

Shows how to:
1. Order variables in output tables
2. Order categories within categorical variables
3. Limit number of categories displayed
"""

import pandas as pd
import numpy as np
from equiflow import EquiFlow

# Generate sample data
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'age': np.random.normal(55, 15, n).clip(18, 95).astype(int),
    'sex': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
    'race': np.random.choice(
        ['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
        n, 
        p=[0.60, 0.15, 0.10, 0.10, 0.05]
    ),
    'diagnosis': np.random.choice(
        ['Sepsis', 'Pneumonia', 'COPD', 'CHF', 'Stroke', 'Trauma', 'Other'],
        n,
        p=[0.20, 0.15, 0.12, 0.15, 0.10, 0.13, 0.15]
    ),
    'score': np.random.exponential(10, n),
})

# Initialize with custom ordering
ef = EquiFlow(
    data=data,
    categorical=['sex', 'race', 'diagnosis'],
    normal=['age'],
    nonnormal=['score'],
    # Specify variable order (sex before race before diagnosis)
    order_vars=['age', 'sex', 'race', 'diagnosis', 'score'],
    # Specify category order within variables
    order_classes={
        'sex': ['Female', 'Male'],  # Female first
        'race': ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
    },
    # Limit diagnosis to top 3 categories
    limit={'diagnosis': 3},
)

# Add exclusion
ef.add_exclusion(
    keep=data['age'] >= 30,
    exclusion_reason="Age < 30"
)

# View tables with custom ordering
print("=== Characteristics Table (with custom ordering) ===")
print(ef.view_table_characteristics())
print()

print("Note: Variables appear in specified order,")
print("categories within variables follow specified order,")
print("and diagnosis is limited to top 3 categories.")
