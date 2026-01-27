"""
Example demonstrating p-value calculations and multiple testing correction.

Shows how to:
1. Generate p-value tables
2. Apply different correction methods (none, bonferroni, fdr_bh)
"""

import pandas as pd
import numpy as np
from equiflow import EquiFlow

# Generate sample data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age': np.random.normal(55, 15, n).clip(18, 95).astype(int),
    'sex': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
    'race': np.random.choice(
        ['White', 'Black', 'Asian', 'Hispanic', 'Other'], 
        n, 
        p=[0.60, 0.15, 0.10, 0.10, 0.05]
    ),
    'score': np.random.exponential(10, n),
})

# Initialize EquiFlow
ef = EquiFlow(
    data=data,
    categorical=['sex', 'race'],
    normal=['age'],
    nonnormal=['score'],
)

# Add exclusion
ef.add_exclusion(
    keep=data['age'] >= 40,
    exclusion_reason="Age < 40"
)

# P-values without correction
print("=== P-values (no correction) ===")
pvals_none = ef.view_table_pvalues(correction="none")
print(pvals_none)
print()

# P-values with Bonferroni correction
print("=== P-values (Bonferroni correction) ===")
pvals_bonf = ef.view_table_pvalues(correction="bonferroni")
print(pvals_bonf)
print()

# P-values with FDR correction
print("=== P-values (FDR/Benjamini-Hochberg correction) ===")
pvals_fdr = ef.view_table_pvalues(correction="fdr_bh")
print(pvals_fdr)
