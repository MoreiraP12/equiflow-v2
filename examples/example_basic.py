"""
Basic example demonstrating EquiFlow usage.

This example shows how to:
1. Initialize EquiFlow with demographic variables
2. Add sequential exclusion steps
3. Generate tables and flow diagrams
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
        ['White', 'Black', 'Asian', 'Other'], 
        n, 
        p=[0.65, 0.15, 0.10, 0.10]
    ),
    'score': np.random.exponential(10, n),
    'outcome': np.random.choice([0, 1], n, p=[0.7, 0.3]),
})

# Initialize EquiFlow
ef = EquiFlow(
    data=data,
    initial_cohort_label="All Patients",
    categorical=['sex', 'race'],
    normal=['age'],
    nonnormal=['score'],
)

# Add exclusion steps
# Note: keep=True means KEEP the row (retain it)
ef.add_exclusion(
    keep=data['age'] >= 30,
    exclusion_reason="Age < 30",
    new_cohort_label="Adults 30+"
)

# Get current cohort for next exclusion
current = ef._dfs[-1]
ef.add_exclusion(
    keep=current['score'] > 5,
    exclusion_reason="Score ≤ 5",
    new_cohort_label="High score patients"
)

# View tables
print("=== Flow Table ===")
print(ef.view_table_flows())
print()

print("=== Characteristics Table ===")
print(ef.view_table_characteristics())
print()

print("=== Drift Table (SMDs) ===")
print(ef.view_table_drifts())
print()

# Generate flow diagram (uncomment to generate)
# ef.plot_flows(output_file="basic_flow", display_flow_diagram=False)
# print("Flow diagram saved to basic_flow.pdf")
