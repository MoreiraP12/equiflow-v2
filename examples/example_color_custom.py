"""
Example demonstrating custom colors in flow diagrams.

Shows how to customize colors for different demographic groups.
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
})

# Initialize EquiFlow
ef = EquiFlow(
    data=data,
    initial_cohort_label="All Patients",
    categorical=['sex', 'race'],
    normal=['age'],
)

# Add exclusions
ef.add_exclusion(
    keep=data['age'] >= 30,
    exclusion_reason="Age < 30",
    new_cohort_label="Adults"
)

# Generate with custom colors
# Uncomment to generate:
# ef.plot_flows(
#     output_file="custom_colors",
#     display_flow_diagram=False,
#     color_var='sex',
#     colors={'Male': '#1f77b4', 'Female': '#ff7f0e'}
# )
# print("Diagram with custom colors saved!")

print("Custom color example - uncomment plot_flows() to generate diagram")
