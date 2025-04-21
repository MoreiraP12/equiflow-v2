import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from equiflow import EquiFlow

# Create a sample dataset
np.random.seed(42)
n = 1000

# Create basic demographics
data = pd.DataFrame({
    'age': np.random.normal(50, 15, n),
    'sex': np.random.choice(['Male', 'Female'], n),
    'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.6, 0.2, 0.15, 0.05]),
    'education': np.random.choice(['High School', 'College', 'Graduate'], n, p=[0.4, 0.4, 0.2]),
    'income': np.random.lognormal(10, 1, n),
    'bmi': np.random.normal(28, 5, n),
    'systolic_bp': np.random.normal(130, 20, n),
    'diastolic_bp': np.random.normal(80, 10, n),
    'smoking': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.5, 0.3, 0.2]),
    'diabetes': np.random.choice(['No', 'Yes'], n, p=[0.8, 0.2]),
})

# Add some missing values
for col in data.columns:
    mask = np.random.rand(n) < 0.05  # 5% missing data
    data.loc[mask, col] = None

# Print basic dataset info
print(f"Original dataset: {len(data)} patients")
print(data.head())

# Initialize EquiFlow
ef = EquiFlow(
    data=data,
    initial_cohort_label="Initial Cohort",
    categorical=['sex', 'race', 'education', 'smoking', 'diabetes'],
    normal=['age', 'bmi'],
    nonnormal=['income', 'systolic_bp', 'diastolic_bp']
)

# Add exclusion steps
print("\nAdding exclusion steps...")

# Exclude patients under 18
ef.add_exclusion(
    mask=data['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adults"
)
print(f"After age exclusion: {len(ef._dfs[-1])} patients")

# Exclude patients with missing BMI
ef.add_exclusion(
    mask=~ef._dfs[-1]['bmi'].isna(),
    exclusion_reason="Missing BMI",
    new_cohort_label="Complete BMI data"
)
print(f"After BMI exclusion: {len(ef._dfs[-1])} patients")

# Exclude patients with very high BMI
ef.add_exclusion(
    mask=ef._dfs[-1]['bmi'] < 40,
    exclusion_reason="BMI ≥ 40",
    new_cohort_label="Normal/overweight BMI"
)
print(f"After high BMI exclusion: {len(ef._dfs[-1])} patients")

# Test 1: Default automatic color shading (our new implementation)
print("\nGenerating diagram with automatic color shading...")
ef.plot_flows(
    output_file="flow_diagram_auto_colors",
    display_flow_diagram=True
)

# Generate tables for inspection
print("\nGenerating tables...")
table_flows = ef.view_table_flows()

print("\nFlow table:")
table_flows

