"""
Test script for EasyFlow simplified interface.
"""

import pandas as pd
import numpy as np
import os
import sys

# Import our package
from equiflow import EasyFlow

# Create a sample dataset (simulates patient data)
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    # Demographic data
    'age': np.random.normal(45, 15, n),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.49, 0.03]),
    'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n),
    
    # Clinical data
    'bmi': np.random.normal(27, 5, n),
    'los_days': np.random.exponential(5, n),  # Length of stay (days)
    'cost': np.random.exponential(10000, n),  # Cost in dollars
    
    # Study-specific data
    'consent': np.random.choice(['yes', 'no'], n, p=[0.85, 0.15]),
    'missing_data': np.random.choice([True, False], n, p=[0.75, 0.25])
})

# Add some constraints to make the data more realistic
data['age'] = data['age'].clip(18, 95).round(0)
data['bmi'] = data['bmi'].clip(15, 50).round(1)
data['los_days'] = data['los_days'].clip(1, 30).round(1)
data['cost'] = data['cost'].clip(1000, 100000).round(-2)

# Method 1: Using the fluent interface
print("Creating flow diagram using the fluent interface...")
flow = (EasyFlow(data, title="Patient Selection")
    .categorize(["gender", "race"])
    .measure_normal(["age", "bmi"])
    .measure_nonnormal(["los_days", "cost"])
    .exclude(data["age"] >= 18, "Removed patients under 18 years")
    .exclude(data["consent"] == "yes", "Excluded non-consenting patients")
    .exclude(data["missing_data"] == True, "Removed records with missing data")
    .generate(output="patient_selection_diagram", show=False)
)

print("\nAccessing results:")
print(f"Flow table shape: {flow.flow_table.shape}")
print(f"Characteristics table shape: {flow.characteristics.shape}")
print(f"Drifts table shape: {flow.drifts.shape}")

print("\nFlow table:")
print(flow.flow_table)

print("\nEasyFlow test completed!")
