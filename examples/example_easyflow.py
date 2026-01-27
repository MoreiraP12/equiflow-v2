"""
Example demonstrating EasyFlow simplified interface.

EasyFlow provides a fluent API for quick analyses with method chaining.
"""

import pandas as pd
import numpy as np
from equiflow import EasyFlow

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

# EasyFlow with method chaining
flow = (EasyFlow(data, title="My Quick Study")
    .categorize(['sex', 'race'])
    .measure_normal(['age'])
    .measure_nonnormal(['score'])
    .exclude(data['age'] >= 30, "Age < 30")
    .exclude(lambda d: d['score'] > 5, "Score ≤ 5"))

print(f"Exclusion steps: {len(flow._exclusion_steps)}")
print(f"Initial cohort: {len(flow._data)}")
print(f"Final cohort: {len(flow._current_data)}")

# Generate output (uncomment to generate diagram)
# flow.generate(output="easyflow_example", show=False)
# print("Results saved!")
