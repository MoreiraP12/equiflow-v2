import pandas as pd
import numpy as np
from equiflow import EquiFlow
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create a sample dataset
np.random.seed(42)
n = 1000

# Generate sample data
data = pd.DataFrame({
    'age': np.random.normal(50, 15, n),
    'sex': np.random.choice(['Female', 'Male'], n),
    'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.6, 0.2, 0.15, 0.05]),
    'income': np.random.lognormal(10, 1, n),
    'bmi': np.random.normal(28, 5, n),
    'bp_systolic': np.random.normal(120, 15, n),
    'bp_diastolic': np.random.normal(80, 10, n),
    'glucose': np.random.normal(100, 25, n),
    'has_diabetes': np.random.choice([True, False], n, p=[0.15, 0.85]),
    'smoker': np.random.choice(['Never', 'Former', 'Current'], n, p=[0.6, 0.25, 0.15]),
})

# Introduce some missing values
data.loc[np.random.choice(n, 50), 'race'] = None
data.loc[np.random.choice(n, 30), 'income'] = None

print(f"Initial cohort size: {len(data)}")

# Create output directory with absolute path
output_dir = os.path.abspath("test_output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)

# EXAMPLE 1: Full metrics version
print("\nCreating EquiFlow instance with ALL metrics...")
ef_full = EquiFlow(
    data=data,
    initial_cohort_label="Initial Patient Cohort",
    categorical=['sex', 'race', 'smoker', 'has_diabetes'],
    normal=['age', 'bmi', 'bp_systolic', 'bp_diastolic'],
    nonnormal=['income', 'glucose'],
    format_cat='%'  # Set to percentage format
)

# Add exclusion steps
print("Adding exclusion criteria...")
ef_full.add_exclusion(
    mask=data['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult Patients"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[1]['race'].isna(),
    exclusion_reason="Missing race data",
    new_cohort_label="Complete Race Data"
)

ef_full.add_exclusion(
    mask=(ef_full._dfs[2]['bmi'] >= 15) & (ef_full._dfs[2]['bmi'] <= 50),
    exclusion_reason="BMI outside range 15-50",
    new_cohort_label="Final Analysis Cohort"
)

# Generate the full flow diagram
print("Generating flow diagram with ALL metrics...")
ef_full.plot_flows(
    output_folder=output_dir,
    output_file="full_flow_diagram",
    plot_dists=True,
    smds=False,
    legend=True,
    box_width=3.5,
    box_height=1.5,
    categorical_bar_colors=['skyblue', 'coral', 'lightgreen', 'plum'],
    display_flow_diagram=True
)

print(f"Full metrics diagram saved to {output_dir}/full_flow_diagram.pdf")

# EXAMPLE 2: Focused metrics version (just race and sex)
print("\nCreating focused EquiFlow instance with ONLY race and sex...")
ef_focused = EquiFlow(
    data=data,
    initial_cohort_label="Initial Patient Cohort",
    categorical=['race', 'sex'],
    normal=None,
    nonnormal=None,
    format_cat='%'  # Set to percentage format
)

# Add the same exclusion steps
print("Adding exclusion criteria...")
ef_focused.add_exclusion(
    mask=data['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult Patients"
)

ef_focused.add_exclusion(
    mask=~ef_focused._dfs[1]['race'].isna(),
    exclusion_reason="Missing race data",
    new_cohort_label="Complete Race Data"
)

ef_focused.add_exclusion(
    mask=(ef_focused._dfs[2]['bmi'] >= 15) & (ef_focused._dfs[2]['bmi'] <= 50),
    exclusion_reason="BMI outside range 15-50",
    new_cohort_label="Final Analysis Cohort"
)

# Generate the focused flow diagram
print("Generating flow diagram with ONLY race and sex...")
ef_focused.plot_flows(
    output_folder=output_dir,
    output_file="focused_flow_diagram",
    plot_dists=True,
    smds=False,
    legend=True,
    box_width=3.5,
    box_height=1.5,
    categorical_bar_colors=['skyblue', 'coral', 'lightgreen', 'plum'],
    display_flow_diagram=True
)

print(f"Focused diagram saved to {output_dir}/focused_flow_diagram.pdf")

print("\nDone! Compare the two diagrams to see the difference:")
print(f"1. Full metrics diagram: {output_dir}/full_flow_diagram.pdf")
print(f"2. Focused diagram (race & sex only): {output_dir}/focused_flow_diagram.pdf")