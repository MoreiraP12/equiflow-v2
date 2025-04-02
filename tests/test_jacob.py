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
data = pd.read_csv("DataFrames/Original_DF2.csv")

# Introduce some missing values
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
    categorical=[ 'Gender', 'Frailty'],
    normal=['Age'],
    format_cat='%'  # Set to percentage format
)

# Add exclusion steps
print("Adding exclusion criteria...")
ef_full.add_exclusion(
    mask=data['Age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult Patients"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[1]['Frailty'].isna(),
    exclusion_reason="Missing race data",
    new_cohort_label="Complete Race Data"
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