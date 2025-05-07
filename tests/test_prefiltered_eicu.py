import pandas as pd
import numpy as np
from equiflow import EquiFlow
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Read the dataset
data = pd.read_csv("../DataFrames/eicu_dataset_pre_filtered.csv")

# Print initial cohort size
print(f"Initial cohort size: {len(data)}")

# Create output directory with absolute path
output_dir = os.path.abspath("test_output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)

# Convert max_troponin to numeric before creating EquiFlow instance
data_processed = data.copy()
data_processed['max_troponin'] = pd.to_numeric(data_processed['max_troponin'], errors='coerce')

# EXAMPLE 1: Full metrics version
print("\nCreating EquiFlow instance with ALL metrics...")
ef_full = EquiFlow(
    data=data_processed,
    initial_cohort_label="Initial Patient Cohort",
    categorical=['gender', 'ethnicity'],
    normal=['age'],
    format_cat='%'  # Set to percentage format
)

# Add exclusion steps
print("Adding exclusion criteria...")

# For the first exclusion, use the first dataframe
ef_full.add_exclusion(
    mask=data_processed['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult Patients"
)

# For each subsequent exclusion, use the latest dataframe from _dfs
ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['gender'].isna(),
    exclusion_reason="Missing gender data",
    new_cohort_label="Complete gender data"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['ethnicity'].isna(),
    exclusion_reason="Missing ethnicity data",
    new_cohort_label="Complete ethnicity data"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['unitadmitsource'].isna(),
    exclusion_reason="Missing pre-ICU location data",
    new_cohort_label="Complete pre-ICU location data"
)

# Add exclusion for cardiac patients
ef_full.add_exclusion(
    mask=ef_full._dfs[-1]['non_cardiac_patient'] != 0,
    exclusion_reason="Patients with heart disease",
    new_cohort_label="Patients without heart disease"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['max_troponin'].isna(),
    exclusion_reason="Missing troponin data",
    new_cohort_label="Complete troponin data"
)

# Generate the full flow diagram
print("Generating flow diagram with ALL metrics...")
ef_full.plot_flows(
    output_folder=output_dir,
    output_file="eicu_full",
    plot_dists=True,
    smds=False,
    legend=True,
    box_width=3.5,
    box_height=1.5,
    categorical_bar_colors=['skyblue', 'coral', 'lightgreen', 'plum'],
    display_flow_diagram=True
)

print(f"Full metrics diagram saved to {output_dir}/eicu_full.pdf")