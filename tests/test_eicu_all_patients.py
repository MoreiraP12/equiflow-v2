import numpy as np
from equiflow import EquiFlow
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')

# Read the dataset
data = pd.read_csv("eicu_all_patients.csv")
# Print initial cohort size
print(f"Initial cohort size: {len(data)}")

# Create output directory with absolute path
output_dir = os.path.abspath("test_output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)

# Convert max_troponin to numeric before creating EquiFlow instance
data_processed = data.copy()
data_processed['max_troponin'] = pd.to_numeric(data_processed['max_troponin'], errors='coerce')
# data_processed['non_cardiac_patient'] = pd.to_numeric(data_processed['non_cardiac_patient'], errors='coerce')

data_processed = data_processed.dropna(subset=['ethnicity'])

print(f"Cohort size before removing unknown gender: {len(data_processed)}")
data_processed = data_processed[~data_processed['gender'].isin(['Other', 'Unknown'])]
print(f"Cohort size after removing unknown gender: {len(data_processed)}")
# data_processed['apacheadmissiondx'].value_counts().to_csv('apacheadmissiondx_counts.csv', header=['count'])

# Check the data after each step
print(f"Initial data shape: {data_processed.shape}")

# Check septic patient distribution
print(f"\nSeptic patient counts:")
print(data_processed['septic_patient'].value_counts())

# Check cardiac patient distribution  
print(f"\nNon-cardiac patient counts:")
print(data_processed['non_cardiac_patient'].value_counts())

# rename maxapachescore to max_apache_score for consistency
data_processed.rename(columns={'maxapachescore': 'max_apache_score'}, inplace=True)
data_processed['max_apache_score'].value_counts().to_csv('max_apache_score_counts.csv', header=['count'])

# Most common ethnicities in order
eth_order = ['Caucasian', 'African American', 'Other/Unknown', 'Hispanic', 'Asian', 'Native American']
data_processed['ethnicity'] = pd.Categorical(
    data_processed['ethnicity'],
    categories=eth_order,
    ordered=True
)
data_processed = data_processed.sort_values('ethnicity', kind='mergesort')

# EXAMPLE 1: Full metrics version
print("\nCreating EquiFlow instance with ALL metrics...")
ef_full = EquiFlow(
    data=data_processed,
    initial_cohort_label="Initial Patient Cohort",
    categorical=['gender', 'ethnicity'],
    normal=['age', 'max_apache_score'],
    format_cat='%'
)

# Add exclusion steps
print("Adding exclusion criteria...")

# Add exclusion for cardiac patients
ef_full.add_exclusion(
    mask=ef_full._dfs[-1]['non_cardiac_patient'] != 0,
    exclusion_reason="heart disease admission",
    new_cohort_label="Patients not admitted for heart disease"
)

# Add exclusion for non-septic patients
ef_full.add_exclusion(
    mask=ef_full._dfs[-1]['septic_patient'] != 0,
    exclusion_reason="sepsis diagnosis",
    new_cohort_label="Patients with sepsis"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['max_troponin'].isna(),
    exclusion_reason="missing troponin data",
    new_cohort_label="Complete troponin data"
)

# Generate the full flow diagram
print("Generating flow diagram with ALL metrics...")
ef_full.plot_flows(
    output_folder=output_dir,
    output_file="v12_mod_eicu_case_study",
    plot_dists=True,
    smds=True,
    legend=True,
    box_width=3.5,
    box_height=1.5,
    display_flow_diagram=True
)

# Generate tables for inspection
print("\nGenerating tables...")
table_flows = ef_full.view_table_flows()
table_characteristics = ef_full.view_table_characteristics()
table_drifts = ef_full.view_table_drifts()

print("\nFlow table:")
print(table_flows)

print("\nCharacteristics table:")
print(table_characteristics)

print("\nDrifts table:")
print(table_drifts)

table_flows.to_csv(os.path.join(output_dir, "v12_table_flows.csv"), index=False)
table_characteristics.to_csv(os.path.join(output_dir, "v12_table_characteristics.csv"), index=False)
table_drifts.to_csv(os.path.join(output_dir, "v12_table_drifts.csv"), index=False)

print(f"Full metrics diagram saved to {output_dir}/v12_mod_eicu_case_study.pdf")