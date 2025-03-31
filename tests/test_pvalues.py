"""
Test script for equiflow package with p-values functionality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Ensure you've implemented the new TablePValues class and added the 
# view_table_pvalues method to EquiFlow before running this

from equiflow import EquiFlow, TablePValues

# ----- Create Sample Dataset -----
def create_test_data(n=1000):
    """Create a sample dataset with various variable types."""
    np.random.seed(42)  # For reproducibility
    
    # Create DataFrame
    data = pd.DataFrame()
    
    # Categorical variables
    data['gender'] = np.random.choice(['Male', 'Female'], size=n)
    data['smoking'] = np.random.choice(['Never', 'Former', 'Current'], size=n, p=[0.6, 0.2, 0.2])
    data['treatment'] = np.random.choice(['A', 'B', 'C'], size=n)
    
    # Normal continuous variables
    data['age'] = np.random.normal(65, 10, size=n)
    data['height'] = np.random.normal(170, 10, size=n)
    data['weight'] = np.random.normal(75, 15, size=n)
    
    # Non-normal continuous variables
    data['lab_value1'] = np.random.exponential(5, size=n)
    data['lab_value2'] = np.random.lognormal(0, 1, size=n)
    data['days_hospitalized'] = np.random.poisson(3, size=n)
    
    # Add some missing values
    for col in data.columns:
        mask = np.random.random(size=n) < 0.05  # 5% missing values
        data.loc[mask, col] = None
    
    return data

# ----- Test Script -----
def test_equiflow_pvalues():
    """Test the p-values functionality in equiflow."""
    print("Creating test data...")
    data = create_test_data(1000)
    
    print(f"Initial dataset shape: {data.shape}")
    
    # Set up variable categories
    categorical = ['gender', 'smoking', 'treatment']
    normal = ['age', 'height', 'weight']
    nonnormal = ['lab_value1', 'lab_value2', 'days_hospitalized']
    
    # Initialize EquiFlow
    print("Initializing EquiFlow...")
    ef = EquiFlow(
        data=data,
        initial_cohort_label="Initial Cohort",
        categorical=categorical,
        normal=normal,
        nonnormal=nonnormal,
        format_cat='N (%)',
        format_normal='Mean ± SD',
        format_nonnormal='Median [IQR]'
    )
    
    # Define exclusions
    print("Adding exclusions...")
    # Exclusion 1: Age < 50
    ef.add_exclusion(
        mask=data['age'] >= 50,
        exclusion_reason="Age < 50",
        new_cohort_label="Age ≥ 50"
    )
    
    # Exclusion 2: Missing lab values
    ef.add_exclusion(
        mask=~data['lab_value1'].isna(),
        exclusion_reason="Missing lab values",
        new_cohort_label="Complete lab data"
    )
    
    # Exclusion 3: Smokers
    ef.add_exclusion(
        mask=data['smoking'] != 'Current',
        exclusion_reason="Current smokers",
        new_cohort_label="Non-smokers"
    )
    
    # Generate tables
    print("Generating tables...")
    
    # Flow table
    flow_table = ef.view_table_flows()
    print("\nCohort Flow Table:")
    print(flow_table)
    
    # Characteristics table
    characteristics = ef.view_table_characteristics()
    print("\nCharacteristics Table (sample):")
    print(characteristics.iloc[:5])
    
    # Drifts table
    drifts = ef.view_table_drifts()
    print("\nDrifts Table (sample):")
    print(drifts.iloc[:5])
    
    # P-values table (the new feature)
    print("\nCalculating p-values...")
    pvalues = ef.view_table_pvalues()
    print("\nP-values Table (sample):")
    print(pvalues.iloc[:5])
    
    # Generate flow diagram
    print("\nGenerating flow diagram...")
    ef.plot_flows(
        output_file="test_flow_with_pvalues",
        display_flow_diagram=True
    )
    
    print("\nTest complete!")
    return ef, flow_table, characteristics, drifts, pvalues

# ----- Run the test -----
if __name__ == "__main__":
    print("Testing equiflow with p-values functionality...\n")
    ef, flow_table, characteristics, drifts, pvalues = test_equiflow_pvalues()
    
    # Optionally export to Excel
    with pd.ExcelWriter('equiflow_test_results.xlsx') as writer:
        flow_table.to_excel(writer, sheet_name='Flow')
        characteristics.to_excel(writer, sheet_name='Characteristics')
        drifts.to_excel(writer, sheet_name='Drifts')
        pvalues.to_excel(writer, sheet_name='P-values')
    
    print("\nResults exported to 'equiflow_test_results.xlsx'")