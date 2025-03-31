"""
Test script to verify the order_vars feature in EquiFlow.

This script creates a sample dataset and generates tables with different
variable orderings to verify that the order_vars feature works correctly.
"""

import pandas as pd
import numpy as np
import re

# Assuming equiflow has been modified with the order_vars feature
from equiflow import EquiFlow

def print_separator():
    """Print a separator line for better readability."""
    print("\n" + "="*80 + "\n")

def strip_label_suffix(var_name):
    """Remove format suffixes from variable names for comparison."""
    # Match patterns like ", Mean ± SD", ", N (%)", etc.
    pattern = r',\s*(Mean ± SD|Median \[IQR\]|N \(%\)|%|N)$'
    return re.sub(pattern, '', var_name)

def test_order_vars():
    """Test the order_vars feature in EquiFlow."""
    print("Testing the order_vars feature in EquiFlow")
    print_separator()
    
    # Create a sample dataset
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        # Demographics
        'age': np.random.normal(65, 15, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.6, 0.2, 0.15, 0.05]),
        
        # Vitals
        'sbp': np.random.normal(130, 20, n),
        'dbp': np.random.normal(80, 10, n),
        'hr': np.random.normal(75, 12, n),
        
        # Labs
        'glucose': np.random.lognormal(4.5, 0.3, n),
        'hba1c': np.random.lognormal(1.7, 0.2, n),
        'creatinine': np.random.lognormal(0, 0.4, n)
    })
    
    # Define variable types
    categorical = ['gender', 'race']
    normal = ['age', 'sbp', 'dbp', 'hr']
    nonnormal = ['glucose', 'hba1c', 'creatinine']
    
    # Define variable renaming for display
    rename_dict = {
        'age': 'Age (years)',
        'gender': 'Sex',
        'race': 'Race/Ethnicity',
        'sbp': 'Systolic BP (mmHg)',
        'dbp': 'Diastolic BP (mmHg)',
        'hr': 'Heart Rate (bpm)',
        'glucose': 'Glucose (mg/dL)',
        'hba1c': 'HbA1c (%)',
        'creatinine': 'Creatinine (mg/dL)'
    }
    
    # Create exclusion criteria
    exclusion1 = data['age'] >= 18
    exclusion2 = data['glucose'] < 200
    
    print("Test 1: Default ordering (no order_vars)")
    print("-----------------------------------------")
    
    # Initialize EquiFlow without order_vars
    ef_default = EquiFlow(
        data=data,
        categorical=categorical,
        normal=normal,
        nonnormal=nonnormal,
        rename=rename_dict,
        initial_cohort_label="Initial Sample"
    )
    
    # Add exclusions
    ef_default.add_exclusion(mask=exclusion1, exclusion_reason="Age < 18")
    ef_default.add_exclusion(mask=exclusion2, exclusion_reason="Glucose >= 200")
    
    # Get table characteristics
    table_default = ef_default.view_table_characteristics()
    
    # Print variable order from default table (excluding Overall)
    default_variables = [v for v in table_default.index.get_level_values(0).unique() if v != 'Overall']
    print("Default variable order (with format suffixes):")
    for i, var in enumerate(default_variables, 1):
        print(f"{i}. {var}")
    
    print_separator()
    print("Test 2: Custom ordering with order_vars")
    print("---------------------------------------")
    
    # Define custom order (mixing categories deliberately)
    custom_order = [
        'age',      # normal
        'gender',   # categorical
        'glucose',  # nonnormal
        'sbp',      # normal
        'dbp',      # normal
        'race',     # categorical
        'hr',       # normal
        'hba1c',    # nonnormal
        'creatinine'  # nonnormal
    ]
    
    # Convert original variable names to renamed versions for comparison
    custom_order_renamed = [rename_dict.get(var, var) for var in custom_order]
    
    # Initialize EquiFlow with order_vars
    ef_ordered = EquiFlow(
        data=data,
        categorical=categorical,
        normal=normal,
        nonnormal=nonnormal,
        rename=rename_dict,
        order_vars=custom_order,
        initial_cohort_label="Initial Sample"
    )
    
    # Add the same exclusions
    ef_ordered.add_exclusion(mask=exclusion1, exclusion_reason="Age < 18")
    ef_ordered.add_exclusion(mask=exclusion2, exclusion_reason="Glucose >= 200")
    
    # Get table characteristics with custom order
    table_ordered = ef_ordered.view_table_characteristics()
    
    # Print variable order from ordered table (excluding Overall)
    ordered_variables = [v for v in table_ordered.index.get_level_values(0).unique() if v != 'Overall']
    print("Custom variable order (with format suffixes):")
    for i, var in enumerate(ordered_variables, 1):
        print(f"{i}. {var}")
    
    # Normalize variable names for comparison (strip suffixes)
    ordered_variables_normalized = [strip_label_suffix(var) for var in ordered_variables]
    
    print("\nComparing variable orders (without suffixes):")
    print(f"Expected: {custom_order_renamed}")
    print(f"Actual (normalized): {ordered_variables_normalized}")
    
    print_separator()
    print("Test 3: Overriding order_vars in view_table_characteristics()")
    print("-----------------------------------------------------------")
    
    # Define a different order for this specific table
    override_order = [
        'gender',   # categorical at the top
        'race',     # categorical next
        'glucose',  # nonnormal
        'hba1c',    # nonnormal
        'creatinine',  # nonnormal
        'age',      # normal
        'sbp',      # normal
        'dbp',      # normal
        'hr'        # normal
    ]
    
    # Convert original variable names to renamed versions for comparison
    override_order_renamed = [rename_dict.get(var, var) for var in override_order]
    
    # Get table characteristics with overridden order
    table_override = ef_ordered.view_table_characteristics(order_vars=override_order)
    
    # Print variable order from overridden table (excluding Overall)
    override_variables = [v for v in table_override.index.get_level_values(0).unique() if v != 'Overall']
    print("Overridden variable order (with format suffixes):")
    for i, var in enumerate(override_variables, 1):
        print(f"{i}. {var}")
    
    # Normalize variable names for comparison (strip suffixes)
    override_variables_normalized = [strip_label_suffix(var) for var in override_variables]
    
    print("\nComparing variable orders (without suffixes):")
    print(f"Expected: {override_order_renamed}")
    print(f"Actual (normalized): {override_variables_normalized}")
    
    print_separator()
    print("Test 4: Verifying order across different table types")
    print("-------------------------------------------------")
    
    # Get drifts table with original custom order
    drifts_table = ef_ordered.view_table_drifts()
    
    # Get p-values table with original custom order
    pvalues_table = ef_ordered.view_table_pvalues()
    
    # Print variable order from drifts table (excluding Overall)
    drifts_variables = [v for v in drifts_table.index.get_level_values(0).unique() if v != 'Overall']
    print("Drifts table variable order:")
    for i, var in enumerate(drifts_variables, 1):
        print(f"{i}. {var}")
    
    print("\nP-values table variable order:")
    pvalues_variables = [v for v in pvalues_table.index.get_level_values(0).unique() if v != 'Overall']
    for i, var in enumerate(pvalues_variables, 1):
        print(f"{i}. {var}")
    
    # Normalize variable names for comparison (strip suffixes)
    drifts_variables_normalized = [strip_label_suffix(var) for var in drifts_variables]
    pvalues_variables_normalized = [strip_label_suffix(var) for var in pvalues_variables]
    
    print("\nComparing variable orders (without suffixes):")
    print(f"Expected: {custom_order_renamed}")
    print(f"Drifts table (normalized): {drifts_variables_normalized}")
    print(f"P-values table (normalized): {pvalues_variables_normalized}")
    
    print_separator()
    print("Verification Summary")
    print("-------------------")
    
    # Check if Overall is first in all tables
    overall_first_default = table_default.index.get_level_values(0)[0] == 'Overall'
    overall_first_ordered = table_ordered.index.get_level_values(0)[0] == 'Overall'
    overall_first_override = table_override.index.get_level_values(0)[0] == 'Overall'
    
    print(f"1. 'Overall' appears first in default table: {overall_first_default}")
    print(f"2. 'Overall' appears first in ordered table: {overall_first_ordered}")
    print(f"3. 'Overall' appears first in override table: {overall_first_override}")
    
    # Check if custom order is respected (comparing without suffixes)
    ordered_matches_custom = ordered_variables_normalized == custom_order_renamed
    
    print(f"4. Ordered table follows custom_order (ignoring suffixes): {ordered_matches_custom}")
    if not ordered_matches_custom:
        print(f"   Expected: {custom_order_renamed}")
        print(f"   Actual: {ordered_variables_normalized}")
    
    # Check if override order is respected (comparing without suffixes)
    override_matches = override_variables_normalized == override_order_renamed
    
    print(f"5. Override table follows override_order (ignoring suffixes): {override_matches}")
    if not override_matches:
        print(f"   Expected: {override_order_renamed}")
        print(f"   Actual: {override_variables_normalized}")
    
    # Check if drift and p-values tables follow custom order
    drifts_matches_custom = drifts_variables_normalized == custom_order_renamed
    pvalues_matches_custom = pvalues_variables_normalized == custom_order_renamed
    
    print(f"6. Drifts table follows custom_order (ignoring suffixes): {drifts_matches_custom}")
    if not drifts_matches_custom:
        print(f"   Expected: {custom_order_renamed}")
        print(f"   Actual: {drifts_variables_normalized}")
    
    print(f"7. P-values table follows custom_order (ignoring suffixes): {pvalues_matches_custom}")
    if not pvalues_matches_custom:
        print(f"   Expected: {custom_order_renamed}")
        print(f"   Actual: {pvalues_variables_normalized}")

    # Test passed if all checks are true
    test_passed = all([
        overall_first_default,
        overall_first_ordered, 
        overall_first_override,
        ordered_matches_custom,
        override_matches,
        drifts_matches_custom,
        pvalues_matches_custom
    ])
    
    print_separator()
    print(f"Overall Test Result: {'PASSED' if test_passed else 'FAILED'}")

if __name__ == "__main__":
    test_order_vars()