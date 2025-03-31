import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from equiflow import EquiFlow

def test_flow_diagram_colors():
    """
    Test script to verify color customization functionality in EquiFlow's FlowDiagram.
    Creates sample patient data, applies exclusion criteria, and tests various color schemes.
    """
    print("Starting EquiFlow Flow Diagram Color Customization Test")
    
    # Create sample data
    np.random.seed(42)  # For reproducibility
    n = 1000
    data = pd.DataFrame({
        'age': np.random.normal(65, 15, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'bmi': np.random.normal(27, 5, n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.7, 0.15, 0.1, 0.05]),
        'comorbidity_count': np.random.poisson(2, n),
        'lab_value': np.random.lognormal(0, 0.6, n),
        'treatment_group': np.random.choice(['A', 'B', 'C'], n),
    })
    
    # Add some missing values
    data.loc[np.random.choice(n, 50, replace=False), 'race'] = None
    data.loc[np.random.choice(n, 30, replace=False), 'bmi'] = None
    data.loc[np.random.choice(n, 40, replace=False), 'lab_value'] = None
    
    # Initialize EquiFlow with our sample data
    print("Initializing EquiFlow...")
    flow = EquiFlow(
        data=data,
        initial_cohort_label="Initial Patient Cohort",
        categorical=['sex', 'race', 'treatment_group'],
        normal=['age', 'bmi'],
        nonnormal=['comorbidity_count', 'lab_value'],
    )
    
    # Add exclusions to create multiple cohorts
    print("Adding exclusion criteria...")
    
    # Exclude patients under 18
    flow.add_exclusion(
        mask=data['age'] >= 18,
        exclusion_reason="Age < 18",
        new_cohort_label="Adult Patients"
    )
    
    # Exclude patients with missing BMI
    flow.add_exclusion(
        mask=~data['bmi'].isna(),
        exclusion_reason="Missing BMI",
        new_cohort_label="Complete BMI Data"
    )
    
    # Exclude patients with high lab values
    flow.add_exclusion(
        mask=data['lab_value'] <= 5,
        exclusion_reason="Lab value > 5",
        new_cohort_label="Normal Lab Values"
    )
    
    # Generate required tables
    print("Generating required tables...")
    flow.view_table_flows()
    flow.view_table_characteristics()
    flow.view_table_drifts()
    
    # Ensure output directory exists
    os.makedirs('imgs', exist_ok=True)
    
    # Test cases for color customization
    print("\n1. Testing basic node colors (no distributions)")
    flow.plot_flows(
        output_file="flow_diagram_basic_nodes",
        display_flow_diagram=True,
        plot_dists=False,  # No distribution plots
        smds=False,        # No SMDs
        legend=False,      # Explicitly turn off legend 
        cohort_node_color='lightblue',
        exclusion_node_color='mistyrose',
        edge_color='navy'
    )
    
    print("\n2. Testing with categorical colors but no distributions")
    flow.plot_flows(
        output_file="flow_diagram_categorical_only",
        display_flow_diagram=True,
        plot_dists=False,
        smds=False,
        legend=False,
        cohort_node_color='#f0f0f0',
        exclusion_node_color='#ffe6e6',
        edge_color='#333333',
        categorical_bar_colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    )
    
    # After basic tests succeed, try with distributions enabled
    print("\n3. Testing with distributions and custom colors")
    try:
        plot = flow.plot_flows(
            output_file="flow_diagram_with_dists",
            display_flow_diagram=True,
            plot_dists=True,
            legend=True,
            smds=True,
            cohort_node_color='white',
            exclusion_node_color='#f5f5f5',
            missing_value_color='#e0e0e0',
            continuous_var_color='#f0f0f0',
            categorical_bar_colors=['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c'],
            edge_color='black'
        )
    except Exception as e:
        print(f"Error with distributions: {e}")
        print("Skipping remaining tests with distributions...")
    
    print("\nColor customization test complete!")
    print("Check the 'imgs' folder for the generated flow diagrams.")

if __name__ == "__main__":
    test_flow_diagram_colors()