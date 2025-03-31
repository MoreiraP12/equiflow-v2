import pandas as pd
import numpy as np
from equiflow import EquiFlow

# Create a small test dataset with clear categorical variables
def create_test_data(n=500):
    np.random.seed(42)  # For reproducibility
    
    data = pd.DataFrame({
        # Demographics
        'age': np.random.normal(60, 15, n).round().astype(int),
        'sex': np.random.choice(['Male', 'Female'], n),
        
        # Race with many categories to test limiting
        'race': np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Pacific Islander', 'Other'], 
            n, p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]
        ),
        
        # Binary variables to test limit=1
        'hypertension': np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6]),
        'diabetes': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        
        # Simple numeric values
        'bmi': np.random.normal(27, 5, n).round(1)
    })
    
    # Add some missing values
    for col in data.columns:
        missing_mask = np.random.choice([True, False], n, p=[0.05, 0.95])
        data.loc[missing_mask, col] = None
    
    return data

# Create the test data
print("Creating test dataset...")
data = create_test_data()
print(f"Dataset shape: {data.shape}")

# Print value counts for race to verify later
print("\nRace distribution:")
print(data['race'].value_counts())

# Initialize a basic EquiFlow object
ef = EquiFlow(
    data=data,
    initial_cohort_label="Initial Cohort",
    categorical=['sex', 'race', 'hypertension', 'diabetes'],
    normal=['age', 'bmi'],
    rename={'race': 'Race/Ethnicity'}  # Test with renamed variable
)

# Add simple exclusions
ef.add_exclusion(
    mask=data['age'] >= 18,
    exclusion_reason="Age < 18",
    new_cohort_label="Adults"
)

ef.add_exclusion(
    mask=data['bmi'].notna(),
    exclusion_reason="Missing BMI",
    new_cohort_label="Complete BMI"
)

# Test 1: Basic characteristics table without special parameters
print("\n*** Test 1: Basic characteristics table ***")
basic_table = ef.view_table_characteristics()
print("\nRace/Ethnicity categories (default order):")
print(basic_table.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test 2: Using limit to restrict number of categories
print("\n*** Test 2: Testing 'limit' parameter ***")
# Limit all categorical variables to 2 classes
limited_table = ef.view_table_characteristics(limit=2)
print("\nRace/Ethnicity with limit=2:")
print(limited_table.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test with specific limit for race only
print("\nRace/Ethnicity with specific limit=3:")
race_limited = ef.view_table_characteristics(limit={'race': 3})
print(race_limited.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test limit=1 for binary variables
print("\nBinary variables with limit=1:")
binary_limited = ef.view_table_characteristics(limit={'hypertension': 1, 'diabetes': 1})
print("\nHypertension:")
print(binary_limited.loc[('hypertension, N (%)', slice(None)), :])
print("\nDiabetes:")
print(binary_limited.loc[('diabetes, N (%)', slice(None)), :])

# Test 3: Using order_classes to specify category order
print("\n*** Test 3: Testing 'order_classes' parameter ***")
# Reverse the order of race categories
ordered_table = ef.view_table_characteristics(
    order_classes={'race': ['Other', 'Pacific Islander', 'Native American', 'Asian', 'Hispanic', 'Black', 'White']}
)
print("\nRace/Ethnicity with reversed order:")
print(ordered_table.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test with alphabetical ordering (useful for health equity guidelines)
print("\nRace/Ethnicity with alphabetical order:")
alpha_order = ef.view_table_characteristics(
    order_classes={'race': sorted(['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Pacific Islander', 'Other'])}
)
print(alpha_order.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test 4: Combining limit and order_classes
print("\n*** Test 4: Combining 'limit' and 'order_classes' ***")
combined = ef.view_table_characteristics(
    limit={'race': 3},
    order_classes={'race': ['Asian', 'Black', 'Hispanic', 'White', 'Native American', 'Pacific Islander', 'Other']}
)
print("\nRace/Ethnicity with limit=3 and custom order:")
print(combined.loc[('Race/Ethnicity, N (%)', slice(None)), :])

# Test 5: Using these parameters with the original variable name (not the renamed version)
print("\n*** Test 5: Using parameters with original variable names ***")
original_name = ef.view_table_characteristics(
    limit={'race': 2},  # Using original variable name
    order_classes={'race': ['Hispanic', 'Black']}  # Should show only these 2 in this order
)
print("\nRace/Ethnicity with parameters using original name 'race':")
print(original_name.loc[('Race/Ethnicity, N (%)', slice(None)), :])

print("\nTests completed!")