from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import pandas as pd
import numpy as np
import os
import inspect
import re

# Import from equiflow package
from equiflow import EquiFlow


class EasyFlow:
    """
    A simplified interface for creating equity-focused cohort flow diagrams,
    building on top of the EquiFlow package with a more user-friendly API.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        title: str = "Cohort Selection",
        auto_detect: bool = True
    ):
        """
        Initialize EasyFlow with a DataFrame and optional title.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The initial cohort dataframe
        title : str, optional
            Title for the flow diagram
        auto_detect : bool, optional
            Whether to automatically detect variable types
        """
        self.data = data
        self.title = title
        self.auto_detect = auto_detect
        
        # Initialize empty lists for variable types
        self._categorical_vars = []
        self._normal_vars = []
        self._nonnormal_vars = []
        
        # Track exclusion steps
        self._exclusion_steps = []
        self._current_data = data
        
        # Store outputs
        self.flow_table = None
        self.characteristics = None
        self.drifts = None
        self.diagram = None
        
        # Stored EquiFlow instance
        self._equiflow = None
        
        # Auto-detect variable types if enabled
        if auto_detect:
            self._detect_variable_types()
    
    def _detect_variable_types(self):
        """
        Automatically detect variable types in the dataframe.
        - Categorical: object, bool, or low cardinality numeric
        - Normal: numeric variables that pass Shapiro-Wilk test
        - Non-normal: numeric variables that fail Shapiro-Wilk test
        """
        # First try the basic categorization without scipy
        for col in self.data.columns:
            # Skip columns with too many missing values
            if self.data[col].isna().mean() > 0.5:
                continue
                
            # Check if column is categorical
            if self.data[col].dtype == 'object' or self.data[col].dtype == 'bool':
                self._categorical_vars.append(col)
                continue
                
            # Check for low cardinality numeric (likely categorical)
            if self.data[col].dtype.kind in 'ifu' and len(self.data[col].unique()) <= 10:
                self._categorical_vars.append(col)
                continue
                
            # For numeric columns, add to non-normal by default
            if self.data[col].dtype.kind in 'ifu':
                self._nonnormal_vars.append(col)
        
        # Now try to improve categorization with scipy if available
        try:
            from scipy import stats
            
            # Clear the normal list to rebuild it
            self._normal_vars = []
            
            # Keep track of columns to remove from non-normal list
            to_remove = []
            
            for col in self._nonnormal_vars:
                # Get non-null values
                values = self.data[col].dropna()
                
                # Sample if necessary to speed up test
                if len(values) > 5000:
                    values = values.sample(5000, random_state=42)
                
                # Skip if too few values
                if len(values) < 8:
                    continue
                    
                # Test for normality
                try:
                    _, p_value = stats.shapiro(values)
                    if p_value > 0.05:  # Normal distribution
                        self._normal_vars.append(col)
                        to_remove.append(col)
                except:
                    # If test fails, keep as non-normal
                    pass
            
            # Remove columns from non-normal that are now in normal
            self._nonnormal_vars = [col for col in self._nonnormal_vars if col not in to_remove]
            
        except ImportError:
            print("Note: scipy not found - using simplified variable type detection.")
            print("Install scipy for better detection of normal vs non-normal variables.")
            print("  pip install scipy")
        
        # Print helpful message
        print(f"Auto-detected {len(self._categorical_vars)} categorical, {len(self._normal_vars)} normal, and {len(self._nonnormal_vars)} non-normal variables.")
        print("You can customize these with .categorize(), .measure_normal(), and .measure_nonnormal() methods.")
    
    def categorize(self, variables: List[str]):
        """
        Define categorical variables to include in the analysis.
        
        Parameters:
        -----------
        variables : List[str]
            List of column names to treat as categorical
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        self._categorical_vars = variables
        return self
    
    def measure_normal(self, variables: List[str]):
        """
        Define normally distributed variables to include in the analysis.
        
        Parameters:
        -----------
        variables : List[str]
            List of column names of normally distributed variables
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        self._normal_vars = variables
        return self
    
    def measure_nonnormal(self, variables: List[str]):
        """
        Define non-normally distributed variables to include in the analysis.
        
        Parameters:
        -----------
        variables : List[str]
            List of column names of non-normally distributed variables
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        self._nonnormal_vars = variables
        return self
    
    def _generate_exclusion_label(self, mask):
        """
        Generate a user-friendly label from a pandas mask condition.
        """
        # Try to extract code from the mask if possible
        try:
            mask_str = inspect.getsource(mask.func if hasattr(mask, 'func') else mask)
            # Clean up the code to make a readable label
            mask_str = mask_str.strip()
            
            # Extract condition from common patterns
            condition_pattern = r'data\["([^"]+)"\]\s*([<>=!]+)\s*([^\s]+)'
            match = re.search(condition_pattern, mask_str)
            
            if match:
                column, operator, value = match.groups()
                return f"{column} {operator} {value}"
            
            return "Exclusion criteria"
        except:
            return "Exclusion criteria"
    
    def exclude(self, condition, label: Optional[str] = None):
        """
        Add an exclusion step based on a boolean condition.
        
        Parameters:
        -----------
        condition : pandas.Series or callable
            Boolean mask to select the subset of data to keep, or a function
            that takes a dataframe and returns a boolean mask
        label : str, optional
            Label describing the exclusion. If not provided, tries to generate one.
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        # Generate label if not provided
        if label is None:
            label = self._generate_exclusion_label(condition)
        
        # Apply the mask directly if it's already a boolean mask
        if isinstance(condition, pd.Series) and condition.dtype == bool:
            # We need to make sure the indices match
            # First, create a mask with the same index as the current data
            aligned_mask = pd.Series(False, index=self._current_data.index)
            
            # Then, update the mask with the condition where indices match
            common_indices = self._current_data.index.intersection(condition.index)
            aligned_mask.loc[common_indices] = condition.loc[common_indices]
            
            # Apply the mask
            new_data = self._current_data[aligned_mask]
        
        # If it's a callable, apply it to the current data
        elif callable(condition):
            mask = condition(self._current_data)
            new_data = self._current_data[mask]
        
        # If it's neither, try direct boolean indexing
        else:
            try:
                new_data = self._current_data[condition]
            except Exception as e:
                raise ValueError(f"Invalid exclusion condition: {e}")
        
        # Store the step information and a copy of the condition
        self._exclusion_steps.append({
            'previous_data': self._current_data.copy(),
            'condition': condition,
            'new_data': new_data.copy(),
            'label': label
        })
        
        # Update current data
        self._current_data = new_data
        
        return self
    
    def _create_equiflow(self):
        """
        Create and configure the underlying EquiFlow instance.
        """
        # Create initial EquiFlow instance with the original data
        ef = EquiFlow(
            data=self.data,
            initial_cohort_label=self.title,
            categorical=self._categorical_vars,
            normal=self._normal_vars,
            nonnormal=self._nonnormal_vars
        )
        
        # Start with the complete original dataset
        current_df = self.data.copy()
        
        # Add each exclusion step
        for i, step in enumerate(self._exclusion_steps):
            # Apply condition to get the new subset
            if isinstance(step['condition'], pd.Series) and step['condition'].dtype == bool:
                # Create a boolean mask that's the same size as the current data
                indices_to_keep = current_df.index.intersection(step['new_data'].index)
                mask = current_df.index.isin(indices_to_keep)
            elif callable(step['condition']):
                # Apply the callable to the current data
                mask = step['condition'](current_df)
            else:
                # Try direct application
                try:
                    mask = step['condition']
                except:
                    # Fallback - just use indices
                    indices_to_keep = current_df.index.intersection(step['new_data'].index)
                    mask = current_df.index.isin(indices_to_keep)
                
            # Add the exclusion to EquiFlow
            ef.add_exclusion(
                mask=mask,
                exclusion_reason=step['label'],
                new_cohort_label=f"Step {i+1}"
            )
            
            # Update the current dataframe for the next iteration
            current_df = current_df[mask]
        
        self._equiflow = ef
        return ef
    
    def generate(self, output: str = "flow_diagram", show: bool = True, format: str = "pdf"):
        """
        Generate the flow diagram and tables.
        
        Parameters:
        -----------
        output : str, optional
            Base name for output files
        show : bool, optional
            Whether to display the flow diagram
        format : str, optional
            Output format for the diagram (pdf, svg, png)
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        # Create EquiFlow instance if not already created
        if self._equiflow is None:
            ef = self._create_equiflow()
        else:
            ef = self._equiflow
        
        # Get tables
        self.flow_table = ef.view_table_flows()
        self.characteristics = ef.view_table_characteristics()
        self.drifts = ef.view_table_drifts()
        
        # Generate and save the diagram
        ef.plot_flows(
            output_file=output,
            display_flow_diagram=show
        )
        
        # Store the diagram path for reference
        self.diagram = os.path.join("imgs", f"{output}.{format}")
        
        return self
    
    @classmethod
    def quick_flow(cls, 
                  data: pd.DataFrame, 
                  exclusions: List[Tuple[Union[pd.Series, Callable], str]], 
                  output: str = "flow_diagram",
                  auto_detect: bool = True):
        """
        Quickly create a flow diagram with a list of exclusion steps.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Initial cohort data
        exclusions : List[Tuple[Union[pd.Series, Callable], str]]
            List of (condition, label) pairs defining exclusions
            The condition can be a boolean mask or a function that takes a dataframe and returns a mask
        output : str, optional
            Base name for output files
        auto_detect : bool, optional
            Whether to auto-detect variable types
            
        Returns:
        --------
        flow : EasyFlow
            The configured EasyFlow instance
        """
        # Create EasyFlow instance
        flow = cls(data, auto_detect=auto_detect)
        
        # Add all exclusion steps
        for condition, label in exclusions:
            flow.exclude(condition, label)
        
        # Generate the diagram
        flow.generate(output=output)
        
        return flow


# Example usage of the quick_flow method:
# 
# data = pd.read_csv("patients.csv")
# exclusions = [
#     (data["age"] >= 18, "Age < 18 years"),
#     (data["consent"] == "yes", "No consent"),
#     (data["complete_data"] == True, "Incomplete data")
# ]
# 
# flow = EasyFlow.quick_flow(data, exclusions, output="patient_diagram")

# Example usage of the quick_flow method:
# 
# data = pd.read_csv("patients.csv")
# exclusions = [
#     (data["age"] >= 18, "Age < 18 years"),
#     (data["consent"] == "yes", "No consent"),
#     (data["complete_data"] == True, "Incomplete data")
# ]
# 
# flow = EasyFlow.quick_flow(data, exclusions, output="patient_diagram")

import pandas as pd
import numpy as np
import os
import sys

# Check required packages and install if necessary
required_packages = ['scipy']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing required packages: {', '.join(missing_packages)}")
    print("The example will continue but some functionality may be limited.")
    print("To install missing packages, run:")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\n")

# Import our package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))

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

""" # Method 1: Using the fluent interface
print("Creating flow diagram using the fluent interface...")
flow = (EasyFlow(data, title="Patient Selection")
    .categorize(["gender", "race"])
    .measure_normal(["age", "bmi"])
    .measure_nonnormal(["los_days", "cost"])
    .exclude(data["age"] >= 18, "Removed patients under 18 years")
    .exclude(data["consent"] == "yes", "Excluded non-consenting patients")
    .exclude(data["missing_data"] == True, "Removed records with missing data")
    .generate(output="patient_selection_diagram")
)
 """
""" flow.diagram

print("\nAccessing results:")
print(f"Flow table shape: {flow.flow_table.shape}")
print(f"Characteristics table shape: {flow.characteristics.shape}")
print(f"Drifts table shape: {flow.drifts.shape}")
print(f"Diagram saved to: {flow.diagram}") 

# Method 2: Using the quick_flow method
print("\nCreating flow diagram using the quick_flow method...") """

exclusions = [
    (data["age"] >= 18, "Age < 18 years"),
    (data["consent"] == "yes", "No consent"),
    (data["missing_data"] == True, "Complete data only")
]

quick_flow = EasyFlow.quick_flow(
    data, 
    exclusions=exclusions, 
    output="quick_flow_diagram"
)

print(f"\nQuick flow diagram saved to: {quick_flow.diagram}")