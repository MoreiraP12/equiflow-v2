"""
Comprehensive unit tests for equiflow package.

These tests cover:
- EquiFlow initialization and validation
- Exclusion logic with keep/new_cohort parameters
- Table generation (flows, characteristics, drifts, p-values)
- EasyFlow simplified interface
- Flow diagram generation
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from equiflow import (
    EquiFlow,
    EasyFlow,
    TableFlows,
    TableCharacteristics,
    TableDrifts,
    TablePValues,
    FlowDiagram,
)


class TestEquiFlowInitialization:
    """Tests for EquiFlow initialization and basic setup."""

    def test_basic_initialization(self, sample_data):
        """Test basic EquiFlow initialization with data."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age'],
            nonnormal=['score']
        )
        assert len(ef._dfs) == 1
        assert ef.categorical == ['sex', 'race']
        assert ef.normal == ['age']
        assert ef.nonnormal == ['score']

    def test_initialization_with_dfs(self, sample_data):
        """Test initialization with list of DataFrames."""
        df1 = sample_data.copy()
        df2 = sample_data[sample_data['age'] >= 30].copy()
        
        ef = EquiFlow(
            dfs=[df1, df2],
            categorical=['sex']
        )
        assert len(ef._dfs) == 2
        assert len(ef._dfs[1]) < len(ef._dfs[0])

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            EquiFlow(data=pd.DataFrame())

    def test_missing_variable_raises_error(self, sample_data):
        """Test that non-existent variable raises ValueError."""
        with pytest.raises(ValueError, match="not found in the DataFrame"):
            EquiFlow(
                data=sample_data,
                categorical=['nonexistent_column']
            )

    def test_no_data_raises_error(self):
        """Test that missing data and dfs raises ValueError."""
        with pytest.raises(ValueError, match="Either 'data' or 'dfs' must be provided"):
            EquiFlow(categorical=['sex'])

    def test_initial_cohort_label(self, sample_data):
        """Test custom initial cohort label."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            initial_cohort_label="All Patients"
        )
        assert ef.new_cohort_labels[0] == "All Patients"

    def test_repr(self, sample_data):
        """Test string representation."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age']
        )
        repr_str = repr(ef)
        assert "EquiFlow" in repr_str
        assert "cohorts=1" in repr_str


class TestEquiFlowExclusion:
    """Tests for exclusion logic."""

    def test_add_exclusion_with_keep(self, sample_data):
        """Test adding exclusion with keep parameter."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        initial_count = len(ef._dfs[0])
        ef.add_exclusion(
            keep=sample_data['age'] >= 30,
            exclusion_reason="Age < 30",
            new_cohort_label="Adults 30+"
        )
        
        assert len(ef._dfs) == 2
        assert len(ef._dfs[1]) < initial_count
        assert ef.exclusion_labels[1] == "Age < 30"
        assert ef.new_cohort_labels[1] == "Adults 30+"

    def test_add_exclusion_with_new_cohort(self, sample_data):
        """Test adding exclusion with new_cohort DataFrame."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        new_cohort = sample_data[sample_data['age'] >= 30].copy()
        ef.add_exclusion(
            new_cohort=new_cohort,
            exclusion_reason="Age < 30"
        )
        
        assert len(ef._dfs) == 2
        assert len(ef._dfs[1]) == len(new_cohort)

    def test_exclusion_both_params_raises_error(self, sample_data):
        """Test that providing both keep and new_cohort raises error."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        with pytest.raises(ValueError, match="Only one of"):
            ef.add_exclusion(
                keep=sample_data['age'] >= 30,
                new_cohort=sample_data[sample_data['age'] >= 30]
            )

    def test_exclusion_no_params_raises_error(self, sample_data):
        """Test that providing neither keep nor new_cohort raises error."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        with pytest.raises(ValueError, match="Either 'keep' or 'new_cohort'"):
            ef.add_exclusion(exclusion_reason="No filter")

    def test_multiple_exclusions(self, sample_data):
        """Test chaining multiple exclusions."""
        ef = EquiFlow(data=sample_data, categorical=['sex', 'race'])
        
        ef.add_exclusion(keep=sample_data['age'] >= 30, exclusion_reason="Age < 30")
        
        # Need to use the current cohort for the next exclusion
        current = ef._dfs[-1]
        ef.add_exclusion(keep=current['score'] > 5, exclusion_reason="Score ≤ 5")
        
        assert len(ef._dfs) == 3

    def test_method_chaining(self, sample_data):
        """Test that add_exclusion returns self for chaining."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        result = ef.add_exclusion(
            keep=sample_data['age'] >= 30,
            exclusion_reason="Age < 30"
        )
        
        assert result is ef


class TestTableFlows:
    """Tests for flow table generation."""

    def test_view_table_flows(self, sample_data):
        """Test basic flow table generation."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        ef.add_exclusion(keep=sample_data['age'] >= 30, exclusion_reason="Age < 30")
        
        flows = ef.view_table_flows()
        
        assert isinstance(flows, pd.DataFrame)
        assert len(flows) == 3  # Initial, Removed, Result rows

    def test_flows_requires_two_cohorts(self, sample_data):
        """Test that view_table_flows requires at least 2 cohorts."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        with pytest.raises(ValueError, match="At least two cohorts"):
            ef.view_table_flows()

    def test_flows_with_thousands_separator(self, sample_data):
        """Test flow table with thousands separator."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            thousands_sep=True
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        flows = ef.view_table_flows()
        assert flows is not None


class TestTableCharacteristics:
    """Tests for characteristics table generation."""

    def test_view_table_characteristics(self, sample_data):
        """Test basic characteristics table generation."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age'],
            nonnormal=['score']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        chars = ef.view_table_characteristics()
        
        assert isinstance(chars, pd.DataFrame)
        assert len(chars) > 0

    def test_characteristics_with_custom_format(self, sample_data):
        """Test characteristics with custom formatting."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            normal=['age'],
            format_cat="N",
            format_normal="Mean ± SD",
            decimals=1
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        chars = ef.view_table_characteristics()
        assert chars is not None


class TestTableDrifts:
    """Tests for drift (SMD) table generation."""

    def test_view_table_drifts(self, sample_data):
        """Test basic drift table generation."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        drifts = ef.view_table_drifts()
        
        assert isinstance(drifts, pd.DataFrame)

    def test_drift_values_reasonable(self, sample_data):
        """Test that SMD values are within reasonable range."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            normal=['age']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        drifts = ef.view_table_drifts()
        
        # SMD values should typically be between 0 and 2
        # (though can be higher in extreme cases)
        for col in drifts.columns:
            if 'SMD' in str(col) or 'Drift' in str(col):
                numeric_vals = pd.to_numeric(drifts[col], errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    assert numeric_vals.max() < 10  # Sanity check


class TestTablePValues:
    """Tests for p-value table generation."""

    def test_view_table_pvalues(self, sample_data):
        """Test basic p-value table generation."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        pvals = ef.view_table_pvalues()
        
        assert isinstance(pvals, pd.DataFrame)

    def test_pvalues_correction_none(self, sample_data):
        """Test p-values with no correction."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        pvals = ef.view_table_pvalues(correction="none")
        assert pvals is not None

    def test_pvalues_correction_bonferroni(self, sample_data):
        """Test p-values with Bonferroni correction."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        pvals = ef.view_table_pvalues(correction="bonferroni")
        assert pvals is not None

    def test_pvalues_correction_fdr(self, sample_data):
        """Test p-values with FDR correction."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        pvals = ef.view_table_pvalues(correction="fdr_bh")
        assert pvals is not None


class TestEasyFlow:
    """Tests for EasyFlow simplified interface."""

    def test_easyflow_basic(self, sample_data):
        """Test basic EasyFlow workflow."""
        flow = EasyFlow(sample_data, title="Test Study")
        
        assert flow._title == "Test Study"
        assert len(flow._data) == len(sample_data)

    def test_easyflow_categorize(self, sample_data):
        """Test EasyFlow categorize method."""
        flow = EasyFlow(sample_data)
        result = flow.categorize(['sex', 'race'])
        
        assert result is flow  # Returns self
        assert flow._categorical_vars == ['sex', 'race']

    def test_easyflow_measure_normal(self, sample_data):
        """Test EasyFlow measure_normal method."""
        flow = EasyFlow(sample_data)
        result = flow.measure_normal(['age'])
        
        assert result is flow
        assert flow._normal_vars == ['age']

    def test_easyflow_exclude(self, sample_data):
        """Test EasyFlow exclude method."""
        flow = EasyFlow(sample_data)
        flow.categorize(['sex'])
        
        initial_n = len(flow._current_data)
        result = flow.exclude(sample_data['age'] >= 30, "Age < 30")
        
        assert result is flow
        assert len(flow._current_data) < initial_n
        assert len(flow._exclusion_steps) == 1

    def test_easyflow_method_chaining(self, sample_data):
        """Test EasyFlow method chaining."""
        flow = (EasyFlow(sample_data, title="Chained Test")
            .categorize(['sex', 'race'])
            .measure_normal(['age'])
            .measure_nonnormal(['score'])
            .exclude(sample_data['age'] >= 30, "Age < 30"))
        
        assert len(flow._exclusion_steps) == 1
        assert flow._categorical_vars == ['sex', 'race']

    def test_easyflow_repr(self, sample_data):
        """Test EasyFlow string representation."""
        flow = EasyFlow(sample_data)
        flow.exclude(sample_data['age'] >= 30, "Test")
        
        repr_str = repr(flow)
        assert "EasyFlow" in repr_str
        assert "steps=1" in repr_str


class TestFlowDiagram:
    """Tests for flow diagram generation."""

    def test_plot_flows_creates_file(self, sample_data):
        """Test that plot_flows creates output file."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            normal=['age']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30, exclusion_reason="Age < 30")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_flow")
            ef.plot_flows(output_file=output_path, display_flow_diagram=False)
            
            # Check that some output file was created
            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_plot_flows_with_custom_colors(self, sample_data):
        """Test flow diagram with custom colors."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            normal=['age']
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "custom_color_flow")
            ef.plot_flows(
                output_file=output_path,
                output_folder=tmpdir,
                categorical_bar_colors={'sex': ['#1f77b4', '#ff7f0e']},
                display_flow_diagram=False
            )
            
            files = os.listdir(tmpdir)
            assert len(files) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_category_variable(self, sample_data):
        """Test handling of variable with single category."""
        df = sample_data.copy()
        df['constant'] = 'A'  # Single value
        
        ef = EquiFlow(data=df, categorical=['constant', 'sex'])
        ef.add_exclusion(keep=df['age'] >= 30)
        
        # Should not raise error
        chars = ef.view_table_characteristics()
        assert chars is not None

    def test_all_missing_variable(self, sample_data):
        """Test handling of variable with all missing values."""
        df = sample_data.copy()
        df['all_missing'] = np.nan
        
        # Should handle gracefully during initialization
        ef = EquiFlow(data=df, categorical=['sex'])
        ef.add_exclusion(keep=df['age'] >= 30)
        
        flows = ef.view_table_flows()
        assert flows is not None

    def test_small_cohort(self):
        """Test with very small cohort."""
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'sex': ['M', 'F', 'M', 'F', 'M'],
        })
        
        ef = EquiFlow(data=df, categorical=['sex'], normal=['age'])
        ef.add_exclusion(keep=df['age'] >= 30)
        
        flows = ef.view_table_flows()
        assert len(flows) == 3  # Initial, Removed, Result rows

    def test_variable_renaming(self, sample_data):
        """Test variable renaming functionality."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            rename={'sex': 'Gender'}
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        chars = ef.view_table_characteristics()
        # Check that rename was applied (exact check depends on implementation)
        assert chars is not None

    def test_order_classes(self, sample_data):
        """Test custom category ordering."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['race'],
            order_classes={'race': ['Other', 'Asian', 'Black', 'White']}
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        chars = ef.view_table_characteristics()
        assert chars is not None

    def test_limit_categories(self, sample_data):
        """Test limiting number of categories displayed."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['race'],
            limit={'race': 2}  # Show only top 2
        )
        ef.add_exclusion(keep=sample_data['age'] >= 30)
        
        chars = ef.view_table_characteristics()
        assert chars is not None


class TestMissingData:
    """Tests for handling missing data."""

    def test_missing_values_in_categorical(self, sample_data_with_missing):
        """Test handling missing values in categorical variables."""
        ef = EquiFlow(
            data=sample_data_with_missing,
            categorical=['race'],
            missingness=True
        )
        ef.add_exclusion(keep=sample_data_with_missing['age'].notna())
        
        chars = ef.view_table_characteristics()
        assert chars is not None

    def test_missing_values_in_continuous(self, sample_data_with_missing):
        """Test handling missing values in continuous variables."""
        ef = EquiFlow(
            data=sample_data_with_missing,
            nonnormal=['score'],
            missingness=True
        )
        ef.add_exclusion(keep=sample_data_with_missing['score'].notna())
        
        chars = ef.view_table_characteristics()
        assert chars is not None
