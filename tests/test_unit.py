"""
Unit tests for equiflow package using pytest.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from equiflow import EquiFlow, EasyFlow, TableFlows, TableCharacteristics, TableDrifts, TablePValues


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'age': np.random.normal(50, 15, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], n, p=[0.6, 0.2, 0.15, 0.05]),
        'bmi': np.random.normal(28, 5, n),
        'income': np.random.lognormal(10, 1, n),
        'los_days': np.random.exponential(5, n),
    })


@pytest.fixture
def sample_data_with_missing(sample_data):
    """Create a sample dataset with missing values."""
    data = sample_data.copy()
    np.random.seed(42)
    n = len(data)
    for col in data.columns:
        mask = np.random.rand(n) < 0.05  # 5% missing
        data.loc[mask, col] = None
    return data


@pytest.fixture
def basic_equiflow(sample_data):
    """Create a basic EquiFlow instance with one exclusion."""
    ef = EquiFlow(
        data=sample_data,
        categorical=['sex', 'race'],
        normal=['age', 'bmi'],
        nonnormal=['income', 'los_days'],
        initial_cohort_label="Initial Cohort"
    )
    ef.add_exclusion(
        keep=sample_data['age'] >= 18,
        exclusion_reason="Age < 18",
        new_cohort_label="Adults"
    )
    return ef


# ============================================================================
# EquiFlow Initialization Tests
# ============================================================================

class TestEquiFlowInitialization:
    """Tests for EquiFlow initialization."""

    def test_basic_initialization(self, sample_data):
        """Test basic initialization with data."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        assert len(ef._dfs) == 1
        assert ef.categorical == ['sex']
        assert ef.normal == []
        assert ef.nonnormal == []

    def test_initialization_with_all_var_types(self, sample_data):
        """Test initialization with all variable types."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex', 'race'],
            normal=['age', 'bmi'],
            nonnormal=['income', 'los_days']
        )
        assert ef.categorical == ['sex', 'race']
        assert ef.normal == ['age', 'bmi']
        assert ef.nonnormal == ['income', 'los_days']

    def test_initialization_with_rename(self, sample_data):
        """Test initialization with variable renaming."""
        ef = EquiFlow(
            data=sample_data,
            categorical=['sex'],
            rename={'sex': 'Gender'}
        )
        assert ef.rename == {'sex': 'Gender'}

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            EquiFlow(data=pd.DataFrame())

    def test_missing_variable_raises(self, sample_data):
        """Test that non-existent variable raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            EquiFlow(data=sample_data, categorical=['nonexistent_column'])

    def test_no_data_raises(self):
        """Test that missing data parameter raises ValueError."""
        with pytest.raises(ValueError, match="must be provided"):
            EquiFlow()

    def test_initialization_with_dfs_list(self, sample_data):
        """Test initialization with list of DataFrames."""
        df1 = sample_data.copy()
        df2 = sample_data[sample_data['age'] >= 30].copy()
        ef = EquiFlow(dfs=[df1, df2], categorical=['sex'])
        assert len(ef._dfs) == 2

    def test_repr(self, basic_equiflow):
        """Test string representation."""
        repr_str = repr(basic_equiflow)
        assert "EquiFlow" in repr_str
        assert "cohorts=" in repr_str


# ============================================================================
# EquiFlow Exclusion Tests
# ============================================================================

class TestEquiFlowExclusion:
    """Tests for EquiFlow exclusion functionality."""

    def test_add_exclusion_with_keep(self, sample_data):
        """Test adding exclusion with keep parameter."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        initial_count = len(ef._dfs[0])
        
        ef.add_exclusion(
            keep=sample_data['age'] >= 30,
            exclusion_reason="Age < 30",
            new_cohort_label="Age 30+"
        )
        
        assert len(ef._dfs) == 2
        assert len(ef._dfs[1]) < initial_count
        assert ef.exclusion_labels[1] == "Age < 30"
        assert ef.new_cohort_labels[1] == "Age 30+"

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

    def test_exclusion_chaining(self, sample_data):
        """Test that exclusions can be chained."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        
        result = ef.add_exclusion(
            keep=sample_data['age'] >= 18,
            exclusion_reason="Age < 18"
        ).add_exclusion(
            keep=sample_data['bmi'] < 40,
            exclusion_reason="BMI >= 40"
        )
        
        assert result is ef  # Returns self
        assert len(ef._dfs) == 3

    def test_exclusion_no_params_raises(self, sample_data):
        """Test that exclusion without keep or new_cohort raises."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        with pytest.raises(ValueError, match="must be provided"):
            ef.add_exclusion(exclusion_reason="Test")

    def test_exclusion_both_params_raises(self, sample_data):
        """Test that exclusion with both keep and new_cohort raises."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        with pytest.raises(ValueError, match="not both"):
            ef.add_exclusion(
                keep=sample_data['age'] >= 18,
                new_cohort=sample_data.head(10),
                exclusion_reason="Test"
            )

    def test_empty_exclusion_warning(self, sample_data):
        """Test that empty cohort after exclusion raises warning."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        with pytest.warns(UserWarning, match="empty cohort"):
            ef.add_exclusion(
                keep=sample_data['age'] > 1000,  # No one passes
                exclusion_reason="Impossible criteria"
            )


# ============================================================================
# Table Generation Tests
# ============================================================================

class TestTableFlows:
    """Tests for TableFlows functionality."""

    def test_view_table_flows(self, basic_equiflow):
        """Test flow table generation."""
        table = basic_equiflow.view_table_flows()
        assert isinstance(table, pd.DataFrame)
        assert "Initial" in str(table.index)
        assert "Removed" in str(table.index)
        assert "Result" in str(table.index)

    def test_flow_table_requires_two_cohorts(self, sample_data):
        """Test that flow table requires at least 2 cohorts."""
        ef = EquiFlow(data=sample_data, categorical=['sex'])
        with pytest.raises(ValueError, match="at least two cohorts"):
            ef.view_table_flows()

    def test_flow_table_thousands_sep(self, basic_equiflow):
        """Test thousands separator option."""
        table_with_sep = basic_equiflow.view_table_flows(thousands_sep=True)
        table_without_sep = basic_equiflow.view_table_flows(thousands_sep=False)
        # Both should work without error
        assert table_with_sep is not None
        assert table_without_sep is not None


class TestTableCharacteristics:
    """Tests for TableCharacteristics functionality."""

    def test_view_table_characteristics(self, basic_equiflow):
        """Test characteristics table generation."""
        table = basic_equiflow.view_table_characteristics()
        assert isinstance(table, pd.DataFrame)
        assert "Overall" in table.index.get_level_values(0)

    def test_characteristics_format_options(self, basic_equiflow):
        """Test different format options."""
        # Test percentage format
        table_pct = basic_equiflow.view_table_characteristics(format_cat='%')
        assert table_pct is not None
        
        # Test N format
        table_n = basic_equiflow.view_table_characteristics(format_cat='N')
        assert table_n is not None
        
        # Test N (%) format
        table_npct = basic_equiflow.view_table_characteristics(format_cat='N (%)')
        assert table_npct is not None

    def test_characteristics_with_limit(self, basic_equiflow):
        """Test category limit parameter."""
        table = basic_equiflow.view_table_characteristics(limit=2)
        assert table is not None

    def test_characteristics_with_order_classes(self, basic_equiflow):
        """Test custom category ordering."""
        table = basic_equiflow.view_table_characteristics(
            order_classes={'race': ['Asian', 'Black', 'White', 'Other']}
        )
        assert table is not None


class TestTableDrifts:
    """Tests for TableDrifts functionality."""

    def test_view_table_drifts(self, basic_equiflow):
        """Test drifts table generation."""
        table = basic_equiflow.view_table_drifts()
        assert isinstance(table, pd.DataFrame)

    def test_drifts_decimals(self, basic_equiflow):
        """Test decimal places parameter."""
        table = basic_equiflow.view_table_drifts(decimals=3)
        assert table is not None


class TestTablePValues:
    """Tests for TablePValues functionality."""

    def test_view_table_pvalues(self, basic_equiflow):
        """Test p-values table generation."""
        table = basic_equiflow.view_table_pvalues()
        assert isinstance(table, pd.DataFrame)

    def test_pvalues_correction_none(self, basic_equiflow):
        """Test p-values without correction."""
        table = basic_equiflow.view_table_pvalues(correction="none")
        assert table is not None

    def test_pvalues_correction_bonferroni(self, basic_equiflow):
        """Test p-values with Bonferroni correction."""
        table = basic_equiflow.view_table_pvalues(correction="bonferroni")
        assert table is not None

    def test_pvalues_correction_fdr(self, basic_equiflow):
        """Test p-values with FDR correction."""
        table = basic_equiflow.view_table_pvalues(correction="fdr_bh")
        assert table is not None

    def test_pvalues_invalid_correction_raises(self, basic_equiflow):
        """Test that invalid correction method raises error."""
        with pytest.raises(ValueError, match="Invalid correction"):
            basic_equiflow.view_table_pvalues(correction="invalid_method")


# ============================================================================
# EasyFlow Tests
# ============================================================================

class TestEasyFlow:
    """Tests for EasyFlow simplified interface."""

    def test_basic_easyflow(self, sample_data):
        """Test basic EasyFlow usage."""
        flow = EasyFlow(sample_data, title="Test Cohort")
        assert flow._data is not None
        assert flow._title == "Test Cohort"

    def test_easyflow_chaining(self, sample_data):
        """Test EasyFlow method chaining."""
        flow = (
            EasyFlow(sample_data, title="Test")
            .categorize(['sex', 'race'])
            .measure_normal(['age', 'bmi'])
            .measure_nonnormal(['income'])
        )
        assert flow._categorical_vars == ['sex', 'race']
        assert flow._normal_vars == ['age', 'bmi']
        assert flow._nonnormal_vars == ['income']

    def test_easyflow_exclude(self, sample_data):
        """Test EasyFlow exclusion."""
        flow = (
            EasyFlow(sample_data)
            .categorize(['sex'])
            .exclude(sample_data['age'] >= 18, "Age < 18")
        )
        assert len(flow._exclusion_steps) == 1
        assert len(flow._current_data) < len(sample_data)

    def test_easyflow_exclude_with_lambda(self, sample_data):
        """Test EasyFlow exclusion with lambda function."""
        flow = (
            EasyFlow(sample_data)
            .categorize(['sex'])
            .exclude(lambda df: df['age'] >= 18, "Age < 18")
        )
        assert len(flow._exclusion_steps) == 1

    def test_easyflow_generate_requires_exclusions(self, sample_data):
        """Test that generate() requires exclusion steps."""
        flow = EasyFlow(sample_data).categorize(['sex'])
        with pytest.raises(ValueError, match="No exclusion steps"):
            flow.generate(show=False)

    def test_easyflow_repr(self, sample_data):
        """Test EasyFlow string representation."""
        flow = EasyFlow(sample_data)
        repr_str = repr(flow)
        assert "EasyFlow" in repr_str


# ============================================================================
# Flow Diagram Tests
# ============================================================================

class TestFlowDiagram:
    """Tests for flow diagram generation."""

    def test_plot_flows_creates_files(self, basic_equiflow):
        """Test that plot_flows creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_equiflow.plot_flows(
                output_folder=tmpdir,
                output_file="test_diagram",
                display_flow_diagram=False,
                plot_dists=False
            )
            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, "test_diagram.pdf"))

    def test_plot_flows_with_distributions(self, basic_equiflow):
        """Test plot_flows with distribution plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_equiflow.plot_flows(
                output_folder=tmpdir,
                output_file="test_with_dists",
                display_flow_diagram=False,
                plot_dists=True,
                smds=True,
                legend=True
            )
            assert os.path.exists(os.path.join(tmpdir, "test_with_dists.pdf"))

    def test_plot_flows_custom_colors(self, basic_equiflow):
        """Test plot_flows with custom colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            basic_equiflow.plot_flows(
                output_folder=tmpdir,
                output_file="test_colors",
                display_flow_diagram=False,
                plot_dists=False,
                cohort_node_color='lightblue',
                exclusion_node_color='mistyrose',
                edge_color='navy'
            )
            assert os.path.exists(os.path.join(tmpdir, "test_colors.pdf"))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_category_variable(self, sample_data):
        """Test handling of variable with single category."""
        data = sample_data.copy()
        data['single_cat'] = 'A'  # Only one category
        
        ef = EquiFlow(data=data, categorical=['single_cat'])
        ef.add_exclusion(keep=data['age'] >= 18)
        
        # Should not raise an error
        table = ef.view_table_characteristics()
        assert table is not None

    def test_all_missing_variable(self, sample_data):
        """Test handling of variable with all missing values."""
        data = sample_data.copy()
        data['all_missing'] = None
        
        ef = EquiFlow(data=data, categorical=['sex'], normal=['all_missing'])
        ef.add_exclusion(keep=data['age'] >= 18)
        
        table = ef.view_table_characteristics()
        assert table is not None

    def test_very_small_cohort(self):
        """Test handling of very small cohorts."""
        small_data = pd.DataFrame({
            'age': [25, 30, 35],
            'sex': ['M', 'F', 'M']
        })
        
        ef = EquiFlow(data=small_data, categorical=['sex'])
        ef.add_exclusion(keep=small_data['age'] >= 28)
        
        table = ef.view_table_characteristics()
        assert table is not None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
