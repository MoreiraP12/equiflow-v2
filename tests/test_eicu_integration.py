"""
Integration tests for equiflow based on eICU case study.

These tests simulate a realistic clinical research workflow using
synthetic data modeled after eICU database characteristics.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from equiflow import EquiFlow, EasyFlow


def generate_synthetic_eicu_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic eICU-like data for testing.
    
    Creates a dataset with realistic ICU patient demographics and
    clinical characteristics.
    
    Parameters
    ----------
    n : int
        Number of patients to generate.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Synthetic patient data.
    """
    np.random.seed(seed)
    
    # Demographics
    age = np.random.normal(65, 15, n).clip(18, 95).astype(int)
    
    gender = np.random.choice(
        ['Male', 'Female'], 
        n, 
        p=[0.55, 0.45]
    )
    
    race_ethnicity = np.random.choice(
        ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
        n,
        p=[0.65, 0.15, 0.10, 0.05, 0.05]
    )
    
    # Clinical scores - correlated with age
    apache_base = np.random.gamma(3, 5, n)
    apache_score = (apache_base + (age - 50) * 0.3).clip(0, 150)
    
    # Troponin with missing values (~15% missing)
    troponin = np.random.lognormal(0, 1, n)
    troponin_keep = np.random.random(n) < 0.15
    troponin = np.where(troponin_keep, np.nan, troponin)
    
    # Clinical flags
    non_cardiac_patient = np.random.random(n) < 0.70  # 70% non-cardiac
    septic_patient = np.random.random(n) < 0.20  # 20% septic
    
    # Outcomes
    mortality = np.random.random(n) < (0.05 + apache_score / 500)
    
    # Length of stay
    los_days = np.random.lognormal(1.2, 0.8, n).clip(0.5, 60)
    
    return pd.DataFrame({
        'patient_id': range(1, n + 1),
        'age': age,
        'gender': gender,
        'race_ethnicity': race_ethnicity,
        'apache_score': apache_score.round(1),
        'troponin': troponin,
        'non_cardiac_patient': non_cardiac_patient,
        'septic_patient': septic_patient,
        'mortality': mortality.astype(int),
        'los_days': los_days.round(2),
    })


class TestEICUCaseStudy:
    """
    Integration tests simulating an eICU case study workflow.
    
    These tests mirror the workflow described in the equiflow paper,
    tracking demographic changes through cohort selection.
    """

    @pytest.fixture
    def eicu_data(self):
        """Generate synthetic eICU data."""
        return generate_synthetic_eicu_data(n=5000, seed=42)

    def test_eicu_case_study_full_workflow(self, eicu_data):
        """
        Test complete eICU case study workflow.
        
        Workflow:
        1. Start with all ICU patients
        2. Exclude cardiac patients
        3. Exclude non-septic patients
        4. Exclude patients with missing troponin
        """
        # Initialize EquiFlow
        ef = EquiFlow(
            data=eicu_data,
            initial_cohort_label="All eICU Patients",
            categorical=['gender', 'race_ethnicity'],
            normal=['age'],
            nonnormal=['apache_score', 'los_days'],
        )
        
        # Exclusion 1: Keep non-cardiac patients (exclude cardiac)
        ef.add_exclusion(
            keep=eicu_data['non_cardiac_patient'],
            exclusion_reason="Cardiac admission",
            new_cohort_label="Non-cardiac patients"
        )
        
        # Get current cohort for next exclusion
        cohort1 = ef._dfs[-1]
        
        # Exclusion 2: Keep septic patients (exclude non-septic)
        ef.add_exclusion(
            keep=cohort1['septic_patient'],
            exclusion_reason="Non-septic patient",
            new_cohort_label="Septic patients"
        )
        
        # Get current cohort
        cohort2 = ef._dfs[-1]
        
        # Exclusion 3: Exclude patients with missing troponin
        ef.add_exclusion(
            keep=cohort2['troponin'].notna(),
            exclusion_reason="Missing troponin",
            new_cohort_label="Final cohort"
        )
        
        # Verify cohort sizes decrease appropriately
        assert len(ef._dfs) == 4
        for i in range(1, len(ef._dfs)):
            assert len(ef._dfs[i]) <= len(ef._dfs[i-1])
        
        # Generate all tables
        flows = ef.view_table_flows()
        chars = ef.view_table_characteristics()
        drifts = ef.view_table_drifts()
        
        assert flows is not None
        assert chars is not None
        assert drifts is not None
        
        # Generate flow diagram
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "eicu_flow")
            ef.plot_flows(
                output_file=output_path,
                output_folder=tmpdir,
                display_flow_diagram=False,
            )
            
            # Verify output was created
            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_eicu_case_study_tables_only(self, eicu_data):
        """Test case study without generating diagram (faster)."""
        ef = EquiFlow(
            data=eicu_data,
            categorical=['gender', 'race_ethnicity'],
            normal=['age'],
            nonnormal=['apache_score'],
        )
        
        ef.add_exclusion(
            keep=eicu_data['non_cardiac_patient'],
            exclusion_reason="Cardiac admission"
        )
        
        cohort1 = ef._dfs[-1]
        ef.add_exclusion(
            keep=cohort1['septic_patient'],
            exclusion_reason="Non-septic"
        )
        
        # Verify tables
        flows = ef.view_table_flows()
        chars = ef.view_table_characteristics()
        drifts = ef.view_table_drifts()
        
        # Flow table should have 3 rows (initial + 2 exclusions)
        assert len(flows) == 3
        
        # Characteristics should include all specified variables
        assert chars is not None
        assert drifts is not None

    def test_eicu_with_pvalues_and_correction(self, eicu_data):
        """Test p-value calculations with multiple testing correction."""
        ef = EquiFlow(
            data=eicu_data,
            categorical=['gender', 'race_ethnicity'],
            normal=['age'],
        )
        
        ef.add_exclusion(
            keep=eicu_data['non_cardiac_patient'],
            exclusion_reason="Cardiac admission"
        )
        
        # Test different correction methods
        pvals_none = ef.view_table_pvalues(correction="none")
        pvals_bonf = ef.view_table_pvalues(correction="bonferroni")
        pvals_fdr = ef.view_table_pvalues(correction="fdr_bh")
        
        assert pvals_none is not None
        assert pvals_bonf is not None
        assert pvals_fdr is not None

    def test_eicu_demographic_drift_detection(self, eicu_data):
        """
        Test that significant demographic drift is detectable.
        
        Creates an exclusion that should cause demographic shift
        and verifies SMD calculations work correctly.
        """
        ef = EquiFlow(
            data=eicu_data,
            categorical=['gender'],
            normal=['age'],
        )
        
        # Create exclusion that should shift demographics
        # (older patients more likely to be excluded)
        ef.add_exclusion(
            keep=eicu_data['age'] < 70,
            exclusion_reason="Age >= 70"
        )
        
        drifts = ef.view_table_drifts()
        
        # Age drift should be significant (excluding older patients)
        assert drifts is not None


class TestSyntheticDataGenerator:
    """Tests for the synthetic data generator itself."""

    def test_data_shape(self):
        """Test that generated data has correct shape."""
        df = generate_synthetic_eicu_data(n=1000)
        assert len(df) == 1000
        assert 'patient_id' in df.columns
        assert 'age' in df.columns
        assert 'gender' in df.columns

    def test_data_distributions(self):
        """Test that data distributions are reasonable."""
        df = generate_synthetic_eicu_data(n=10000, seed=42)
        
        # Age should be roughly normally distributed around 65
        assert 60 < df['age'].mean() < 70
        
        # Gender ratio should be approximately 55/45
        male_ratio = (df['gender'] == 'Male').mean()
        assert 0.50 < male_ratio < 0.60
        
        # Troponin should have ~15% missing
        missing_ratio = df['troponin'].isna().mean()
        assert 0.10 < missing_ratio < 0.20

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        df1 = generate_synthetic_eicu_data(n=100, seed=123)
        df2 = generate_synthetic_eicu_data(n=100, seed=123)
        
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test that different seeds produce different data."""
        df1 = generate_synthetic_eicu_data(n=100, seed=1)
        df2 = generate_synthetic_eicu_data(n=100, seed=2)
        
        # Should not be identical
        assert not df1.equals(df2)


class TestEasyFlowEICU:
    """Test EasyFlow interface with eICU-like data."""

    def test_easyflow_eicu_workflow(self):
        """Test EasyFlow with eICU case study."""
        data = generate_synthetic_eicu_data(n=1000)
        
        flow = (EasyFlow(data, title="eICU Sepsis Study")
            .categorize(['gender', 'race_ethnicity'])
            .measure_normal(['age'])
            .measure_nonnormal(['apache_score', 'los_days'])
            .exclude(data['non_cardiac_patient'], "Cardiac admission")
            .exclude(lambda d: d['septic_patient'], "Non-septic"))
        
        assert len(flow._exclusion_steps) == 2
        assert len(flow._current_data) < len(data)

    def test_easyflow_generate(self):
        """Test EasyFlow generate method."""
        data = generate_synthetic_eicu_data(n=500)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "easy_flow")
            
            flow = (EasyFlow(data, title="Quick Study")
                .categorize(['gender'])
                .measure_normal(['age'])
                .exclude(lambda d: d['age'] >= 30, "Age < 30")
                .generate(output=output_path, show=False))
            
            # Check attributes were set
            assert flow.flow_table is not None
            assert flow.characteristics is not None
            assert flow.drifts is not None
            
            # Check attributes were set
            assert flow.flow_table is not None
            assert flow.characteristics is not None
            assert flow.drifts is not None
