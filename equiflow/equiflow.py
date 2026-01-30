"""
EquiFlow: Equity-focused Cohort Selection Flow Diagrams.

A Python package for creating visual flow diagrams that track demographic
composition changes through sequential cohort exclusion steps in clinical
and epidemiological research.

Features:
- Track cohort sizes through exclusion steps
- Calculate standardized mean differences (SMDs) to detect demographic drift
- Generate publication-ready flow diagrams with embedded distribution plots
- Support for categorical, normally-distributed, and non-normal continuous variables
- Multiple testing correction for p-value calculations
"""

from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import graphviz
import warnings

__version__ = "0.1.8"

__all__ = [
    "EquiFlow",
    "EasyFlow",
    "TableFlows",
    "TableCharacteristics",
    "TableDrifts",
    "TablePValues",
    "FlowDiagram",
]


# Utility Functions

def _safe_scalar(value: Any) -> Any:
    """
    Safely extract a scalar value from a pandas lookup result.
    
    When using .loc[] on a MultiIndex DataFrame, pandas may return a Series
    instead of a scalar if the index isn't properly lexsorted. This function
    ensures we always get a scalar value.
    
    Parameters
    ----------
    value : Any
        The value returned from a .loc[] lookup.
        
    Returns
    -------
    Any
        A scalar value.
    """
    if isinstance(value, pd.Series):
        return value.iloc[0] if len(value) > 0 else value
    return value


def format_smd(smd_value: Any, decimals: int = 2) -> str:
    """
    Format SMD value with specified decimal places.
    
    Shows whole numbers without decimal places (e.g., '0' instead of '0.00').
    
    Parameters
    ----------
    smd_value : Any
        The SMD value to format.
    decimals : int, default=2
        Number of decimal places for non-whole numbers.
        
    Returns
    -------
    str
        Formatted SMD string.
    """
    if pd.isna(smd_value) or smd_value == '':
        return ''
    try:
        float_val = float(smd_value)
        if float_val == int(float_val):
            return str(int(float_val))
        else:
            return f"{float_val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(smd_value)


def validate_variables(
    df: pd.DataFrame,
    categorical: List[str],
    normal: List[str],
    nonnormal: List[str],
    context: str = "EquiFlow"
) -> None:
    """
    Validate that specified variables exist in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    categorical : list
        List of categorical variable names.
    normal : list
        List of normally-distributed variable names.
    nonnormal : list
        List of non-normally distributed variable names.
    context : str, default="EquiFlow"
        Context string for error messages.
        
    Raises
    ------
    ValueError
        If any specified variable is not found in the DataFrame.
    """
    all_vars = set(categorical + normal + nonnormal)
    df_cols = set(df.columns)
    missing = all_vars - df_cols
    
    if missing:
        raise ValueError(
            f"{context}: The following variables were not found in the DataFrame: "
            f"{sorted(missing)}. Available columns: {sorted(df_cols)}"
        )


def validate_dataframe_not_empty(df: pd.DataFrame, context: str = "EquiFlow") -> None:
    """
    Validate that a DataFrame is not empty.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    context : str, default="EquiFlow"
        Context string for error messages.
        
    Raises
    ------
    ValueError
        If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError(f"{context}: DataFrame cannot be empty.")


def check_normality(series: pd.Series, alpha: float = 0.05) -> bool:
    """
    Check if a numeric series is approximately normally distributed.
    
    Uses Shapiro-Wilk test for small samples (n <= 2000) and
    D'Agostino-Pearson test for larger samples.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to test.
    alpha : float, default=0.05
        Significance level for the normality test.
        
    Returns
    -------
    bool
        True if the series appears normally distributed, False otherwise.
    """
    from scipy import stats
    
    clean = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(clean) < 8:
        # Too few samples to reliably test normality
        return False
    
    try:
        if len(clean) <= 2000:
            # Shapiro-Wilk test (more powerful for smaller samples)
            _, p_value = stats.shapiro(clean)
        else:
            # D'Agostino-Pearson test for larger samples
            _, p_value = stats.normaltest(clean)
        
        return p_value >= alpha
    except Exception:
        return False


# Main EquiFlow Class

class EquiFlow:
    """
    Main class for creating equity-focused cohort selection flow diagrams.
    
    EquiFlow tracks how demographic composition changes through sequential
    exclusion steps in cohort selection, helping researchers identify and
    quantify selection bias.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Initial cohort data. Either `data` or `dfs` must be provided.
    dfs : list of pd.DataFrame, optional
        List of DataFrames representing sequential cohorts.
    initial_cohort_label : str, optional
        Label for the initial cohort. Default is "Initial Cohort".
    label_suffix : bool, default=True
        Whether to add format suffixes to variable labels (e.g., ", N (%)").
    thousands_sep : bool, default=True
        Whether to use thousands separators in counts.
    categorical : list, optional
        List of categorical variable names.
    normal : list, optional
        List of normally-distributed continuous variable names.
    nonnormal : list, optional
        List of non-normally distributed continuous variable names.
    order_vars : list, optional
        List specifying the display order of variables.
    order_classes : dict, optional
        Dictionary mapping categorical variable names to lists specifying
        the display order of their categories.
    limit : int or dict, optional
        Maximum number of categories to display. If int, applies to all
        categorical variables. If dict, maps variable names to limits.
    decimals : int, default=1
        Number of decimal places for statistics.
    smd_decimals : int, optional
        Number of decimal places for SMD values. Defaults to `decimals`.
    format_cat : str, default="N (%)"
        Format for categorical variables. Options: "N (%)", "N", "%".
    format_normal : str, default="Mean ± SD"
        Format for normal variables. Options: "Mean ± SD", "Mean", "SD".
    format_nonnormal : str, default="Median [IQR]"
        Format for non-normal variables. Options: "Median [IQR]", "Mean", "SD".
    missingness : bool, default=True
        Whether to display missingness statistics.
    rename : dict, optional
        Dictionary mapping variable names to display names.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from equiflow import EquiFlow
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40, 45, 50, 55, 60],
    ...     'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    ...     'score': [80, 85, 90, 95, 70, 75, 80, 85]
    ... })
    >>> 
    >>> # Initialize EquiFlow
    >>> ef = EquiFlow(
    ...     data=df,
    ...     categorical=['sex'],
    ...     normal=['age'],
    ...     nonnormal=['score']
    ... )
    >>> 
    >>> # Add exclusion step (exclude patients under 35)
    >>> ef.add_exclusion(
    ...     keep=df['age'] >= 35,
    ...     exclusion_reason="Age < 35",
    ...     new_cohort_label="Adults 35+"
    ... )
    >>> 
    >>> # View results
    >>> print(ef.view_table_flows())
    """
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        dfs: Optional[List[pd.DataFrame]] = None,
        initial_cohort_label: Optional[str] = None,
        label_suffix: bool = True,
        thousands_sep: bool = True,
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        decimals: int = 1,
        smd_decimals: Optional[int] = None,
        format_cat: str = "N (%)",
        format_normal: str = "Mean ± SD",
        format_nonnormal: str = "Median [IQR]",
        missingness: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize EquiFlow object for creating cohort flow diagrams."""
        
        # Validate parameters
        if order_classes is not None and not isinstance(order_classes, dict):
            raise ValueError("order_classes must be a dictionary mapping variable names to category lists")
        if limit is not None and not (isinstance(limit, int) or isinstance(limit, dict)):
            raise ValueError("limit must be an integer or a dictionary mapping variable names to integers")
        if order_vars is not None and not isinstance(order_vars, list):
            raise ValueError("order_vars must be a list of variable names")
        if data is None and dfs is None:
            raise ValueError("Either 'data' or 'dfs' must be provided")

        # Initialize DataFrames
        self._dfs: List[pd.DataFrame] = []
        if data is not None:
            validate_dataframe_not_empty(data, "EquiFlow.__init__")
            self._dfs = [data.copy()]
        if dfs is not None:
            if not dfs:
                raise ValueError("dfs list cannot be empty")
            for i, df in enumerate(dfs):
                validate_dataframe_not_empty(df, f"EquiFlow.__init__ (dfs[{i}])")
            self._dfs = [df.copy() for df in dfs]
        
        # Clean missing values
        self._clean_missing()

        # Store parameters
        self.label_suffix = label_suffix
        self.thousands_sep = thousands_sep
        self.categorical = categorical if categorical is not None else []
        self.normal = normal if normal is not None else []
        self.nonnormal = nonnormal if nonnormal is not None else []
        self.order_vars = order_vars
        self.order_classes = order_classes if order_classes is not None else {}
        self.limit = limit
        self.decimals = decimals
        self.smd_decimals = smd_decimals if smd_decimals is not None else decimals
        self.format_cat = format_cat
        self.format_normal = format_normal
        self.format_nonnormal = format_nonnormal
        self.missingness = missingness
        self.rename = rename if rename is not None else {}
        
        # Validate variables exist in the initial DataFrame
        if self._dfs:
            validate_variables(
                self._dfs[0],
                self.categorical,
                self.normal,
                self.nonnormal,
                "EquiFlow.__init__"
            )
        
        # Initialize output containers
        self.table_flows: Optional[TableFlows] = None
        self.table_characteristics: Optional[TableCharacteristics] = None
        self.table_drifts: Optional[TableDrifts] = None
        self.table_pvalues: Optional[TablePValues] = None
        self.flow_diagram: Optional[FlowDiagram] = None
        
        # Initialize labels
        self.exclusion_labels: Dict[int, str] = {}
        self.new_cohort_labels: Dict[int, str] = {}
        if initial_cohort_label is not None:
            self.new_cohort_labels[0] = initial_cohort_label
        else:
            self.new_cohort_labels[0] = "Initial Cohort"

    def __repr__(self) -> str:
        """Return string representation of EquiFlow object."""
        n_cohorts = len(self._dfs)
        n_vars = len(self.categorical) + len(self.normal) + len(self.nonnormal)
        initial_n = len(self._dfs[0]) if self._dfs else 0
        final_n = len(self._dfs[-1]) if self._dfs else 0
        return (
            f"EquiFlow(cohorts={n_cohorts}, variables={n_vars}, "
            f"initial_n={initial_n:,}, final_n={final_n:,})"
        )

    def _clean_missing(self) -> None:
        """
        Standardize missing value representations across all DataFrames.
        
        Converts various missing value representations to pandas NA values
        for consistent handling throughout the package.
        """
        missing_values = [
            "", " ", "NA", "N/A", "na", "n/a", "NA ", "N/A ", "na ", "n/a ",
            "NaN", "nan", "NAN", "Nan", "N/A;", "<NA>", "_", "__", "___",
            "____", "_____", "NaT", "None", "none", "NONE", "Null", "null",
            "NULL", "missing", "Missing",
        ]
        
        for i, df in enumerate(self._dfs):
            # Replace string missing values with pd.NA
            df_cleaned = df.replace(missing_values, pd.NA)
            
            # Also handle numpy NaN explicitly
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    # For object columns, replace np.nan with pd.NA
                    df_cleaned[col] = df_cleaned[col].replace({np.nan: pd.NA})
            
            self._dfs[i] = df_cleaned

    def add_exclusion(
        self,
        keep: Optional[pd.Series] = None,
        new_cohort: Optional[pd.DataFrame] = None,
        exclusion_reason: Optional[str] = None,
        new_cohort_label: Optional[str] = None,
        mask: Optional[pd.Series] = None,  # Deprecated parameter
    ) -> 'EquiFlow':
        """
        Add an exclusion step to the cohort flow.
        
        Parameters
        ----------
        keep : pd.Series of bool, optional
            Boolean mask where True indicates rows to KEEP (retain).
            Rows where keep=False will be excluded.
            Either `keep` or `new_cohort` must be provided, but not both.
        new_cohort : pd.DataFrame, optional
            DataFrame representing the new cohort after exclusion.
            Either `keep` or `new_cohort` must be provided, but not both.
        exclusion_reason : str, optional
            Description of why rows were excluded.
        new_cohort_label : str, optional
            Label for the resulting cohort.
        mask : pd.Series of bool, optional
            .. deprecated:: 0.1.7
                Use `keep` instead. The `mask` parameter will be removed in v0.2.0.
            
        Returns
        -------
        EquiFlow
            Self, for method chaining.
            
        Examples
        --------
        >>> # Keep patients 18 and older (exclude those under 18)
        >>> ef.add_exclusion(
        ...     keep=df['age'] >= 18,
        ...     exclusion_reason="Age < 18",
        ...     new_cohort_label="Adult cohort"
        ... )
        """
        # Handle deprecated 'mask' parameter
        if mask is not None:
            warnings.warn(
                "The 'mask' parameter is deprecated and will be removed in v0.2.0. "
                "Use 'keep' instead. Both parameters have the same behavior: "
                "rows where the value is True are kept (retained).",
                DeprecationWarning,
                stacklevel=2
            )
            if keep is not None:
                raise ValueError(
                    "Cannot specify both 'keep' and 'mask'. "
                    "The 'mask' parameter is deprecated; use 'keep' instead."
                )
            keep = mask
        
        if keep is None and new_cohort is None:
            raise ValueError("Either 'keep' or 'new_cohort' must be provided")
        if keep is not None and new_cohort is not None:
            raise ValueError("Only one of 'keep' or 'new_cohort' can be provided, not both")
        
        if keep is not None:
            # KEEP rows where keep is True
            new_df = self._dfs[-1].loc[keep].copy()
        else:
            new_df = new_cohort.copy()
        
        # Warn if the resulting cohort is empty
        if new_df.empty:
            warnings.warn(
                f"Exclusion step '{exclusion_reason or 'unnamed'}' resulted in an empty cohort. "
                "This may cause issues with downstream calculations.",
                UserWarning
            )
        
        self._dfs.append(new_df)
        
        # Set labels
        step_idx = len(self._dfs) - 1
        if exclusion_reason is not None:
            self.exclusion_labels[step_idx] = exclusion_reason
        else:
            self.exclusion_labels[step_idx] = f"Exclusion {step_idx}"
        
        if new_cohort_label is not None:
            self.new_cohort_labels[step_idx] = new_cohort_label
        else:
            self.new_cohort_labels[step_idx] = f"Cohort {step_idx}"
        
        return self

    def view_table_flows(
        self,
        label_suffix: Optional[bool] = None,
        thousands_sep: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Generate and return the cohort flow table.
        
        Parameters
        ----------
        label_suffix : bool, optional
            Whether to add ", n" suffix to labels.
        thousands_sep : bool, optional
            Whether to use thousands separators.
            
        Returns
        -------
        pd.DataFrame
            Table showing cohort sizes at each step.
            
        Raises
        ------
        ValueError
            If fewer than two cohorts have been defined.
        """
        if len(self._dfs) < 2:
            raise ValueError(
                "At least two cohorts must be provided. "
                "Use add_exclusion() to add exclusion steps."
            )
        
        label_suffix = label_suffix if label_suffix is not None else self.label_suffix
        thousands_sep = thousands_sep if thousands_sep is not None else self.thousands_sep
        
        self.table_flows = TableFlows(
            dfs=self._dfs,
            label_suffix=label_suffix,
            thousands_sep=thousands_sep
        )
        return self.table_flows.view()

    def view_table_characteristics(
        self,
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        decimals: Optional[int] = None,
        format_cat: Optional[str] = None,
        format_normal: Optional[str] = None,
        format_nonnormal: Optional[str] = None,
        thousands_sep: Optional[bool] = None,
        missingness: Optional[bool] = None,
        label_suffix: Optional[bool] = None,
        rename: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Generate a table of cohort characteristics.
        
        Parameters
        ----------
        categorical : list, optional
            List of categorical variable names.
        normal : list, optional
            List of normally-distributed variable names.
        nonnormal : list, optional
            List of non-normally distributed variable names.
        order_vars : list, optional
            List specifying the display order of variables.
        order_classes : dict, optional
            Dictionary mapping variable names to category order lists.
        limit : int or dict, optional
            Maximum categories to display per variable.
        decimals : int, optional
            Number of decimal places.
        format_cat : str, optional
            Format for categorical variables.
        format_normal : str, optional
            Format for normal variables.
        format_nonnormal : str, optional
            Format for non-normal variables.
        thousands_sep : bool, optional
            Whether to use thousands separators.
        missingness : bool, optional
            Whether to show missingness statistics.
        label_suffix : bool, optional
            Whether to add format suffixes to labels.
        rename : dict, optional
            Variable name mappings.
            
        Returns
        -------
        pd.DataFrame
            Table of characteristics for each cohort.
        """
        if len(self._dfs) < 2:
            raise ValueError("At least two cohorts must be provided.")
        
        # Use instance defaults if not specified
        categorical = categorical if categorical is not None else self.categorical
        normal = normal if normal is not None else self.normal
        nonnormal = nonnormal if nonnormal is not None else self.nonnormal
        order_vars = order_vars if order_vars is not None else self.order_vars
        order_classes = order_classes if order_classes is not None else self.order_classes
        limit = limit if limit is not None else self.limit
        decimals = decimals if decimals is not None else self.decimals
        format_cat = format_cat if format_cat is not None else self.format_cat
        format_normal = format_normal if format_normal is not None else self.format_normal
        format_nonnormal = format_nonnormal if format_nonnormal is not None else self.format_nonnormal
        thousands_sep = thousands_sep if thousands_sep is not None else self.thousands_sep
        missingness = missingness if missingness is not None else self.missingness
        label_suffix = label_suffix if label_suffix is not None else self.label_suffix
        rename = rename if rename is not None else self.rename
        
        self.table_characteristics = TableCharacteristics(
            dfs=self._dfs,
            categorical=categorical,
            normal=normal,
            nonnormal=nonnormal,
            order_vars=order_vars,
            order_classes=order_classes,
            limit=limit,
            decimals=decimals,
            format_cat=format_cat,
            format_normal=format_normal,
            format_nonnormal=format_nonnormal,
            thousands_sep=thousands_sep,
            missingness=missingness,
            label_suffix=label_suffix,
            rename=rename,
        )
        return self.table_characteristics.view()

    def view_table_drifts(
        self,
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        decimals: Optional[int] = None,
        missingness: Optional[bool] = None,
        rename: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Generate a table of standardized mean differences (SMDs/drifts).
        
        SMDs quantify how much the distribution of each variable changes
        between consecutive cohorts, helping identify demographic drift
        during cohort selection.
        
        Parameters
        ----------
        categorical : list, optional
            List of categorical variable names.
        normal : list, optional
            List of normally-distributed variable names.
        nonnormal : list, optional
            List of non-normally distributed variable names.
        order_vars : list, optional
            List specifying the display order of variables.
        order_classes : dict, optional
            Dictionary mapping variable names to category order lists.
        limit : int or dict, optional
            Maximum categories to display per variable.
        decimals : int, optional
            Number of decimal places for SMD values.
        missingness : bool, optional
            Whether to include missingness in calculations.
        rename : dict, optional
            Variable name mappings.
            
        Returns
        -------
        pd.DataFrame
            Table of SMD values between consecutive cohorts.
        """
        if len(self._dfs) < 2:
            raise ValueError("At least two cohorts must be provided.")
        
        categorical = categorical if categorical is not None else self.categorical
        normal = normal if normal is not None else self.normal
        nonnormal = nonnormal if nonnormal is not None else self.nonnormal
        order_vars = order_vars if order_vars is not None else self.order_vars
        order_classes = order_classes if order_classes is not None else self.order_classes
        limit = limit if limit is not None else self.limit
        decimals = decimals if decimals is not None else self.smd_decimals
        missingness = missingness if missingness is not None else self.missingness
        rename = rename if rename is not None else self.rename
        
        self.table_drifts = TableDrifts(
            dfs=self._dfs,
            categorical=categorical,
            normal=normal,
            nonnormal=nonnormal,
            order_vars=order_vars,
            order_classes=order_classes,
            limit=limit,
            decimals=decimals,
            missingness=missingness,
            rename=rename,
        )
        return self.table_drifts.view()

    def view_table_pvalues(
        self,
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        alpha: float = 0.05,
        decimals: int = 3,
        min_n_expected: int = 5,
        min_samples: int = 30,
        rename: Optional[Dict[str, str]] = None,
        correction: str = "none",
    ) -> pd.DataFrame:
        """
        Generate a table of p-values between consecutive cohorts.
        
        Parameters
        ----------
        categorical : list, optional
            List of categorical variable names.
        normal : list, optional
            List of normally-distributed variable names.
        nonnormal : list, optional
            List of non-normally distributed variable names.
        order_vars : list, optional
            List specifying the display order of variables.
        order_classes : dict, optional
            Dictionary mapping variable names to category order lists.
        limit : int or dict, optional
            Maximum categories to display per variable.
        alpha : float, default=0.05
            Significance level for hypothesis testing.
        decimals : int, default=3
            Number of decimal places for p-values.
        min_n_expected : int, default=5
            Minimum expected count for chi-square test validity.
        min_samples : int, default=30
            Minimum sample size for parametric tests.
        rename : dict, optional
            Variable name mappings.
        correction : str, default="none"
            Multiple testing correction method applied globally across all
            non-missingness tests. Options:
            - "none": No correction (default)
            - "bonferroni": Bonferroni correction (controls FWER)
            - "fdr_bh" or "bh": Benjamini-Hochberg procedure (controls FDR)
            
        Returns
        -------
        pd.DataFrame
            Table of p-values with significance indicators.
        """
        if len(self._dfs) < 2:
            raise ValueError("At least two cohorts must be provided.")
        
        categorical = categorical if categorical is not None else self.categorical
        normal = normal if normal is not None else self.normal
        nonnormal = nonnormal if nonnormal is not None else self.nonnormal
        order_vars = order_vars if order_vars is not None else self.order_vars
        order_classes = order_classes if order_classes is not None else self.order_classes
        limit = limit if limit is not None else self.limit
        rename = rename if rename is not None else self.rename
        
        self.table_pvalues = TablePValues(
            dfs=self._dfs,
            categorical=categorical,
            normal=normal,
            nonnormal=nonnormal,
            order_vars=order_vars,
            order_classes=order_classes,
            limit=limit,
            alpha=alpha,
            decimals=decimals,
            min_n_expected=min_n_expected,
            min_samples=min_samples,
            rename=rename,
            correction=correction,
        )
        return self.table_pvalues.view()

    def plot_flows(
        self,
        new_cohort_labels: Optional[List[str]] = None,
        exclusion_labels: Optional[List[str]] = None,
        box_width: float = 2.5,
        box_height: float = 1.0,
        plot_dists: bool = True,
        smds: bool = True,
        smd_decimals: Optional[int] = None,
        pvalues: bool = False,
        pvalue_decimals: int = 3,
        legend: bool = True,
        legend_with_vars: bool = True,
        output_folder: str = "imgs",
        output_file: str = "flow_diagram",
        display_flow_diagram: bool = True,
        cohort_node_color: str = "white",
        exclusion_node_color: str = "floralwhite",
        categorical_bar_colors: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        missing_value_color: str = "lightgray",
        continuous_var_color: str = "lavender",
        edge_color: str = "black",
        pvalue_correction: str = "none",
    ) -> None:
        """
        Create and display a flow diagram of the cohort exclusion process.
        
        Parameters
        ----------
        new_cohort_labels : list of str, optional
            Labels for each cohort node. Use "___" as placeholder for counts.
        exclusion_labels : list of str, optional
            Labels for each exclusion node. Use "___" as placeholder for counts.
        box_width : float, default=2.5
            Width of cohort boxes in the diagram.
        box_height : float, default=1.0
            Height of cohort boxes in the diagram.
        plot_dists : bool, default=True
            Whether to include distribution plots.
        smds : bool, default=True
            Whether to display SMD values in the distribution plots.
        smd_decimals : int, optional
            Number of decimal places for SMD values in plots.
        pvalues : bool, default=False
            Whether to display p-values in the flow diagram.
        pvalue_decimals : int, default=3
            Number of decimal places for p-values in the plot.
        legend : bool, default=True
            Whether to include a legend.
        legend_with_vars : bool, default=True
            Whether to include variable names in legend entries.
        output_folder : str, default="imgs"
            Folder for output files.
        output_file : str, default="flow_diagram"
            Base name for output files.
        display_flow_diagram : bool, default=True
            Whether to display the diagram after creation.
        cohort_node_color : str, default="white"
            Fill color for cohort nodes.
        exclusion_node_color : str, default="floralwhite"
            Fill color for exclusion nodes.
        categorical_bar_colors : list or dict, optional
            Colors for categorical variable bars.
        missing_value_color : str, default="lightgray"
            Color for missing value bars.
        continuous_var_color : str, default="lavender"
            Color for continuous variable bars.
        edge_color : str, default="black"
            Color for diagram edges.
        pvalue_correction : str, default="none"
            Multiple testing correction method for p-values.
        """
        if len(self._dfs) < 2:
            raise ValueError("At least two cohorts must be provided.")
        
        if new_cohort_labels is None:
            new_cohort_labels = [
                "___ patients\n" + label for label in self.new_cohort_labels.values()
            ]
        if exclusion_labels is None:
            exclusion_labels = [
                "___ patients excluded for\n" + label
                for label in self.exclusion_labels.values()
            ]
        
        smd_decimals = smd_decimals if smd_decimals is not None else self.smd_decimals
        
        # Create internal tables for plotting
        plot_table_flows = TableFlows(
            dfs=self._dfs,
            label_suffix=True,
            thousands_sep=True
        )
        plot_table_characteristics = TableCharacteristics(
            dfs=self._dfs,
            categorical=self.categorical,
            normal=self.normal,
            nonnormal=self.nonnormal,
            order_classes=self.order_classes,
            limit=self.limit,
            decimals=self.decimals,
            format_cat="%",
            format_normal=self.format_normal,
            format_nonnormal=self.format_nonnormal,
            thousands_sep=False,
            missingness=True,
            label_suffix=True,
            rename=self.rename,
        )
        plot_table_drifts = TableDrifts(
            dfs=self._dfs,
            categorical=self.categorical,
            normal=self.normal,
            nonnormal=self.nonnormal,
            order_classes=self.order_classes,
            limit=self.limit,
            decimals=smd_decimals,
            missingness=self.missingness,
            rename=self.rename,
        )
        
        plot_table_pvalues = None
        if pvalues:
            plot_table_pvalues = TablePValues(
                dfs=self._dfs,
                categorical=self.categorical,
                normal=self.normal,
                nonnormal=self.nonnormal,
                order_classes=self.order_classes,
                limit=self.limit,
                decimals=pvalue_decimals,
                rename=self.rename,
                correction=pvalue_correction,
            )
        
        self.flow_diagram = FlowDiagram(
            table_flows=plot_table_flows,
            table_characteristics=plot_table_characteristics,
            table_drifts=plot_table_drifts,
            table_pvalues=plot_table_pvalues,
            new_cohort_labels=new_cohort_labels,
            exclusion_labels=exclusion_labels,
            box_width=box_width,
            box_height=box_height,
            plot_dists=plot_dists,
            smds=smds,
            smd_decimals=smd_decimals,
            pvalues=pvalues,
            pvalue_decimals=pvalue_decimals,
            legend=legend,
            legend_with_vars=legend_with_vars,
            output_folder=output_folder,
            output_file=output_file,
            display_flow_diagram=display_flow_diagram,
            cohort_node_color=cohort_node_color,
            exclusion_node_color=exclusion_node_color,
            categorical_bar_colors=categorical_bar_colors,
            missing_value_color=missing_value_color,
            continuous_var_color=continuous_var_color,
            edge_color=edge_color,
        )
        self.flow_diagram.view()


# TableFlows Class

class TableFlows:
    """
    Generate tables showing cohort flow (sizes at each step).
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames representing sequential cohorts.
    label_suffix : bool, default=True
        Whether to add ", n" suffix to labels.
    thousands_sep : bool, default=True
        Whether to use thousands separators in counts.
    """
    
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        label_suffix: bool = True,
        thousands_sep: bool = True
    ):
        if not dfs:
            raise ValueError("dfs list cannot be empty")
        self._dfs = dfs
        self._label_suffix = label_suffix
        self._thousands_sep = thousands_sep

    def __repr__(self) -> str:
        return f"TableFlows(cohorts={len(self._dfs)})"

    def view(self) -> pd.DataFrame:
        """
        Generate the cohort flow table.
        
        Returns
        -------
        pd.DataFrame
            Pivot table showing Initial, Removed, and Result counts
            for each cohort transition.
        """
        rows = []
        suffix = ", n" if self._label_suffix else ""
        
        for i in range(len(self._dfs) - 1):
            df_0, df_1 = self._dfs[i], self._dfs[i + 1]
            label = f"{i} to {i + 1}"
            
            n_initial = len(df_0)
            n_removed = len(df_0) - len(df_1)
            n_result = len(df_1)
            
            if self._thousands_sep:
                n0, n1, n2 = f"{n_initial:,}", f"{n_removed:,}", f"{n_result:,}"
            else:
                n0, n1, n2 = n_initial, n_removed, n_result
            
            rows.append({"Cohort Flow": label, "": "Initial" + suffix, "N": n0})
            rows.append({"Cohort Flow": label, "": "Removed" + suffix, "N": n1})
            rows.append({"Cohort Flow": label, "": "Result" + suffix, "N": n2})
        
        return pd.DataFrame(rows).pivot(index="", columns="Cohort Flow", values="N")


# TableCharacteristics Class

class TableCharacteristics:
    """
    Generate tables of cohort characteristics (distributions).
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames representing sequential cohorts.
    categorical : list, optional
        List of categorical variable names.
    normal : list, optional
        List of normally-distributed variable names.
    nonnormal : list, optional
        List of non-normally distributed variable names.
    order_vars : list, optional
        List specifying the display order of variables.
    order_classes : dict, optional
        Dictionary mapping variable names to category order lists.
    limit : int or dict, optional
        Maximum categories to display per variable.
    decimals : int, default=1
        Number of decimal places.
    format_cat : str, default="N (%)"
        Format for categorical variables.
    format_normal : str, default="Mean ± SD"
        Format for normal variables.
    format_nonnormal : str, default="Median [IQR]"
        Format for non-normal variables.
    thousands_sep : bool, default=True
        Whether to use thousands separators.
    missingness : bool, default=True
        Whether to show missingness statistics.
    label_suffix : bool, default=True
        Whether to add format suffixes to labels.
    rename : dict, optional
        Variable name mappings.
    """
    
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        decimals: int = 1,
        format_cat: str = "N (%)",
        format_normal: str = "Mean ± SD",
        format_nonnormal: str = "Median [IQR]",
        thousands_sep: bool = True,
        missingness: bool = True,
        label_suffix: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ):
        if not dfs:
            raise ValueError("dfs list cannot be empty")
        
        self._dfs = dfs
        self._categorical = categorical or []
        self._normal = normal or []
        self._nonnormal = nonnormal or []
        self._order_vars = order_vars
        self._order_classes = order_classes or {}
        self._limit = limit
        self._decimals = decimals
        self._missingness = missingness
        self._format_cat = format_cat
        self._format_normal = format_normal
        self._format_nonnormal = format_nonnormal
        self._thousands_sep = thousands_sep
        self._label_suffix = label_suffix
        self._rename = rename or {}
        
        # Validate variables exist
        if dfs:
            validate_variables(
                dfs[0],
                self._categorical,
                self._normal,
                self._nonnormal,
                "TableCharacteristics"
            )

    def __repr__(self) -> str:
        n_vars = len(self._categorical) + len(self._normal) + len(self._nonnormal)
        return f"TableCharacteristics(cohorts={len(self._dfs)}, variables={n_vars})"

    def _get_limit_for_var(self, var: str) -> Optional[int]:
        """Get the category limit for a specific variable."""
        if self._limit is None:
            return None
        if isinstance(self._limit, int):
            return self._limit
        return self._limit.get(var)

    def _get_ordered_categories(self, col: str) -> List[Any]:
        """
        Get ordered unique values for a categorical variable.
        
        Applies order_classes if specified, then limit if specified.
        """
        # Get all unique values from the initial cohort
        all_values = self._dfs[0][col].dropna().unique().tolist()
        
        # Apply custom ordering if specified
        if col in self._order_classes:
            ordered = self._order_classes[col]
            # Start with specified order, then add any remaining values
            result = [v for v in ordered if v in all_values]
            result.extend([v for v in all_values if v not in ordered])
        else:
            # Default: sort by frequency in initial cohort (descending)
            value_counts = self._dfs[0][col].value_counts()
            result = [v for v in value_counts.index if v in all_values]
        
        # Apply limit if specified
        limit = self._get_limit_for_var(col)
        if limit is not None and len(result) > limit:
            result = result[:limit]
        
        return result

    def _my_value_counts(
        self,
        df: pd.DataFrame,
        col: str,
        categories: List[Any]
    ) -> pd.DataFrame:
        """
        Calculate value counts for a categorical variable.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to analyze.
        col : str
            Column name.
        categories : list
            List of categories to include (in order).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with counts/percentages for each category.
        """
        counts = pd.DataFrame(columns=[col], index=categories)
        n = len(df)
        
        if n == 0:
            for cat in categories:
                if self._format_cat == "%":
                    counts.loc[cat, col] = 0.0
                elif self._format_cat == "N":
                    counts.loc[cat, col] = "0" if self._thousands_sep else 0
                else:
                    counts.loc[cat, col] = "0 (0.0)" if self._thousands_sep else "0 (0.0)"
            return counts
        
        # Denominator: total n if showing missingness, else non-missing count
        denom = n if self._missingness else max(n - df[col].isna().sum(), 1)
        
        for cat in categories:
            count = (df[col] == cat).sum()
            
            if self._format_cat == "%":
                counts.loc[cat, col] = round(count / denom * 100, self._decimals)
            elif self._format_cat == "N":
                counts.loc[cat, col] = f"{count:,}" if self._thousands_sep else count
            else:
                perc = round(count / denom * 100, self._decimals)
                counts.loc[cat, col] = (
                    f"{count:,} ({perc})" if self._thousands_sep else f"{count} ({perc})"
                )
        
        return counts

    def view(self) -> pd.DataFrame:
        """
        Generate the characteristics table.
        
        Returns
        -------
        pd.DataFrame
            Multi-level table with characteristics for each cohort.
        """
        table = pd.DataFrame()
        
        # Pre-compute ordered categories for each categorical variable
        var_categories = {
            col: self._get_ordered_categories(col) for col in self._categorical
        }
        
        for i, df in enumerate(self._dfs):
            # Initialize DataFrame with MultiIndex
            index = pd.MultiIndex(
                levels=[[], []],
                codes=[[], []],
                names=["variable", "index"],
            )
            df_dists = pd.DataFrame(columns=["value"], index=index)
            
            # Determine processing order
            all_vars = set(self._categorical + self._normal + self._nonnormal)
            if self._order_vars:
                processing_order = [v for v in self._order_vars if v in all_vars]
                processing_order.extend([
                    v for v in self._categorical + self._normal + self._nonnormal
                    if v not in self._order_vars
                ])
            else:
                processing_order = self._categorical + self._normal + self._nonnormal
            
            for col in processing_order:
                display_name = self._rename.get(col, col)
                
                if col in self._categorical:
                    categories = var_categories[col]
                    counts = self._my_value_counts(df, col, categories)
                    
                    # Add rows for each category
                    for cat in categories:
                        df_dists.loc[(display_name, str(cat)), "value"] = counts.loc[cat, col]
                
                elif col in self._normal:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if self._format_normal == "Mean ± SD":
                        mean_val = np.round(vals.mean(), self._decimals)
                        std_val = np.round(vals.std(), self._decimals)
                        df_dists.loc[(display_name, " "), "value"] = f"{mean_val} ± {std_val}"
                    elif self._format_normal == "Mean":
                        df_dists.loc[(display_name, " "), "value"] = np.round(vals.mean(), self._decimals)
                    else:  # SD
                        df_dists.loc[(display_name, " "), "value"] = np.round(vals.std(), self._decimals)
                
                elif col in self._nonnormal:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if self._format_nonnormal == "Median [IQR]":
                        median_val = np.round(vals.median(), self._decimals)
                        q25 = np.round(vals.quantile(0.25), self._decimals)
                        q75 = np.round(vals.quantile(0.75), self._decimals)
                        df_dists.loc[(display_name, " "), "value"] = f"{median_val} [{q25}, {q75}]"
                    elif self._format_nonnormal == "Mean":
                        df_dists.loc[(display_name, " "), "value"] = np.round(vals.mean(), self._decimals)
                    else:  # SD
                        df_dists.loc[(display_name, " "), "value"] = np.round(vals.std(), self._decimals)
                
                # Add missingness row
                if self._missingness:
                    n = len(df)
                    n_miss = df[col].isna().sum() if n > 0 else 0
                    perc = round(n_miss / n * 100, self._decimals) if n > 0 else 0
                    
                    if self._format_cat == "%":
                        df_dists.loc[(display_name, "Missing"), "value"] = perc
                    elif self._format_cat == "N":
                        df_dists.loc[(display_name, "Missing"), "value"] = (
                            f"{n_miss:,}" if self._thousands_sep else n_miss
                        )
                    else:
                        df_dists.loc[(display_name, "Missing"), "value"] = (
                            f"{n_miss:,} ({perc})" if self._thousands_sep else f"{n_miss} ({perc})"
                        )
                
                # Add label suffix
                if self._label_suffix:
                    if col in self._categorical:
                        suffix = ", " + self._format_cat
                    elif col in self._normal:
                        suffix = ", " + self._format_normal
                    else:
                        suffix = ", " + self._format_nonnormal
                    
                    # Rename the variable level
                    df_dists = df_dists.rename(
                        index={display_name: display_name + suffix},
                        level=0
                    )
            
            # Add overall count
            df_dists.loc[("Overall", " "), "value"] = (
                f"{len(df):,}" if self._thousands_sep else len(df)
            )
            
            df_dists.rename(columns={"value": i}, inplace=True)
            table = pd.concat([table, df_dists], axis=1)
        
        # Set up multi-level column index
        table = table.set_axis(
            pd.MultiIndex.from_product([["Cohort"], table.columns]), axis=1
        )
        table.index.names = ["Variable", "Value"]
        
        # Sort to put Overall first
        return table.sort_index(
            level=0,
            key=lambda x: x == "Overall",
            ascending=False,
            sort_remaining=False
        )


# TableDrifts Class

class TableDrifts:
    """
    Calculate standardized mean differences (SMDs) between cohorts.
    
    SMDs quantify the magnitude of distribution changes between consecutive
    cohorts, helping identify demographic drift during cohort selection.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames representing sequential cohorts.
    categorical : list, optional
        List of categorical variable names.
    normal : list, optional
        List of normally-distributed variable names.
    nonnormal : list, optional
        List of non-normally distributed variable names.
    order_vars : list, optional
        List specifying the display order of variables.
    order_classes : dict, optional
        Dictionary mapping variable names to category order lists.
    limit : int or dict, optional
        Maximum categories to include in SMD calculations.
    decimals : int, default=2
        Number of decimal places for SMD values.
    missingness : bool, default=True
        Whether to include missingness in calculations.
    rename : dict, optional
        Variable name mappings.
    """
    
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        decimals: int = 2,
        missingness: bool = True,
        rename: Optional[Dict[str, str]] = None,
    ):
        if not dfs:
            raise ValueError("dfs list cannot be empty")
        
        self._dfs = dfs
        self._categorical = categorical or []
        self._normal = normal or []
        self._nonnormal = nonnormal or []
        self._order_vars = order_vars
        self._order_classes = order_classes or {}
        self._limit = limit
        self._decimals = decimals
        self._missingness = missingness
        self._rename = rename or {}
        
        # Validate variables exist
        if dfs:
            validate_variables(
                dfs[0],
                self._categorical,
                self._normal,
                self._nonnormal,
                "TableDrifts"
            )
        
        # Build complete rename mapping
        for c in self._categorical + self._normal + self._nonnormal:
            if c not in self._rename:
                self._rename[c] = c
        
        # Create helper tables for calculations
        self._table_flows = TableFlows(dfs, label_suffix=False, thousands_sep=False).view()
        self._table_cat_n = TableCharacteristics(
            dfs,
            categorical=self._categorical,
            normal=self._normal,
            nonnormal=self._nonnormal,
            order_classes=self._order_classes,
            limit=self._limit,
            decimals=self._decimals,
            format_cat="N",
            format_normal="Mean",
            format_nonnormal="Mean",
            thousands_sep=False,
            missingness=self._missingness,
            label_suffix=False,
            rename=self._rename,
        ).view()
        self._table_cat_perc = TableCharacteristics(
            dfs,
            categorical=self._categorical,
            normal=self._normal,
            nonnormal=self._nonnormal,
            order_classes=self._order_classes,
            limit=self._limit,
            decimals=self._decimals,
            format_cat="%",
            format_normal="SD",
            format_nonnormal="SD",
            thousands_sep=False,
            missingness=self._missingness,
            label_suffix=False,
            rename=self._rename,
        ).view()

    def __repr__(self) -> str:
        n_vars = len(self._categorical) + len(self._normal) + len(self._nonnormal)
        return f"TableDrifts(cohorts={len(self._dfs)}, variables={n_vars})"

    def _get_limit_for_var(self, var: str) -> Optional[int]:
        """Get the category limit for a specific variable."""
        if self._limit is None:
            return None
        if isinstance(self._limit, int):
            return self._limit
        return self._limit.get(var)

    def _cat_smd_binary(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int
    ) -> float:
        """
        Calculate SMD for a binary categorical variable.
        
        Uses Cohen's h with Hedges' correction for small samples.
        
        Returns np.nan if calculation cannot be performed.
        """
        # Handle edge cases
        if n1 <= 0 or n2 <= 0:
            return np.nan
        
        diff = abs(p1 - p2)
        pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        
        if pooled_var <= 0:
            return np.nan
        
        smd = diff / np.sqrt(pooled_var)
        
        # Hedges' correction for small samples
        df = n1 + n2 - 2
        if df > 0:
            correction = 1 - (3 / (4 * df - 1))
            smd *= correction
        
        return np.round(smd, self._decimals)

    def _cat_smd_multinomial(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: int,
        n2: int
    ) -> float:
        """
        Calculate SMD for a multinomial categorical variable.
        
        Uses Mahalanobis distance approach with pooled covariance matrix.
        
        Returns np.nan if calculation cannot be performed.
        """
        p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
        
        # Handle edge cases
        if n1 <= 0 or n2 <= 0 or len(p1) < 2:
            return np.nan
        
        # Calculate covariance matrices for each group
        def calc_cov(p):
            v = p * (1 - p)
            c = -np.outer(p, p)
            np.fill_diagonal(c, v)
            return c
        
        cov1, cov2 = calc_cov(p1), calc_cov(p2)
        pooled_cov = (cov1 + cov2) / 2
        
        diff = (p2 - p1).reshape(1, -1)
        
        try:
            # Use pseudo-inverse for numerical stability
            sq = diff @ np.linalg.pinv(pooled_cov) @ diff.T
            smd = np.sqrt(max(0, sq[0, 0]))  # Ensure non-negative
        except Exception:
            return np.nan
        
        return np.round(smd, self._decimals)

    def _cont_smd(
        self,
        m1: float,
        m2: float,
        s1: float,
        s2: float,
        n1: int,
        n2: int
    ) -> float:
        """
        Calculate SMD for a continuous variable.
        
        Uses Cohen's d with Hedges' correction for small samples.
        
        Parameters
        ----------
        m1, m2 : float
            Means of the two groups.
        s1, s2 : float
            Standard deviations of the two groups.
        n1, n2 : int
            Sample sizes of the two groups.
            
        Returns
        -------
        float
            Standardized mean difference, or np.nan if calculation fails.
        """
        # Handle edge cases
        if n1 <= 0 or n2 <= 0:
            return np.nan
        
        # Handle missing/invalid values
        if pd.isna(m1) or pd.isna(m2) or pd.isna(s1) or pd.isna(s2):
            return np.nan
        
        # Pooled standard deviation
        denom = np.sqrt((s1**2 + s2**2) / 2)
        
        if denom <= 0 or np.isnan(denom):
            return np.nan
        
        smd = (m2 - m1) / denom
        
        # Hedges' correction for small samples
        df = 4 * (n1 + n2 - 2) - 1
        if df > 0:
            correction = 1 - (3 / df)
            smd *= correction
        
        return np.round(smd, self._decimals)

    def view(self) -> pd.DataFrame:
        """
        Generate the SMD table.
        
        Returns
        -------
        pd.DataFrame
            Table of SMD values between consecutive cohorts.
        """
        inverse_rename = {v: k for k, v in self._rename.items()}
        
        # Get variable names (excluding Overall)
        cols = [
            c for c in self._table_cat_n.index.get_level_values(0).unique()
            if c != "Overall"
        ]
        
        table = pd.DataFrame(index=cols, columns=self._table_flows.columns)
        
        for i, display_name in enumerate(cols):
            orig_name = inverse_rename.get(display_name, display_name)
            
            for j, col in enumerate(self._table_flows.columns):
                if orig_name in self._categorical:
                    # Get category rows (excluding Missing)
                    cat_rows = [
                        r for r in self._table_cat_n.index
                        if r[0] == display_name and r[1] not in ["Missing"]
                    ]
                    
                    if not cat_rows:
                        table.iloc[i, j] = np.nan
                        continue
                    
                    # Get proportions for each group
                    p1 = [self._table_cat_perc.loc[r, :].iloc[j] / 100 for r in cat_rows]
                    p2 = [self._table_cat_perc.loc[r, :].iloc[j + 1] / 100 for r in cat_rows]
                    
                    # Get sample sizes
                    n1 = int(sum(self._table_cat_n.loc[r, :].iloc[j] for r in cat_rows))
                    n2 = int(sum(self._table_cat_n.loc[r, :].iloc[j + 1] for r in cat_rows))
                    
                    if len(p1) == 2:
                        # Binary variable
                        table.iloc[i, j] = self._cat_smd_binary(p1[0], p2[0], n1, n2)
                    elif len(p1) > 2:
                        # Multinomial variable
                        table.iloc[i, j] = self._cat_smd_multinomial(
                            np.array(p1), np.array(p2), n1, n2
                        )
                    else:
                        table.iloc[i, j] = np.nan
                
                elif orig_name in self._normal or orig_name in self._nonnormal:
                    # Continuous variable
                    m1 = self._table_cat_n.loc[(display_name, " "), :].iloc[j]
                    s1 = self._table_cat_perc.loc[(display_name, " "), :].iloc[j]
                    m2 = self._table_cat_n.loc[(display_name, " "), :].iloc[j + 1]
                    s2 = self._table_cat_perc.loc[(display_name, " "), :].iloc[j + 1]
                    n1 = int(self._table_cat_n.loc[("Overall", " "), :].iloc[j])
                    n2 = int(self._table_cat_n.loc[("Overall", " "), :].iloc[j + 1])
                    
                    table.iloc[i, j] = self._cont_smd(m1, m2, s1, s2, n1, n2)
        
        return table


# TablePValues Class

class TablePValues:
    """
    Calculate p-values between consecutive cohorts.
    
    Uses appropriate statistical tests based on variable type:
    - Categorical: Chi-square test (or Fisher's exact for 2x2 tables)
    - Normal continuous: Welch's t-test
    - Non-normal continuous: Kruskal-Wallis test
    - Missingness: Two-proportion z-test
    
    Supports multiple testing correction (Bonferroni or Benjamini-Hochberg FDR).
    
    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of DataFrames representing sequential cohorts.
    categorical : list, optional
        List of categorical variable names.
    normal : list, optional
        List of normally-distributed variable names.
    nonnormal : list, optional
        List of non-normally distributed variable names.
    order_vars : list, optional
        List specifying the display order of variables.
    order_classes : dict, optional
        Dictionary mapping variable names to category order lists.
    limit : int or dict, optional
        Maximum categories to include in tests.
    alpha : float, default=0.05
        Significance level.
    decimals : int, default=3
        Number of decimal places for p-values.
    min_n_expected : int, default=5
        Minimum expected count for chi-square test validity.
    min_samples : int, default=30
        Minimum sample size for parametric tests.
    rename : dict, optional
        Variable name mappings.
    correction : str, default="none"
        Multiple testing correction method applied globally across all
        non-missingness tests. Options: "none", "bonferroni", "fdr_bh"/"bh".
    """
    
    VALID_CORRECTIONS = ["none", "bonferroni", "fdr_bh", "bh"]
    
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        categorical: Optional[List[str]] = None,
        normal: Optional[List[str]] = None,
        nonnormal: Optional[List[str]] = None,
        order_vars: Optional[List[str]] = None,
        order_classes: Optional[Dict[str, List[str]]] = None,
        limit: Optional[Union[int, Dict[str, int]]] = None,
        alpha: float = 0.05,
        decimals: int = 3,
        min_n_expected: int = 5,
        min_samples: int = 30,
        rename: Optional[Dict[str, str]] = None,
        correction: str = "none",
    ):
        from scipy import stats  # noqa: F401
        
        if not dfs:
            raise ValueError("dfs list cannot be empty")
        
        self._dfs = dfs
        self._categorical = categorical or []
        self._normal = normal or []
        self._nonnormal = nonnormal or []
        self._order_vars = order_vars
        self._order_classes = order_classes or {}
        self._limit = limit
        self._alpha = alpha
        self._decimals = decimals
        self._min_n_expected = min_n_expected
        self._min_samples = min_samples
        self._rename = rename or {}
        
        # Validate and normalize correction method
        correction = correction.lower() if isinstance(correction, str) else "none"
        if correction not in self.VALID_CORRECTIONS:
            raise ValueError(
                f"Invalid correction method: '{correction}'. "
                f"Valid options are: {', '.join(self.VALID_CORRECTIONS)}"
            )
        if correction == "bh":
            correction = "fdr_bh"
        self._correction = correction
        
        # Validate variables exist
        if dfs:
            validate_variables(
                dfs[0],
                self._categorical,
                self._normal,
                self._nonnormal,
                "TablePValues"
            )
        
        self._table_flows = TableFlows(dfs=self._dfs, label_suffix=False, thousands_sep=False)
        
        # Build complete rename mapping
        for c in self._categorical + self._normal + self._nonnormal:
            if c not in self._rename:
                self._rename[c] = c

    def __repr__(self) -> str:
        n_vars = len(self._categorical) + len(self._normal) + len(self._nonnormal)
        return f"TablePValues(cohorts={len(self._dfs)}, variables={n_vars}, correction='{self._correction}')"

    def _benjamini_hochberg(self, pvals: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Parameters
        ----------
        pvals : np.ndarray
            Array of p-values to correct.
            
        Returns
        -------
        np.ndarray
            Corrected p-values.
        """
        n = len(pvals)
        if n == 0:
            return pvals
        
        sorted_indices = np.argsort(pvals)
        sorted_pvals = pvals[sorted_indices]
        
        ranks = np.arange(1, n + 1)
        adjusted = sorted_pvals * n / ranks
        
        # Enforce monotonicity (from largest to smallest)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.minimum(adjusted, 1.0)
        
        result = np.empty(n)
        result[sorted_indices] = adjusted
        return result

    def _apply_global_correction(
        self,
        pvals_df: pd.DataFrame,
        missingness_mask: pd.Series
    ) -> pd.DataFrame:
        """
        Apply multiple testing correction globally across all non-missingness tests.
        
        Parameters
        ----------
        pvals_df : pd.DataFrame
            DataFrame of raw p-values.
        missingness_mask : pd.Series
            Boolean Series indicating which rows are missingness tests.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with corrected p-values.
        """
        if self._correction == "none":
            return pvals_df
        
        corrected = pvals_df.copy()
        
        # Get all non-missingness p-values as a flat array
        non_missing_rows = ~missingness_mask
        non_missing_pvals = pvals_df.loc[non_missing_rows].values.flatten()
        
        # Remove NaN values for correction
        valid_mask = ~np.isnan(non_missing_pvals)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            return corrected
        
        valid_pvals = non_missing_pvals[valid_mask]
        
        if self._correction == "bonferroni":
            corrected_pvals = np.minimum(valid_pvals * n_valid, 1.0)
        elif self._correction == "fdr_bh":
            corrected_pvals = self._benjamini_hochberg(valid_pvals)
        else:
            corrected_pvals = valid_pvals
        
        # Put corrected values back
        non_missing_flat = corrected.loc[non_missing_rows].values.flatten()
        non_missing_flat[valid_mask] = corrected_pvals
        
        corrected.loc[non_missing_rows] = non_missing_flat.reshape(
            corrected.loc[non_missing_rows].shape
        )
        
        return corrected

    def _chi2_test(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        var: str
    ) -> Tuple[float, bool]:
        """
        Perform chi-square test for categorical variables.
        
        Falls back to Fisher's exact test for 2x2 tables with low expected counts.
        """
        from scipy import stats
        
        if df1[var].dropna().empty or df2[var].dropna().empty:
            return np.nan, False
        
        all_vals = pd.concat([df1[var].dropna(), df2[var].dropna()]).unique()
        
        if len(all_vals) < 2:
            return np.nan, False
        
        # Build contingency table
        tbl = np.zeros((2, len(all_vals)))
        for i, v in enumerate(all_vals):
            tbl[0, i] = (df1[var] == v).sum()
            tbl[1, i] = (df2[var] == v).sum()
        
        if np.any(tbl.sum(axis=0) == 0) or np.any(tbl.sum(axis=1) == 0):
            return np.nan, False
        
        try:
            _, p, _, exp = stats.chi2_contingency(tbl, correction=True)
            valid = not np.any(exp < self._min_n_expected)
            
            # Use Fisher's exact for 2x2 tables with low expected counts
            if not valid and len(all_vals) == 2:
                try:
                    _, p = stats.fisher_exact(tbl)
                    valid = True
                except Exception:
                    pass
            
            return p, valid
        except Exception:
            return np.nan, False

    def _t_test(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        var: str
    ) -> Tuple[float, bool]:
        """Perform Welch's t-test for normally-distributed continuous variables."""
        from scipy import stats
        
        try:
            v1 = pd.to_numeric(df1[var], errors="coerce").dropna()
            v2 = pd.to_numeric(df2[var], errors="coerce").dropna()
            
            if len(v1) < 2 or len(v2) < 2:
                return np.nan, False
            
            _, p = stats.ttest_ind(v1, v2, equal_var=False)
            return (np.nan, False) if np.isnan(p) else (p, True)
        except Exception:
            return np.nan, False

    def _kruskal_test(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        var: str
    ) -> Tuple[float, bool]:
        """Perform Kruskal-Wallis test for non-normal continuous variables."""
        from scipy import stats
        
        try:
            v1 = pd.to_numeric(df1[var], errors="coerce").dropna()
            v2 = pd.to_numeric(df2[var], errors="coerce").dropna()
            
            if len(v1) < 2 or len(v2) < 2:
                return np.nan, False
            
            _, p = stats.kruskal(v1, v2)
            return (np.nan, False) if np.isnan(p) else (p, True)
        except Exception:
            return np.nan, False

    def _missingness_test(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        var: str
    ) -> Tuple[float, bool]:
        """Perform two-proportion z-test for missingness differences."""
        from scipy import stats
        
        n1, n2 = len(df1), len(df2)
        
        if n1 == 0 or n2 == 0:
            return np.nan, False
        
        p1 = df1[var].isna().mean()
        p2 = df2[var].isna().mean()
        
        if p1 == p2:
            return 1.0, True
        
        # Pooled proportion
        pp = (p1 * n1 + p2 * n2) / (n1 + n2)
        
        # Standard error
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        
        if se == 0:
            return np.nan, False
        
        z = (p1 - p2) / se
        return 2 * (1 - stats.norm.cdf(abs(z))), True

    def view(self) -> pd.DataFrame:
        """
        Generate the p-value table with significance indicators.
        
        Returns
        -------
        pd.DataFrame
            Table of p-values with significance markers:
            * p < 0.05, ** p < 0.01, *** p < 0.001.
            † indicates corrected p-values.
        """
        tf = self._table_flows.view()
        cols = tf.columns
        proc = self._categorical + self._normal + self._nonnormal
        
        # Build index
        idx = [("Overall", " ")] + \
              [(self._rename[v], " ") for v in proc] + \
              [(self._rename[v], "Missing") for v in proc]
        index = pd.MultiIndex.from_tuples(idx, names=["Variable", "Value"])
        
        # Create missingness mask for correction
        missingness_rows = pd.Series(
            [r[1] == "Missing" for r in idx],
            index=index
        )
        
        # Calculate raw p-values
        pvals = pd.DataFrame(np.nan, index=index, columns=cols)
        
        for col in cols:
            try:
                fr, to = map(int, col.split(" to "))
                d1, d2 = self._dfs[fr], self._dfs[to]
                
                for v in self._categorical:
                    p, _ = self._chi2_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
                
                for v in self._normal:
                    p, _ = self._t_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
                
                for v in self._nonnormal:
                    p, _ = self._kruskal_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
            except Exception:
                pass
        
        # Apply global correction (excluding missingness tests)
        corrected_pvals = self._apply_global_correction(pvals, missingness_rows)
        
        # Format p-values with significance indicators
        table = pd.DataFrame("", index=index, columns=cols)
        correction_indicator = "†" if self._correction != "none" else ""
        
        for col in cols:
            for i in corrected_pvals.index:
                p = corrected_pvals.loc[i, col]
                if pd.notna(p):
                    if p < 0.001:
                        table.loc[i, col] = f"<0.001***{correction_indicator}"
                    elif p < 0.01:
                        table.loc[i, col] = f"{p:.{self._decimals}f}**{correction_indicator}"
                    elif p < 0.05:
                        table.loc[i, col] = f"{p:.{self._decimals}f}*{correction_indicator}"
                    else:
                        table.loc[i, col] = f"{p:.{self._decimals}f}{correction_indicator}"
        
        table = table.set_axis(
            pd.MultiIndex.from_product([["p-value"], table.columns]), axis=1
        )
        
        return table.sort_index(
            level=0,
            key=lambda x: x == "Overall",
            ascending=False,
            sort_remaining=False
        )

    def get_raw_pvalues(self) -> pd.DataFrame:
        """
        Get raw (uncorrected) p-values.
        
        Returns
        -------
        pd.DataFrame
            Table of raw p-values without formatting or correction.
        """
        tf = self._table_flows.view()
        cols = tf.columns
        proc = self._categorical + self._normal + self._nonnormal
        
        idx = [("Overall", " ")] + \
              [(self._rename[v], " ") for v in proc] + \
              [(self._rename[v], "Missing") for v in proc]
        index = pd.MultiIndex.from_tuples(idx, names=["Variable", "Value"])
        pvals = pd.DataFrame(np.nan, index=index, columns=cols)
        
        for col in cols:
            try:
                fr, to = map(int, col.split(" to "))
                d1, d2 = self._dfs[fr], self._dfs[to]
                
                for v in self._categorical:
                    p, _ = self._chi2_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
                
                for v in self._normal:
                    p, _ = self._t_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
                
                for v in self._nonnormal:
                    p, _ = self._kruskal_test(d1, d2, v)
                    pvals.loc[(self._rename[v], " "), col] = p
                    mp, _ = self._missingness_test(d1, d2, v)
                    pvals.loc[(self._rename[v], "Missing"), col] = mp
            except Exception:
                pass
        
        return pvals.sort_index(
            level=0,
            key=lambda x: x == "Overall",
            ascending=False,
            sort_remaining=False
        )

    def get_corrected_pvalues(self) -> pd.DataFrame:
        """
        Get corrected p-values (if correction is applied).
        
        Returns
        -------
        pd.DataFrame
            Table of corrected p-values without formatting.
        """
        pvals = self.get_raw_pvalues()
        
        if self._correction == "none":
            return pvals
        
        # Create missingness mask
        missingness_rows = pd.Series(
            [r[1] == "Missing" for r in pvals.index],
            index=pvals.index
        )
        
        return self._apply_global_correction(pvals, missingness_rows)


# FlowDiagram Class

class FlowDiagram:
    """
    Generate visual flow diagrams of cohort exclusion processes.
    
    Creates publication-ready flow diagrams using Graphviz with embedded
    distribution plots showing how variable distributions change at each step.
    
    Parameters
    ----------
    table_flows : TableFlows
        TableFlows object with cohort size data.
    table_characteristics : TableCharacteristics, optional
        TableCharacteristics object for distribution plots.
    table_drifts : TableDrifts, optional
        TableDrifts object for SMD values.
    table_pvalues : TablePValues, optional
        TablePValues object for p-values.
    new_cohort_labels : list of str, optional
        Labels for cohort nodes.
    exclusion_labels : list of str, optional
        Labels for exclusion nodes.
    box_width : float, default=2.5
        Width of diagram boxes.
    box_height : float, default=1.0
        Height of diagram boxes.
    plot_dists : bool, default=True
        Whether to include distribution plots.
    smds : bool, default=True
        Whether to show SMD values.
    smd_decimals : int, default=2
        Decimal places for SMD values.
    pvalues : bool, default=False
        Whether to show p-values.
    pvalue_decimals : int, default=3
        Decimal places for p-values.
    legend : bool, default=True
        Whether to include a legend.
    legend_with_vars : bool, default=True
        Whether to include variable names in legend.
    output_folder : str, default="imgs"
        Output folder for generated files.
    output_file : str, default="flow_diagram"
        Base name for output files.
    display_flow_diagram : bool, default=True
        Whether to display the diagram.
    cohort_node_color : str, default="white"
        Fill color for cohort nodes.
    exclusion_node_color : str, default="floralwhite"
        Fill color for exclusion nodes.
    categorical_bar_colors : list or dict, optional
        Colors for categorical bars.
    missing_value_color : str, default="lightgray"
        Color for missing value bars.
    continuous_var_color : str, default="lavender"
        Color for continuous variable bars.
    edge_color : str, default="black"
        Color for diagram edges.
    """
    
    def __init__(
        self,
        table_flows: TableFlows,
        table_characteristics: Optional[TableCharacteristics] = None,
        table_drifts: Optional[TableDrifts] = None,
        table_pvalues: Optional[TablePValues] = None,
        new_cohort_labels: Optional[List[str]] = None,
        exclusion_labels: Optional[List[str]] = None,
        box_width: float = 2.5,
        box_height: float = 1.0,
        plot_dists: bool = True,
        smds: bool = True,
        smd_decimals: int = 2,
        pvalues: bool = False,
        pvalue_decimals: int = 3,
        legend: bool = True,
        legend_with_vars: bool = True,
        output_folder: str = "imgs",
        output_file: str = "flow_diagram",
        display_flow_diagram: bool = True,
        cohort_node_color: str = "white",
        exclusion_node_color: str = "floralwhite",
        categorical_bar_colors: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        missing_value_color: str = "lightgray",
        continuous_var_color: str = "lavender",
        edge_color: str = "black",
    ):
        self._table_flows = table_flows.view()
        self._table_characteristics = table_characteristics
        self._table_drifts = table_drifts
        self._table_pvalues = table_pvalues
        self._smd_decimals = smd_decimals
        self._pvalues = pvalues
        self._pvalue_decimals = pvalue_decimals
        self._cohort_labels = new_cohort_labels or [
            f"Cohort {i}" for i in range(len(self._table_flows.columns) + 1)
        ]
        self._exclusion_labels = exclusion_labels or [
            f"Exclusion {i}" for i in range(len(self._table_flows.columns))
        ]
        self._width = box_width
        self._height = box_height
        self._plot_dists = plot_dists
        self._smds = smds if plot_dists else False
        self._legend = legend if plot_dists else False
        self._legend_with_vars = legend_with_vars
        self._output_file = output_file
        self._output_folder = output_folder
        self._cohort_node_color = cohort_node_color
        self._exclusion_node_color = exclusion_node_color
        self._categorical_bar_colors = categorical_bar_colors
        self._missing_value_color = missing_value_color
        self._continuous_var_color = continuous_var_color
        self._edge_color = edge_color
        self._display = display_flow_diagram
        
        os.makedirs(self._output_folder, exist_ok=True)

    def __repr__(self) -> str:
        n_cohorts = len(self._table_flows.columns) + 1
        return f"FlowDiagram(cohorts={n_cohorts}, plot_dists={self._plot_dists})"

    def _format_smd_value(self, v: Any) -> str:
        """Format SMD value for display in plots."""
        return format_smd(v, self._smd_decimals)

    def _format_pvalue_value(self, p: Any) -> str:
        """Format p-value for display in plots."""
        if pd.isna(p) or p == "":
            return ""
        try:
            if isinstance(p, str):
                # Remove significance markers
                return p.replace("*", "").replace("†", "").strip()
            f = float(p)
            if f < 0.001:
                return "<.001"
            return f"{f:.3f}"[1:] if f < 1 else f"{f:.2f}"
        except Exception:
            return str(p)

    def _plot_dists_internal(self) -> None:
        """Generate distribution plots for each cohort."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        imgs_dir = os.path.join(self._output_folder, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        
        categorical = self._table_characteristics._categorical
        table = self._table_characteristics.view()
        table_smds = self._table_drifts.view() if self._smds else None
        table_pvals = self._table_pvalues.view() if self._pvalues and self._table_pvalues else None
        
        vars_list = [
            v for v in table.index.get_level_values(0).unique() if v != "Overall"
        ]
        cohorts = table.columns.get_level_values(1).unique().tolist()
        
        # Assign colors to categorical variables
        var_colors = {
            var: cm.get_cmap("tab10")(i % 10) for i, var in enumerate(categorical)
        }
        legend_handles, legend_labels = [], []
        
        # Pre-compute sorted order of categories for each variable
        var_sorted_categories = {}
        for var in vars_list:
            orig_var = var.split(", ")[0] if ", " in var else var
            var_original = next(
                (k for k, v in self._table_characteristics._rename.items() if v == orig_var),
                orig_var,
            )
            if var_original in categorical:
                vals = [
                    v for v in table.loc[
                        table.index.get_level_values(0) == var
                    ].index.get_level_values(1)
                    if v not in ["Missing"]
                ]
                # Get percentages for sorting
                percs = {}
                for v in vals:
                    try:
                        val = _safe_scalar(table.loc[(var, v), ("Cohort", 0)])
                        if isinstance(val, str):
                            val = (
                                float(val.replace("%", "").split("(")[-1].replace(")", ""))
                                if "(" in val
                                else float(val.replace("%", ""))
                            )
                        percs[v] = float(val)
                    except Exception:
                        percs[v] = 0
                var_sorted_categories[var] = sorted(
                    percs.keys(), key=lambda x: percs[x], reverse=True
                )
        
        # Font sizes for plots
        label_fontsize = 18
        value_fontsize = 15
        bar_height = 0.9
        
        for c, coh in enumerate(cohorts):
            num_vars = len(vars_list)
            fig_height = max(4.5, num_vars * 1.1)
            
            # Determine figure width based on columns shown
            if self._smds and self._pvalues:
                fig, ax = plt.subplots(1, 1, figsize=(10, fig_height), dpi=150)
                ax.set_xlim([-32, 125])
            elif self._smds or self._pvalues:
                fig, ax = plt.subplots(1, 1, figsize=(9.5, fig_height), dpi=150)
                ax.set_xlim([-18, 125])
            else:
                fig, ax = plt.subplots(1, 1, figsize=(9, fig_height), dpi=150)
                ax.set_xlim([0, 125])
            
            for v, var in enumerate(vars_list):
                orig_var = var.split(", ")[0] if ", " in var else var
                var_original = next(
                    (k for k, vn in self._table_characteristics._rename.items() if vn == orig_var),
                    orig_var,
                )
                
                if var_original in categorical:
                    vals = var_sorted_categories.get(var, [])
                    cum = 0
                    
                    for vi, val in enumerate(vals):
                        value = _safe_scalar(table.loc[(var, val), ("Cohort", coh)])
                        if isinstance(value, str):
                            try:
                                value = (
                                    float(
                                        value.replace("%", "")
                                        .split("(")[-1]
                                        .replace(")", "")
                                    )
                                    if "(" in value
                                    else float(value.replace("%", ""))
                                )
                            except Exception:
                                value = 0
                        value = float(value) if not isinstance(value, (int, float)) else value
                        
                        # Determine bar color
                        if isinstance(self._categorical_bar_colors, dict) and (
                            var_original in self._categorical_bar_colors
                        ):
                            col_list = self._categorical_bar_colors[var_original]
                            color = col_list[vi] if vi < len(col_list) else None
                            bar = (
                                ax.barh(
                                    v, value, left=cum, height=bar_height,
                                    color=color, edgecolor="white", linewidth=0.5,
                                )
                                if color
                                else ax.barh(
                                    v, value, left=cum, height=bar_height,
                                    edgecolor="white", linewidth=0.5,
                                )
                            )
                        elif (
                            self._categorical_bar_colors
                            and isinstance(self._categorical_bar_colors, list)
                            and vi < len(self._categorical_bar_colors)
                        ):
                            bar = ax.barh(
                                v, value, left=cum, height=bar_height,
                                color=self._categorical_bar_colors[vi],
                                edgecolor="white", linewidth=0.5,
                            )
                        elif var_original in var_colors:
                            bc = var_colors[var_original]
                            lt = 0.3 + 0.6 * (vi / max(1, len(vals) - 1))
                            ac = mcolors.rgb_to_hsv(bc[:3])
                            ac[1] *= lt
                            ac[2] = min(1.0, ac[2] * (1.5 - lt * 0.5))
                            bar = ax.barh(
                                v, value, left=cum, height=bar_height,
                                color=mcolors.hsv_to_rgb(ac),
                                edgecolor="white", linewidth=0.5,
                            )
                        else:
                            bar = ax.barh(
                                v, value, left=cum, height=bar_height,
                                edgecolor="white", linewidth=0.5,
                            )
                        
                        if coh == 0:
                            legend_handles.append(bar[0])
                            legend_labels.append(
                                f"{orig_var}: {val}" if self._legend_with_vars else val
                            )
                        
                        # Show value labels inside bars
                        if value > 5:
                            ax.text(
                                cum + value / 2, v,
                                f"{value:.1f}",
                                ha="center", va="center",
                                color="white" if value > 25 else "black",
                                fontsize=value_fontsize,
                                fontweight="medium",
                            )
                        cum += value
                    
                    # Add missing values bar
                    if (
                        "Missing"
                        in table.loc[
                            table.index.get_level_values(0) == var
                        ].index.get_level_values(1)
                    ):
                        mv = _safe_scalar(table.loc[(var, "Missing"), ("Cohort", coh)])
                        if isinstance(mv, str):
                            try:
                                mv = (
                                    float(
                                        mv.replace("%", "")
                                        .split("(")[-1]
                                        .replace(")", "")
                                    )
                                    if "(" in mv
                                    else float(mv.replace("%", ""))
                                )
                            except Exception:
                                mv = 0
                        if mv > 0:
                            bar = ax.barh(
                                v, mv, left=cum, height=bar_height,
                                color=self._missing_value_color,
                                hatch="///////",
                                edgecolor="white", linewidth=0.5,
                            )
                            if coh == 0 and "Missing" not in legend_labels:
                                legend_handles.append(bar[0])
                                legend_labels.append("Missing")
                            if mv > 5:
                                ax.text(
                                    cum + mv / 2, v,
                                    f"{mv:.1f}",
                                    ha="center", va="center",
                                    color="black",
                                    fontsize=value_fontsize,
                                    fontweight="medium",
                                )
                    
                    # Add SMD
                    if coh > 0 and self._smds and table_smds is not None:
                        if orig_var in table_smds.index:
                            smd = _safe_scalar(table_smds.loc[orig_var, f"{coh-1} to {coh}"])
                            ax.text(
                                -3, v,
                                self._format_smd_value(smd),
                                ha="right", va="center",
                                fontsize=value_fontsize,
                                color="black", fontweight="medium",
                            )
                    
                    # Add p-value
                    if coh > 0 and self._pvalues and table_pvals is not None:
                        vr = self._table_characteristics._rename.get(var_original, var_original)
                        try:
                            if (vr, " ") in table_pvals.index:
                                pv = _safe_scalar(table_pvals.loc[(vr, " "), ("p-value", f"{coh-1} to {coh}")])
                                ax.text(
                                    -18 if self._smds else -3, v,
                                    self._format_pvalue_value(pv),
                                    ha="right", va="center",
                                    fontsize=value_fontsize,
                                    color="black", fontweight="medium",
                                )
                        except Exception:
                            pass
                else:
                    # Continuous variable
                    value = _safe_scalar(table.loc[(var, " "), ("Cohort", coh)])
                    ax.barh(
                        v, 100, left=0, height=bar_height,
                        color=self._continuous_var_color,
                        edgecolor="white", linewidth=0.5,
                    )
                    ax.text(
                        50, v,
                        f"{value}",
                        ha="center", va="center",
                        color="black",
                        fontsize=value_fontsize,
                        fontweight="medium",
                    )
                    
                    # Add SMD
                    if coh > 0 and self._smds and table_smds is not None:
                        if orig_var in table_smds.index:
                            smd = _safe_scalar(table_smds.loc[orig_var, f"{coh-1} to {coh}"])
                            ax.text(
                                -3, v,
                                self._format_smd_value(smd),
                                ha="right", va="center",
                                fontsize=value_fontsize,
                                color="black", fontweight="medium",
                            )
                    
                    # Add p-value
                    if coh > 0 and self._pvalues and table_pvals is not None:
                        vr = self._table_characteristics._rename.get(var_original, var_original)
                        try:
                            if (vr, " ") in table_pvals.index:
                                pv = _safe_scalar(table_pvals.loc[(vr, " "), ("p-value", f"{coh-1} to {coh}")])
                                ax.text(
                                    -18 if self._smds else -3, v,
                                    self._format_pvalue_value(pv),
                                    ha="right", va="center",
                                    fontsize=value_fontsize,
                                    color="black", fontweight="medium",
                                )
                        except Exception:
                            pass
                
                # Variable label on the right
                ax.text(
                    105, v,
                    orig_var,
                    ha="left", va="center",
                    fontsize=label_fontsize,
                    color="black", fontweight="medium",
                )
            
            # Add column headers
            if self._smds and coh > 0:
                ax.text(
                    -3, len(vars_list) + 0.6,
                    "SMD",
                    ha="right", va="center",
                    fontsize=label_fontsize,
                    color="black", fontweight="bold",
                )
            if self._pvalues and coh > 0:
                ax.text(
                    -18 if self._smds else -3, len(vars_list) + 0.6,
                    "p",
                    ha="right", va="center",
                    fontsize=label_fontsize,
                    color="black", fontweight="bold",
                )
            
            # Clean up axes
            ax.set_yticks([])
            ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(imgs_dir, f"part{c}.svg"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        
        # Create legend
        if self._legend and legend_handles:
            if "Missing" in legend_labels:
                mi = legend_labels.index("Missing")
                legend_labels.append(legend_labels.pop(mi))
                legend_handles.append(legend_handles.pop(mi))
            
            lf, la = plt.subplots(figsize=(len(legend_labels) / 2.5, 2.8))
            la.axis("off")
            leg = la.legend(
                legend_handles,
                legend_labels,
                loc="center",
                ncol=1,
                fontsize=label_fontsize,
                frameon=True,
                edgecolor="black",
                fancybox=False,
                title="Legend",
                title_fontsize=label_fontsize,
            )
            leg.get_title().set_fontweight("bold")
            leg.get_frame().set_linewidth(2)
            lf.savefig(
                os.path.join(imgs_dir, "legend.svg"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(lf)

    def view(self) -> None:
        """Create and display the flow diagram."""
        if self._plot_dists:
            self._plot_dists_internal()
        
        os.makedirs(self._output_folder, exist_ok=True)
        imgs_dir = os.path.join(self._output_folder, "imgs")
        os.makedirs(imgs_dir, exist_ok=True)
        
        dot = graphviz.Digraph(
            comment="Cohort Exclusion Process",
            format="svg",
            graph_attr={"fontname": "Helvetica", "splines": "ortho"},
            node_attr={
                "shape": "box",
                "style": "filled",
                "fixedsize": "true",
                "width": str(self._width),
                "height": str(self._height),
                "fontname": "Helvetica",
            },
            edge_attr={
                "dir": "forward",
                "arrowhead": "vee",
                "arrowsize": "0.5",
                "minlen": "1",
            },
        )
        
        cols = self._table_flows.columns.tolist()
        nc = len(cols)
        
        # Add cohort nodes
        initial_counts = self._table_flows.loc["Initial, n"]
        for i, (cnt, col) in enumerate(zip(initial_counts, cols)):
            dot.node(
                f"A{i}",
                self._cohort_labels[i].replace("___", f"{cnt}"),
                shape="box",
                style="filled",
                fillcolor=self._cohort_node_color,
                fontname="Helvetica",
            )
        
        dot.node(
            f"A{nc}",
            self._cohort_labels[-1].replace(
                "___", f"{self._table_flows.loc['Result, n'].iloc[-1]}"
            ),
            shape="box",
            style="filled",
            fillcolor=self._cohort_node_color,
            fontname="Helvetica",
        )
        
        # Calculate plot width for image nodes
        pw = (
            2.8
            if self._smds and self._pvalues
            else (2.5 if self._smds or self._pvalues else 2.2)
        )
        
        if self._plot_dists:
            # Use relative path from output folder
            img_path = os.path.join("imgs", f"part{nc}.svg")
            dot.node(
                f"plot_dist{nc}",
                label="",
                image=img_path,
                imagepos="bc",
                imagescale="true",
                shape="box",
                color="transparent",
                width=str(self._width + pw),
                height=str(self._height + 1.4),
            )
            with dot.subgraph() as s:
                s.attr(rank="same")
                s.node(f"A{nc}")
                s.node(f"plot_dist{nc}")
        
        # Add exclusion nodes
        removed = self._table_flows.loc["Removed, n"]
        for i, (cnt, col) in enumerate(zip(removed, cols)):
            dot.node(
                f"E{i}",
                self._exclusion_labels[i].replace("___", f"{cnt}"),
                shape="box",
                style="filled",
                fillcolor=self._exclusion_node_color,
            )
        
        # Add invisible nodes and edges for layout
        for i in range(nc + 1):
            dot.node(f"IA{i}", "", shape="point", height="0")
        
        for i in range(nc):
            dot.edge(f"A{i}", f"IA{i}", arrowhead="none", color=self._edge_color)
            dot.edge(f"IA{i}", f"A{i+1}", color=self._edge_color)
            dot.edge(f"IA{i}", f"E{i}", constraint="false", color=self._edge_color)
            with dot.subgraph() as s:
                s.attr(rank="same")
                s.node(f"IA{i}")
                s.node(f"E{i}")
        
        if self._plot_dists:
            for i in range(nc):
                # Use relative path from output folder
                img_path = os.path.join("imgs", f"part{i}.svg")
                dot.node(
                    f"plot_dist{i}",
                    label="",
                    image=img_path,
                    imagepos="bc",
                    imagescale="true",
                    shape="box",
                    color="transparent",
                    width=str(self._width + pw),
                    height=str(self._height + 1.4),
                )
                dot.edge(f"A{i}", f"plot_dist{i}", constraint="false", style="invis")
                with dot.subgraph() as s:
                    s.attr(rank="same")
                    s.node(f"A{i}")
                    s.node(f"plot_dist{i}")
        
        if self._legend:
            # Use relative path from output folder
            legend_path = os.path.join("imgs", "legend.svg")
            dot.node(
                "legend",
                label="",
                image=legend_path,
                imagescale="true",
                shape="box",
                color="transparent",
                imagepos="bl",
                width=str(self._width + 1.8),
                height=str(self._height + 1.9),
            )
            dot.edge("E0", "legend", style="invis")
            with dot.subgraph() as s:
                s.attr(rank="same")
                s.node("E0")
                s.node("legend")
        
        # Render from output folder so relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(self._output_folder)
            dot.render(
                self._output_file,
                view=self._display,
                format="pdf",
            )
        finally:
            os.chdir(original_cwd)


# EasyFlow Class (Simplified Interface)

class EasyFlow:
    """
    Simplified interface for creating equity-focused cohort flow diagrams.
    
    EasyFlow provides a chainable API for quick flow diagram generation
    with automatic variable type detection.
    
    Parameters
    ----------
    data : pd.DataFrame
        The initial cohort data.
    title : str, default="Cohort Selection"
        Title for the initial cohort.
    auto_detect : bool, default=True
        Whether to automatically detect variable types.
        
    Examples
    --------
    >>> from equiflow import EasyFlow
    >>> 
    >>> flow = (
    ...     EasyFlow(df, title="Study Cohort")
    ...     .categorize(['sex', 'race'])
    ...     .measure_normal(['age'])
    ...     .measure_nonnormal(['income'])
    ...     .exclude(df['age'] >= 18, "Age < 18", "Adult cohort")
    ...     .exclude(df['missing_data'] <= 0.5, "Too much missing data", "Complete data")
    ...     .generate(output="my_flow", show=True)
    ... )
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        title: str = "Cohort Selection",
        auto_detect: bool = True
    ):
        validate_dataframe_not_empty(data, "EasyFlow.__init__")
        
        self._data = data.copy()
        self._title = title
        self._categorical_vars: List[str] = []
        self._normal_vars: List[str] = []
        self._nonnormal_vars: List[str] = []
        self._exclusion_steps: List[Dict[str, Any]] = []
        self._current_data = data.copy()
        self._equiflow: Optional[EquiFlow] = None
        
        if auto_detect:
            self._detect_variable_types()

    def __repr__(self) -> str:
        n_steps = len(self._exclusion_steps)
        initial_n = len(self._data)
        current_n = len(self._current_data)
        return (
            f"EasyFlow(steps={n_steps}, initial_n={initial_n:,}, "
            f"current_n={current_n:,})"
        )

    def _detect_variable_types(self) -> None:
        """
        Automatically detect variable types based on data characteristics.
        
        Uses the following heuristics:
        - Object/bool dtype or <= 10 unique values -> categorical
        - Numeric with > 10 unique values -> check normality
        - Normal distribution -> normal
        - Non-normal distribution -> nonnormal
        """
        for col in self._data.columns:
            # Skip columns with too much missing data
            if self._data[col].isna().mean() > 0.5:
                continue
            
            if self._data[col].dtype in ["object", "bool"]:
                self._categorical_vars.append(col)
            elif self._data[col].dtype.kind in "ifu":
                n_unique = len(self._data[col].dropna().unique())
                if n_unique <= 10:
                    self._categorical_vars.append(col)
                else:
                    # Check for normality
                    if check_normality(self._data[col]):
                        self._normal_vars.append(col)
                    else:
                        self._nonnormal_vars.append(col)

    def categorize(self, variables: List[str]) -> 'EasyFlow':
        """
        Set categorical variables.
        
        Parameters
        ----------
        variables : list of str
            Names of categorical variables.
            
        Returns
        -------
        EasyFlow
            Self, for method chaining.
        """
        self._categorical_vars = variables
        return self

    def measure_normal(self, variables: List[str]) -> 'EasyFlow':
        """
        Set normally-distributed continuous variables.
        
        Parameters
        ----------
        variables : list of str
            Names of normal continuous variables.
            
        Returns
        -------
        EasyFlow
            Self, for method chaining.
        """
        self._normal_vars = variables
        return self

    def measure_nonnormal(self, variables: List[str]) -> 'EasyFlow':
        """
        Set non-normally distributed continuous variables.
        
        Parameters
        ----------
        variables : list of str
            Names of non-normal continuous variables.
            
        Returns
        -------
        EasyFlow
            Self, for method chaining.
        """
        self._nonnormal_vars = variables
        return self

    def exclude(
        self,
        condition: Union[pd.Series, Callable[[pd.DataFrame], pd.Series]],
        label: Optional[str] = None,
        cohort_label: Optional[str] = None
    ) -> 'EasyFlow':
        """
        Add an exclusion step.
        
        Note: Rows where condition is True are KEPT (retained).
        This follows the same convention as EquiFlow.add_exclusion().
        
        Parameters
        ----------
        condition : pd.Series or callable
            Boolean mask where True indicates rows to KEEP,
            or a function that takes a DataFrame and returns such a mask.
        label : str, optional
            Label for this exclusion step (describes what was excluded).
        cohort_label : str, optional
            Label for the resulting cohort after this exclusion.
            
        Returns
        -------
        EasyFlow
            Self, for method chaining.
            
        Examples
        --------
        >>> # Keep patients 18 and older (exclude those under 18)
        >>> flow.exclude(df['age'] >= 18, "Age < 18", "Adult cohort")
        >>> 
        >>> # Keep using a function with custom cohort label
        >>> flow.exclude(lambda d: d['score'] >= 50, "Low score", "High performers")
        """
        label = label or f"Exclusion {len(self._exclusion_steps) + 1}"
        
        if callable(condition):
            mask = condition(self._current_data)
        elif isinstance(condition, pd.Series) and condition.dtype == bool:
            # Align mask with current data
            mask = pd.Series(False, index=self._current_data.index)
            common_idx = self._current_data.index.intersection(condition.index)
            mask.loc[common_idx] = condition.loc[common_idx]
        else:
            mask = condition
        
        # KEEP rows where mask is True
        new_data = self._current_data[mask]
        
        if new_data.empty:
            warnings.warn(
                f"Exclusion step '{label}' resulted in an empty cohort.",
                UserWarning
            )
        
        self._exclusion_steps.append({
            "previous_data": self._current_data.copy(),
            "condition": condition,
            "new_data": new_data.copy(),
            "label": label,
            "cohort_label": cohort_label,
        })
        self._current_data = new_data
        
        return self

    def generate(
        self,
        output: str = "flow_diagram",
        show: bool = True,
        pvalues: bool = False,
        pvalue_decimals: int = 3,
        pvalue_correction: str = "none",
    ) -> 'EasyFlow':
        """
        Generate the flow diagram.
        
        Parameters
        ----------
        output : str, default="flow_diagram"
            Base name for output files.
        show : bool, default=True
            Whether to display the diagram.
        pvalues : bool, default=False
            Whether to show p-values.
        pvalue_decimals : int, default=3
            Decimal places for p-values.
        pvalue_correction : str, default="none"
            Multiple testing correction method.
            
        Returns
        -------
        EasyFlow
            Self, for method chaining.
        """
        if not self._exclusion_steps:
            raise ValueError(
                "No exclusion steps defined. Use exclude() to add steps."
            )
        
        # Create EquiFlow object
        ef = EquiFlow(
            data=self._data,
            initial_cohort_label=self._title,
            categorical=self._categorical_vars,
            normal=self._normal_vars,
            nonnormal=self._nonnormal_vars,
        )
        
        # Add exclusion steps using stored filtered DataFrames directly
        # This avoids index alignment issues that occur when recomputing masks
        for i, step in enumerate(self._exclusion_steps):
            # Use custom cohort_label if provided, otherwise fall back to default
            cohort_label = step.get("cohort_label") or f"Step {i + 1}"
            ef.add_exclusion(
                new_cohort=step["new_data"],
                exclusion_reason=step["label"],
                new_cohort_label=cohort_label,
            )
        
        self._equiflow = ef
        
        # Generate tables
        self.flow_table = ef.view_table_flows()
        self.characteristics = ef.view_table_characteristics()
        self.drifts = ef.view_table_drifts()
        
        # Generate flow diagram
        ef.plot_flows(
            output_file=output,
            display_flow_diagram=show,
            pvalues=pvalues,
            pvalue_decimals=pvalue_decimals,
            pvalue_correction=pvalue_correction,
        )
        
        return self
