"""
The equiflow package is used for creating "Equity-focused Cohort Section Flow Diagrams"
for cohort selection in clinical and machine learning papers.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import graphviz


class EquiFlow:


  def __init__(self,
               data: Optional[pd.DataFrame] = None,
               dfs: Optional[list] = None,
               initial_cohort_label: Optional[str] = None,
               label_suffix: Optional[bool] = True,
               thousands_sep: Optional[bool] = True,
               categorical: Optional[list] = None,
               normal: Optional[list] = None,
               nonnormal: Optional[list] = None,
               decimals: Optional[int] = 1,
               format_cat: Optional[str] = 'N (%)',
               format_normal: Optional[str] = 'Mean ± SD',
               format_nonnormal: Optional[str] = 'Median [IQR]',
               missingness: Optional[bool] = True,
               rename: Optional[dict] = None,

               ) -> None:
    """
    
    """
    
    if (data is None) & (dfs is None):
      raise ValueError("Either data or dfs must be provided")
    
    if (data is not None) & (dfs is not None):
      raise ValueError("Only one of data or dfs must be provided")
    
    if (data is not None) & (not isinstance(data, pd.DataFrame)):
      raise ValueError("data must be a pandas DataFrame")
    
    if (dfs is not None) & (not isinstance(dfs, list) or len(dfs) < 1):
      raise ValueError("dfs must be a list with length ≥ 1")
    
    if (initial_cohort_label is not None) & (not isinstance(initial_cohort_label, str)):
      raise ValueError("initial_cohort_label must be a string")
    
    if not isinstance(label_suffix, bool):
      raise ValueError("label_suffix must be a boolean")
    
    if not isinstance(thousands_sep, bool):
      raise ValueError("thousands_sep must be a boolean")
    
    if (categorical is not None) & (not isinstance(categorical, list)):
      raise ValueError("categorical must be a list")
    
    if (normal is not None) & (not isinstance(normal, list)):
      raise ValueError("normal must be a list")
    
    if (nonnormal is not None) & (not isinstance(nonnormal, list)):
      raise ValueError("nonnormal must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
      raise ValueError("decimals must be a non-negative integer")
    
    if format_cat not in ['%', 'N', 'N (%)']:
      raise ValueError("format must be '%', 'N', or 'N (%)'")
    
    if format_normal not in ['Mean ± SD', 'Mean', 'SD']:
      raise ValueError("format must be 'Mean ± SD' or 'Mean' or 'SD'")
    
    if format_nonnormal not in ['Median [IQR]', 'Mean', 'SD']:
      raise ValueError("format must be 'Median [IQR]' or 'Mean' or 'SD'")
    
    if not isinstance(missingness, bool):
      raise ValueError("missingness must be a boolean")
    
    if (rename is not None) & (not isinstance(rename, dict)):
      raise ValueError("rename must be a dictionary")
    
    if data is not None:
      self._dfs = [data]

    if dfs is not None:
      self._dfs = dfs

    self._clean_missing()

    self.label_suffix = label_suffix
    self.thousands_sep = thousands_sep
    self.categorical = categorical
    self.normal = normal
    self.nonnormal = nonnormal
    self.decimals = decimals
    self.format_cat = format_cat
    self.format_normal = format_normal
    self.format_nonnormal = format_nonnormal
    self.missingness = missingness
    self.rename = rename

    self.table_flows = None
    self.table_characteristics = None
    self.table_drifts = None
    self.flow_diagram = None

    self.exclusion_labels = {}
    self.new_cohort_labels = {}

    if initial_cohort_label is not None:
      self.new_cohort_labels[0] = initial_cohort_label
    else:
      self.new_cohort_labels[0] = 'Initial Cohort'



  # method to categorize missing values under the same label
  def _clean_missing(self): 
    
    for i, df in enumerate(self._dfs):

      # map all missing values possibilities to Null
      df = df.replace(['', ' ', 'NA', 'N/A', 'na', 'n/a',
                      'NA ', 'N/A ', 'na ', 'n/a ', 'NaN',
                      'nan', 'NAN', 'Nan', 'N/A;', '<NA>',
                      "_", '__', '___', '____', '_____',
                      'NaT', 'None', 'none', 'NONE', 'Null',
                      'null', 'NULL', 'missing', 'Missing',
                      np.nan, pd.NA], None)
  
    self._dfs[i] = df
      

 
  def add_exclusion(self,
                    mask: Optional[bool] = None,
                    new_cohort: Optional[pd.DataFrame] = None,
                    exclusion_reason: Optional[str] = None,
                    new_cohort_label: Optional[str] = None,
                    ):
    
    if (mask is None) & (new_cohort is None):
      raise ValueError("Either mask or new_cohort must be provided")
    
    if (mask is not None) & (new_cohort is not None):
      raise ValueError("Only one of mask or new_cohort must be provided")
    
    if mask is not None:
      self._dfs.append(self._dfs[-1].loc[mask])

    if new_cohort is not None:
      # first make sure that the new cohort has the same columns as the previous one
      if not set(new_cohort.columns).issubset(self._dfs[-1].columns):
        raise ValueError("new_cohort must have the same columns as the previous cohort. Only rows/indices should differ")
      
      # make sure that the new cohort is not bigger than the previous one; we are excluding!
      if len(new_cohort) > len(self._dfs[-1]):
        raise ValueError("new_cohort must have fewer or equal rows than the previous cohort")
      
      self._dfs.append(new_cohort)

    if exclusion_reason is not None:
      self.exclusion_labels[len(self._dfs) - 1] = exclusion_reason
    else:
      self.exclusion_labels[len(self._dfs) - 1] = f'Exclusion {len(self._dfs) - 1}'

    if new_cohort_label is not None:
      self.new_cohort_labels[len(self._dfs) - 1] = new_cohort_label
    else:
      self.new_cohort_labels[len(self._dfs) - 1] = f'Cohort {len(self._dfs) - 1}'
       

  def view_table_flows(self, 
                       label_suffix: Optional[bool] = None,
                       thousands_sep: Optional[bool] = None) -> pd.DataFrame:
    
    
    if len(self._dfs) < 2:
      raise ValueError("At least two cohorts must be provided. Please use add_exclusion() to add exclusions.")
    
    if label_suffix is None:
      label_suffix = self.label_suffix 

    if thousands_sep is None:
      thousands_sep = self.thousands_sep

    
    self.table_flows = TableFlows(
       dfs=self._dfs,
       label_suffix=label_suffix,
       thousands_sep=thousands_sep,
    )

    return self.table_flows.view()

  def view_table_characteristics(self,
                                 categorical: Optional[list] = None,
                                 normal: Optional[list] = None,
                                 nonnormal: Optional[list] = None,
                                 decimals: Optional[int] = None,
                                 format_cat: Optional[str] = None,
                                 format_normal: Optional[str] = None,
                                 format_nonnormal: Optional[str] = None,
                                 thousands_sep: Optional[bool] = None,
                                 missingness: Optional[bool] = None,
                                 label_suffix: Optional[bool] = None,
                                 rename: Optional[dict] = None) -> pd.DataFrame:
    
    if len(self._dfs) < 2:
      raise ValueError("At least two cohorts must be provided. Please use add_exclusion() to add exclusions")
        
    if categorical is None:
      categorical = self.categorical

    if normal is None:
      normal = self.normal

    if nonnormal is None:
      nonnormal = self.nonnormal

    if decimals is None:
      decimals = self.decimals

    if format_cat is None:
      format_cat = self.format_cat

    if format_normal is None:
      format_normal = self.format_normal

    if format_nonnormal is None:
      format_nonnormal = self.format_nonnormal

    if thousands_sep is None:
      thousands_sep = self.thousands_sep

    if missingness is None:
      missingness = self.missingness

    if label_suffix is None:
      label_suffix = self.label_suffix

    if rename is None:
      rename = self.rename

    self.table_characteristics = TableCharacteristics(
      dfs=self._dfs,
      categorical=categorical,
      normal=normal,
      nonnormal=nonnormal,
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

  def view_table_drifts(self,
                        drifts_by_class: Optional[bool] = False,
                        categorical: Optional[list] = None,
                        normal: Optional[list] = None,
                        nonnormal: Optional[list] = None,
                        decimals: Optional[int] = None,
                        missingness: Optional[bool] = None,
                        rename: Optional[dict] = None) -> pd.DataFrame:
    
    if len(self._dfs) < 2:
      raise ValueError("At least two cohorts must be provided. Please use add_exclusion() to add exclusions")

    if categorical is None:
      categorical = self.categorical

    if normal is None:
      normal = self.normal

    if nonnormal is None:
      nonnormal = self.nonnormal

    if decimals is None:
      decimals = self.decimals

    if missingness is None:
      missingness = self.missingness

    if rename is None:
      rename = self.rename

    self.table_drifts = TableDrifts(
      dfs=self._dfs,
      categorical=categorical,
      normal=normal,
      nonnormal=nonnormal,
      decimals=decimals,
      missingness=missingness,
      rename=rename,
    )

    if drifts_by_class:
      return self.table_drifts.view_simple()
    
    else:
       return self.table_drifts.view()
  

  def plot_flows(self,
                 new_cohort_labels: Optional[list] = None,
                 exclusion_labels: Optional[list] = None,
                 box_width: Optional[int] = 2.5,
                 box_height: Optional[int] = 1,
                 plot_dists: Optional[bool] = True,
                 smds: Optional[bool] = True,
                 legend: Optional[bool] = True,
                 legend_with_vars: Optional[bool] = True,
                 output_folder: Optional[str] = 'imgs',
                 output_file: Optional[str] = 'flow_diagram',
                 display_flow_diagram: Optional[bool] = True,
                 ) -> None:
    
    if len(self._dfs) < 2: 
      raise ValueError("At least two cohorts must be provided. Please use add_exclusion() to add exclusions")
       
    
    if new_cohort_labels is None:
      new_cohort_labels = self.new_cohort_labels.values()
      new_cohort_labels = ["___ patients\n" + label for label in new_cohort_labels]

    if exclusion_labels is None:
      exclusion_labels = self.exclusion_labels.values()
      exclusion_labels = ["___ patients excluded for\n" + label for label in exclusion_labels]

    plot_table_flows = TableFlows(
      dfs=self._dfs,
      label_suffix=True,
      thousands_sep=True,
    )

    plot_table_characteristics = TableCharacteristics(
      dfs=self._dfs,
      categorical=self.categorical,
      normal=self.normal,
      nonnormal=self.nonnormal,
      decimals=self.decimals,
      format_cat='%',
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
      decimals=self.decimals,
      missingness=self.missingness,
      rename=self.rename,
    )
     
    self.flow_diagram = FlowDiagram(
      table_flows=plot_table_flows,
      table_characteristics=plot_table_characteristics,
      table_drifts=plot_table_drifts,
      new_cohort_labels=new_cohort_labels,
      exclusion_labels=exclusion_labels,
      box_width=box_width,
      box_height=box_height,
      plot_dists=plot_dists,
      smds=smds,
      legend=legend,
      legend_with_vars=legend_with_vars,
      output_folder=output_folder,
      output_file=output_file,
      display_flow_diagram=display_flow_diagram,
    )

    self.flow_diagram.view()

  def view_table_pvalues(self,
                      categorical: Optional[list] = None,
                      normal: Optional[list] = None,
                      nonnormal: Optional[list] = None,
                      alpha: Optional[float] = 0.05,
                      decimals: Optional[int] = 3,
                      min_n_expected: Optional[int] = 5,
                      min_samples: Optional[int] = 30,
                      rename: Optional[dict] = None) -> pd.DataFrame:
    """
    Generate a table of p-values between consecutive cohorts using appropriate statistical tests.
    
    Parameters
    ----------
    categorical : list, optional
        List of categorical variable names
    normal : list, optional
        List of normally distributed continuous variable names
    nonnormal : list, optional
        List of non-normally distributed continuous variable names
    alpha : float, optional
        Significance level (default: 0.05)
    decimals : int, optional
        Number of decimal places for p-values (default: 3)
    min_n_expected : int, optional
        Minimum expected count for Chi-square validity (default: 5)
    min_samples : int, optional
        Threshold for small sample considerations (default: 30)
    rename : dict, optional
        Dictionary mapping original variable names to display names
        
    Returns
    -------
    pd.DataFrame
        Table of p-values between consecutive cohorts
    
    Notes
    -----
    Statistical tests used:
    - Categorical variables: Chi-squared test (or Fisher's exact for small samples)
    - Normal continuous variables: Two-sample t-test (or One-Way ANOVA for small samples)
    - Non-normal continuous variables: Kruskal-Wallis test
    
    P-values are marked with:
    - * for p < 0.05
    - ** for p < 0.01
    - *** for p < 0.001
    - † for tests with small sample considerations
    """
    if len(self._dfs) < 2:
        raise ValueError("At least two cohorts must be provided. Please use add_exclusion() to add exclusions")
    
    if categorical is None:
        categorical = self.categorical
    
    if normal is None:
        normal = self.normal
    
    if nonnormal is None:
        nonnormal = self.nonnormal
    
    if decimals is None:
        decimals = self.decimals
    
    if rename is None:
        rename = self.rename
    
    self.table_pvalues = TablePValues(
        dfs=self._dfs,
        categorical=categorical,
        normal=normal,
        nonnormal=nonnormal,
        alpha=alpha,
        decimals=decimals,
        min_n_expected=min_n_expected,
        min_samples=min_samples,
        rename=rename,
    )
    
    return self.table_pvalues.view()


class TableFlows:
  def __init__(
        self,
        dfs: list,
        label_suffix: Optional[bool] = True,
        thousands_sep: Optional[bool] = True,
    ) -> None:

    if not isinstance(dfs, list) or len(dfs) < 2:
      raise ValueError("dfs must be a list with length ≥ 2")
    
    if not isinstance(label_suffix, bool):
      raise ValueError("label_suffix must be a boolean")
    
    self._dfs = dfs
    self._label_suffix = label_suffix
    self._thousands_sep = thousands_sep


  def view(self):

    table = pd.DataFrame(columns=['Cohort Flow', '', 'N',])
    rows = []

    for i in range(len(self._dfs) - 1):

      df_0 = self._dfs[i]
      df_1 = self._dfs[i+1]
      label = f"{i} to {i+1}"

      if self._label_suffix:
        suffix = ', n'

      else:
        suffix = ''

      if self._thousands_sep:
        n0_string = f"{len(df_0):,}"
        n1_string = f"{len(df_0) - len(df_1):,}"
        n2_string = f"{len(df_1):,}"

      else:
        n0_string = len(df_0)
        n1_string = len(df_0) - len(df_1)
        n2_string = len(df_1)


      rows.append({'Cohort Flow': label,
                   '': 'Initial' + suffix,
                   'N': n0_string})
      
      rows.append({'Cohort Flow': label,
                   '': 'Removed' + suffix,
                   'N': n1_string})
      
      rows.append({'Cohort Flow': label,
                   '': 'Result' + suffix,
                   'N': n2_string})

    table = pd.DataFrame(rows)

    table = table.pivot(index='', columns='Cohort Flow', values='N')

    return table
  

class TableCharacteristics:
  def __init__(
      self,
      dfs: list,
      categorical: Optional[list] = None,
      normal: Optional[list] = None,
      nonnormal: Optional[list] = None,
      decimals: Optional[int] = 1,
      format_cat: Optional[str] = 'N (%)',
      format_normal: Optional[str] = 'Mean ± SD',
      format_nonnormal: Optional[str] = 'Median [IQR]',
      thousands_sep: Optional[bool] = True,
      missingness: Optional[bool] = True,
      label_suffix: Optional[bool] = True,
      rename: Optional[dict] = None,
  ) -> None:
        
    if not isinstance(dfs, list) or len(dfs) < 2:
        raise ValueError("dfs must be a list with length ≥ 2")
    
    if (categorical is None) & (normal is None) & (nonnormal is None):
        raise ValueError("At least one of categorical, normal, or nonnormal must be provided")
       
    if (categorical is not None) & (not isinstance(categorical, list)):
        raise ValueError("categorical must be a list")

    if (normal is not None) & (not isinstance(normal, list)):
        raise ValueError("normal must be a list")
    
    if (nonnormal is not None) & (not isinstance(nonnormal, list)):
        raise ValueError("nonnormal must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if format_cat not in ['%', 'N', 'N (%)']:
        raise ValueError("format must be '%', 'N', or 'N (%)'")
    
    if format_normal not in ['Mean ± SD', 'Mean', 'SD']:
        raise ValueError("format must be 'Mean ± SD' or 'Mean' or 'SD'")
    
    if format_nonnormal not in ['Median [IQR]', 'Mean', 'SD']:
        raise ValueError("format must be 'Median [IQR]' or 'Mean' or 'SD'")
    
    if not isinstance(thousands_sep, bool):
        raise ValueError("thousands_sep must be a boolean")
    
    if not isinstance(missingness, bool):
        raise ValueError("missingness must be a boolean")
    
    if not isinstance(label_suffix, bool):
        raise ValueError("label_suffix must be a boolean")
    
    if (rename is not None) & (not isinstance(rename, dict)):
      raise ValueError("rename must be a dictionary")
    
    self._dfs = dfs

    if categorical is None:
      self._categorical = []
    else:
       self._categorical = categorical
      
    if normal is None:
      self._normal = []
    else:
      self._normal = normal
    
    if nonnormal is None:
      self._nonnormal = []
    else:
      self._nonnormal = nonnormal

    self._decimals = decimals
    self._missingness = missingness
    self._format_cat = format_cat
    self._format_normal = format_normal
    self._format_nonnormal = format_nonnormal
    self._thousands_sep = thousands_sep
    self._label_suffix = label_suffix
    
    if rename is not None:
      self._rename = rename
    else:
       self._rename = dict()

    if rename is not None:
      if self._label_suffix:
          self._renamed_categorical = [
             self._rename[c] + ', ' + self._format_cat if c in self._rename.keys() \
              else c + ', ' + self._format_cat for c in self._categorical
          ]
          
          self._renamed_normal = [
              self._rename[n] + ', ' + self._format_normal if n in self._rename.keys() \
              else n + ', ' + self._format_normal for n in self._normal
          ]

          self._renamed_nonnormal = [
              self._rename[nn] + ', ' + self._format_nonnormal if nn in self._rename.keys() \
              else nn + ', ' + self._format_nonnormal for nn in self._nonnormal
          ]


      else:
        self._renamed_categorical = [
            self._rename[c] if c in self._rename.keys() else c for c in self._categorical
        ]

        self._renamed_normal = [
            self._rename[n] if n in self._rename.keys() else n for n in self._normal
        ]

        self._renamed_nonnormal = [
            self._rename[nn] if nn in self._rename.keys() else nn for nn in self._nonnormal
        ]

    else:
      if self._label_suffix:
        self._renamed_categorical = [c + ', ' + self._format_cat for c in self._categorical]
        self._renamed_normal = [n + ', ' + self._format_normal for n in self._normal]
        self._renamed_nonnormal = [nn + ', ' + self._format_nonnormal for nn in self._nonnormal]
      else:
        self._renamed_categorical = self._categorical
        self._renamed_normal = self._normal
        self._renamed_nonnormal = self._nonnormal


  # method to get the unique values, before any exclusion (at i=0)
  def _get_original_uniques(self, cols):

    original_uniques = dict()

    # get uniques values ignoring NaNs
    for c in cols:
      original_uniques[c] = self._dfs[0][c].dropna().unique()

    return original_uniques


  # method to get the value counts for a given column
  def _my_value_counts(self,
                       df: pd.DataFrame(),
                       original_uniques: dict,
                       col: str,
                      ) -> pd.DataFrame(): # type: ignore

    o_uniques = original_uniques[col]
    counts = pd.DataFrame(columns=[col], index=o_uniques)

    # get the number of observations, based on whether we want to include missingness
    if self._missingness:
      n = len(df)
    else:
      n = len(df) - df[col].isnull().sum() # denominator will be the number of non-missing observations

    for o in o_uniques:
      if self._format_cat == '%':
        counts.loc[o,col] = ((df[col] == o).sum() / n * 100).round(self._decimals)
  
      elif self._format_cat == 'N':
        if self._thousands_sep:
          counts.loc[o,col] = f"{(df[col] == o).sum():,}"
        else:
          counts.loc[o,col] = (df[col] == o).sum()
   
      elif self._format_cat == 'N (%)':
        n_counts = (df[col] == o).sum()
        perc_counts = (n_counts / n * 100).round(self._decimals)
        if self._thousands_sep:
          counts.loc[o,col] = f"{n_counts:,} ({perc_counts})"
        else:
          counts.loc[o,col] = f"{n_counts} ({perc_counts})"

      else:
        raise ValueError("format must be '%', 'N', or 'N (%)'")

    return counts 
  
  # method to report distribution of normal variables
  def _normal_vars_dist(self,
                        df: pd.DataFrame(),
                        col: str,
                        df_dists: pd.DataFrame(),
                        ) -> pd.DataFrame():
    
    df.loc[:,col] = pd.to_numeric(df[col], errors='raise')
    
    if self._format_normal == 'Mean ± SD':
      col_mean = np.round(df[col].mean(), self._decimals)
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = f"{col_mean} ± {col_std}"

    elif self._format_normal == 'Mean':
      col_mean = np.round(df[col].mean(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_mean
    
    elif self._format_normal == 'SD':
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_std

    return df_dists
  
  def _nonnormal_vars_dist(self,
                           df: pd.DataFrame(),
                           col: str,
                           df_dists: pd.DataFrame(),
                          ) -> pd.DataFrame():
     
    df.loc[:,col] = pd.to_numeric(df[col], errors='raise')

    if self._format_nonnormal == 'Mean':
      col_mean = np.round(df[col].mean(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_mean

    elif self._format_nonnormal == 'Median [IQR]':
      col_median = np.round(df[col].median(), self._decimals)
      col_q1 = np.round(df[col].quantile(0.25), self._decimals)
      col_q3 = np.round(df[col].quantile(0.75), self._decimals)

      df_dists.loc[(col, ' '), 'value'] = f"{col_median} [{col_q1}, {col_q3}]"

    elif self._format_nonnormal == 'SD':
      col_std = np.round(df[col].std(), self._decimals)
      df_dists.loc[(col, ' '), 'value'] = col_std

    return df_dists
  

  # method to add missing counts to the table
  def _add_missing_counts(self,
                           df: pd.DataFrame(),
                           col: str,
                           df_dists: pd.DataFrame(),
                           ) -> pd.DataFrame(): # type: ignore

    n = len(df)

    if self._format_cat == '%':
      df_dists.loc[(col,'Missing'),'value'] = (df[col].isnull().sum() / n * 100).round(self._decimals)
    
    elif self._format_cat == 'N':
      if self._thousands_sep:
        df_dists.loc[(col,'Missing'),'value'] = f"{df[col].isnull().sum():,}"
      else:
        df_dists.loc[(col,'Missing'),'value'] = df[col].isnull().sum()

    elif self._format_cat == 'N (%)':
      n_missing = df[col].isnull().sum()
      perc_missing = df[col].isnull().sum() / n * 100
      if self._thousands_sep:
        df_dists.loc[(col,'Missing'),'value'] = f"{n_missing:,} ({(perc_missing).round(self._decimals)})"
      else: 
        df_dists.loc[(col,'Missing'),'value'] = f"{n_missing} ({(perc_missing).round(self._decimals)})"

    else:
      raise ValueError("format must be '%', 'N', or 'N (%)'")

    return df_dists
  
  
  # method to add overall counts to the table
  def _add_overall_counts(self,
                           df,
                           df_dists
                           ) -> pd.DataFrame(): # type: ignore

    if self._thousands_sep:
      df_dists.loc[('Overall', ' '), 'value'] = f"{len(df):,}"
    else:
      df_dists.loc[('Overall', ' '), 'value'] = len(df)


    return df_dists
  
  # method to add label_suffix to the table
  def _add_label_suffix(self,
                         col: str,
                         df_dists: pd.DataFrame(),
                         suffix: str,
                         ) -> pd.DataFrame(): # type: ignore

    new_col = col + suffix
    df_dists = df_dists.rename(index={col: new_col}) 

    return df_dists
  
  # method to rename columns
  def _rename_columns(self,
                       df_dists: pd.DataFrame(),
                       col: str,
                      ) -> pd.DataFrame():
    
    return self._rename[col], df_dists.rename(index={col: self._rename[col]})
  
  def view(self):

    table = pd.DataFrame()

    # get the unique values, before any exclusion, for categorical variables
    original_uniques = self._get_original_uniques(self._categorical)

    for i, df in enumerate(self._dfs):

      index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['variable', 'index'])
      df_dists = pd.DataFrame(columns=['value'], index=index)

      # get distribution for categorical variables
      for col in self._categorical:

        counts = self._my_value_counts(df, original_uniques, col)

        melted_counts = pd.melt(counts.reset_index(), id_vars=['index']) \
                          .set_index(['variable','index'])
        
        df_dists = pd.concat([df_dists, melted_counts], axis=0)

        if self._missingness:
          df_dists = self._add_missing_counts(df, col, df_dists)

        # rename if applicable
        if col in self._rename.keys():
          col, df_dists = self._rename_columns(df_dists, col)

        if self._label_suffix:
            df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_cat)
          

      # get distribution for normal variables
      for col in self._normal:
  
          df_dists = self._normal_vars_dist(df, col, df_dists)

          if self._missingness:
            df_dists = self._add_missing_counts(df, col, df_dists)

          if col in self._rename.keys():
            col, df_dists = self._rename_columns(df_dists, col)

          if self._label_suffix:
            df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_normal)
        
      # get distribution for nonnormal variables
      for col in self._nonnormal:

        df_dists = self._nonnormal_vars_dist(df, col, df_dists)

        if self._missingness:
          df_dists = self._add_missing_counts(df, col, df_dists)
        
        if col in self._rename.keys():
          col, df_dists = self._rename_columns(df_dists, col)

        if self._label_suffix:
          df_dists = self._add_label_suffix(col, df_dists, ', ' + self._format_nonnormal)


      df_dists = self._add_overall_counts(df, df_dists)
    
      df_dists.rename(columns={'value': i}, inplace=True)
      table = pd.concat([table, df_dists], axis=1)

    # add super header
    table = table.set_axis(
        pd.MultiIndex.from_product([['Cohort'], table.columns]),
        axis=1)

    # renames indexes
    table.index.names = ['Variable', 'Value']

    # reorder values of "Variable" (level 0) such that 'Overall' comes first
    table = table.sort_index(level=0, key=lambda x: x == 'Overall',
                             ascending=False, sort_remaining=False)

    return table
    


class TableDrifts():
  def __init__(
      self,
      dfs: list,
      categorical: Optional[list] = None,
      normal: Optional[list] = None,
      nonnormal: Optional[list] = None,
      decimals: Optional[int] = 1,
      missingness: Optional[bool] = True,
      rename: Optional[dict] = None,
  ) -> None:
    
    if not isinstance(dfs, list) or len(dfs) < 1:
        raise ValueError("dfs must be a list with length ≥ 1")
    
    if (categorical is None) & (normal is None) & (nonnormal is None):
        raise ValueError("At least one of categorical, normal, or nonnormal must be provided")
    
    if (categorical is not None) & (not isinstance(categorical, list)):
        raise ValueError("categorical must be a list")
    
    if (normal is not None) & (not isinstance(normal, list)):
        raise ValueError("normal must be a list")
    
    if (nonnormal is not None) & (not isinstance(nonnormal, list)):
        raise ValueError("nonnormal must be a list")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer")
    
    if (rename is not None) & (not isinstance(rename, dict)):
        raise ValueError("rename must be a dictionary")
    
    self._dfs = dfs
    if categorical is None:
      self._categorical = []
    else:
      self._categorical = categorical

    if normal is None:
      self._normal = []
    else:
      self._normal = normal

    if nonnormal is None:
      self._nonnormal = []
    else:
      self._nonnormal = nonnormal

    self._decimals = decimals
    self._missingness = missingness

    if rename is None:
      self._rename = dict()
    else:
      self._rename = rename

    # make rename have the same keys as the original variable names if no rename
    for c in self._categorical + self._normal + self._nonnormal:
      if c not in self._rename.keys():
        self._rename[c] = c

  
    self._table_flows = TableFlows(
      dfs,
      label_suffix=False,
      thousands_sep=False,
    ).view()

    self._table_characteristics = TableCharacteristics(
      dfs,
      categorical=self._categorical,
      normal=self._normal,
      nonnormal=self._nonnormal,
      decimals=self._decimals,
      missingness=False,
      format_cat='N',
      format_normal='Mean',
      format_nonnormal='Mean',
      thousands_sep=False,
      label_suffix=False,
      rename=self._rename,
    ).view()

    # auxiliary tables
    self._table_cat_n = TableCharacteristics(
      dfs,
      categorical=self._categorical,
      normal=self._normal,
      nonnormal=self._nonnormal,
      decimals=self._decimals,
      format_cat='N',
      format_normal='Mean',
      format_nonnormal='Mean',
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
      decimals=self._decimals,
      format_cat='%',
      format_normal='SD',
      format_nonnormal='SD',
      thousands_sep=False,
      missingness=self._missingness,
      label_suffix=False,
      rename=self._rename,
    ).view()



  def view(self):

    inverse_rename = {value: key for key, value in self._rename.items()}

    table = pd.DataFrame(index=self._table_characteristics.index,
                         columns=self._table_flows.columns)
    
    for i, index_name in enumerate(self._table_characteristics.index):
      for j, column_name in enumerate(self._table_flows.columns):
        # skip if index_name is 'Overall' or 'Missing'
        if (index_name[0] == 'Overall'): # | (index_name[1] == 'Missing'):
          table.iloc[i,j] = ''
          continue
        
        # use cat_smd for categorical variables
        if inverse_rename[index_name[0]] in self._categorical:
          cat_n_0 = self._table_cat_n.loc[index_name, :].iloc[j]
          cat_perc_0 = self._table_cat_perc.loc[index_name, :].iloc[j]
          cat_n_1 = self._table_cat_n.loc[index_name, :].iloc[j+1]
          cat_perc_1 = self._table_cat_perc.loc[index_name, :].iloc[j+1]
          table.iloc[i,j] = self._cat_smd(
             prop1=[cat_perc_0/100],
             prop2=[cat_perc_1/100],
             n1=cat_n_0,
             n2=cat_n_1,
             unbiased=True
          )
        
        # use cont_smd for continuous variables
        elif (inverse_rename[index_name[0]] in self._normal) | (inverse_rename[index_name[0]] in self._nonnormal):
          mean_0 = self._table_cat_n.loc[index_name, :].iloc[j]
          sd_0 = self._table_cat_perc.loc[index_name, :].iloc[j]
          mean_1 = self._table_cat_n.loc[index_name, :].iloc[j+1]
          sd_1 = self._table_cat_perc.loc[index_name, :].iloc[j+1]
          n_0 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j]
          n_1 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j+1]
          table.iloc[i,j] = self._cont_smd(
             mean1=mean_0,
             mean2=mean_1,
             sd1=sd_0,
             sd2=sd_1,
             n1=n_0,
             n2=n_1,
             unbiased=True
          )
          
    return table
  

  def view_simple(self):

    inverse_rename = {value: key for key, value in self._rename.items()}

    cols = self._table_characteristics.index.get_level_values(0).unique()

    # remove 'Overall' from cols
    cols = [c for c in cols if c != 'Overall']

    table = pd.DataFrame(index=cols,
                         columns=self._table_flows.columns)
    
    for i, index_name in enumerate(cols):
      for j, column_name in enumerate(self._table_flows.columns):
        # skip if index_name is 'Overall' or 'Missing'
        if (index_name == 'Overall'): # | (index_name[1] == 'Missing'):
          table.iloc[i,j] = ''
          continue

        # use cat_smd for categorical variables
        if inverse_rename[index_name] in self._categorical:
          cat_n_0 = self._table_cat_n.loc[index_name, :].iloc[:, j].to_list()
          cat_perc_0 = self._table_cat_perc.loc[index_name, :].iloc[:, j].to_list()
          cat_n_1 = self._table_cat_n.loc[index_name, :].iloc[:, j+1].to_list()
          cat_perc_1 = self._table_cat_perc.loc[index_name, :].iloc[:, j+1].to_list()
          table.iloc[i,j] = self._cat_smd(
             prop1=[c/100 for c in cat_perc_0],
             prop2=[c/100 for c in cat_perc_1],
             n1=cat_n_0,
             n2=cat_n_1,
             unbiased=False
          )

        # use cont_smd for continuous variables
        elif (inverse_rename[index_name] in self._normal) | (inverse_rename[index_name] in self._nonnormal):
          mean_0 = self._table_cat_n.loc[(index_name, ' '), :].iloc[j]
          sd_0 = self._table_cat_perc.loc[(index_name, ' '), :].iloc[j]
          mean_1 = self._table_cat_n.loc[(index_name, ' '), :].iloc[j+1]
          sd_1 = self._table_cat_perc.loc[(index_name, ' '), :].iloc[j+1]
          n_0 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j]
          n_1 = self._table_characteristics.loc[('Overall', ' '), :].iloc[j+1]
          table.iloc[i,j] = self._cont_smd(
             mean1=mean_0,
             mean2=mean_1,
             sd1=sd_0,
             sd2=sd_1,
             n1=n_0,
             n2=n_1,
             unbiased=False
          )
          
    return table

        

  # adapted from: https://github.com/tompollard/tableone/blob/main/tableone/tableone.py#L659
  def _cat_smd(self,
               prop1=None,
               prop2=None,
               n1=None,
               n2=None,
               unbiased=False):
      """
      Compute the standardized mean difference (regular or unbiased) using
      either raw data or summary measures.

      Parameters
      ----------
      prop1 : list
          Proportions (range 0-1) for each categorical value in dataset 1
          (control). 
      prop2 : list
          Proportions (range 0-1) for each categorical value in dataset 2
          (treatment).
      n1 : int
          Sample size of dataset 1 (control).
      n2 : int
          Sample size of dataset 2 (treatment).
      unbiased : bool
          Return an unbiased estimate using Hedges' correction. Correction
          factor approximated using the formula proposed in Hedges 2011.
          (default = False)

      Returns
      -------
      smd : float
          Estimated standardized mean difference.
      """
      # Categorical SMD Yang & Dalton 2012
      # https://support.sas.com/resources/papers/proceedings12/335-2012.pdf
      prop1 = np.asarray(prop1)
      prop2 = np.asarray(prop2)

      lst_cov = []
      for p in [prop1, prop2]:
          variance = p * (1 - p)
          covariance = - np.outer(p, p)  # type: ignore
          covariance[np.diag_indices_from(covariance)] = variance
          lst_cov.append(covariance)

      mean_diff = np.asarray(prop2 - prop1).reshape((1, -1))  # type: ignore
      mean_cov = (lst_cov[0] + lst_cov[1])/2

      try:
          sq_md = mean_diff @ np.linalg.inv(mean_cov) @ mean_diff.T
      except np.linalg.LinAlgError:
          sq_md = 0

      try:
          smd = np.asarray(np.sqrt(sq_md))[0][0]
      except IndexError:
          smd = 0

      # standard error
      # v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
      # se = np.sqrt(v_d)

      if unbiased:
          # Hedges correction (J. Hedges, 1981)
          # Approximation for the the correction factor from:
          # Introduction to Meta-Analysis. Michael Borenstein,
          # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
          # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
          j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
          smd = j * smd
          # v_g = (j ** 2) * v_d
          # se = np.sqrt(v_g)

      return np.round(smd, self._decimals)
  
    # adapted from: https://github.com/tompollard/tableone/blob/main/tableone/tableone.py#L581
  def _cont_smd(self,
                mean1=None, mean2=None,
                sd1=None, sd2=None,
                n1=None, n2=None,
                unbiased=False):
    """
    Compute the standardized mean difference (regular or unbiased) using
    either raw data or summary measures.

    Parameters
    ----------
    mean1 : float
        Mean of dataset 1 (control).
    mean2 : float
        Mean of dataset 2 (treatment).
    sd1 : float
        Standard deviation of dataset 1 (control).
    sd2 : float
        Standard deviation of dataset 2 (treatment).
    n1 : int
        Sample size of dataset 1 (control).
    n2 : int
        Sample size of dataset 2 (treatment).
    unbiased : bool
        Return an unbiased estimate using Hedges' correction. Correction
        factor approximated using the formula proposed in Hedges 2011.
        (default = False)

    Returns
    -------
    smd : float
        Estimated standardized mean difference.
    """

    # cohens_d
    denominator = np.sqrt((sd1 ** 2 + sd2 ** 2) / 2) 
    if denominator == 0:
       return 0
    else:
      smd = (mean2 - mean1) / denominator # type: ignore

    # standard error
    # v_d = ((n1+n2) / (n1*n2)) + ((smd ** 2) / (2*(n1+n2)))  # type: ignore
    # se = np.sqrt(v_d)

    if unbiased:
        # Hedges correction (J. Hedges, 1981)
        # Approximation for the the correction factor from:
        # Introduction to Meta-Analysis. Michael Borenstein,
        # L. V. Hedges, J. P. T. Higgins and H. R. Rothstein
        # Wiley (2011). Chapter 4. Effect Sizes Based on Means.
        j = 1 - (3/(4*(n1+n2-2)-1))  # type: ignore
        smd = j * smd
        # v_g = (j ** 2) * v_d
        # se = np.sqrt(v_g)

    return np.round(smd, self._decimals)


class FlowDiagram:
    def __init__(self,
                 table_flows: TableFlows,
                 table_characteristics: TableCharacteristics = None,
                 table_drifts: TableDrifts = None,
                 new_cohort_labels: list = None,
                 exclusion_labels: list = None,
                 box_width: int = 2.5,
                 box_height: int = 1,
                 plot_dists: bool = True,
                 smds: bool = True,
                 legend: bool = True,
                 legend_with_vars: bool = True,
                 output_folder: str = 'imgs',
                 output_file: str = 'flow_diagram',
                 display_flow_diagram: bool = True,
                 ):
        
        if (table_characteristics is None) & (plot_dists):
            raise ValueError("table_characteristics must be provided if plot_dists is True")
        
        if (table_drifts is None) & (smds):
            raise ValueError("table_drifts must be provided if smds is True")
        
        self.table_flows = table_flows.view()
        self.table_characteristics = table_characteristics
        self.table_drifts = table_drifts
    

        if new_cohort_labels is None:
            new_cohort_labels = [f'Cohort {i},\n ___ subjects' for i in range(len(table_flows.view().columns))]

        if exclusion_labels is None:
            exclusion_labels = [f'Exclusion {i},\n ___ subjects' for i in range(len(table_flows.view().columns))]

        self.cohort_labels = new_cohort_labels
        self.exclusion_labels = exclusion_labels
        self.width = box_width
        self.height = box_height
        self.plot_dists = plot_dists

        if self.plot_dists == False:
          self.smds = False
          self.legend = False
          self.legend_with_vars = False
           

        self.smds = smds
        self.legend = legend
        self.legend_with_vars = legend_with_vars
        self.output_file = output_file
        self.output_folder = output_folder

        # Create the imgs folder if it does not exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.output_path = os.path.join(self.output_folder, self.output_file)
        self.display = display_flow_diagram


    def _plot_dists(self):

        # Extract data from table_characteristics
        categorical = self.table_characteristics._renamed_categorical

        table = self.table_characteristics.view()
        if self.smds:
          table_smds = self.table_drifts.view_simple()

        vars = table.loc[
            table.index.get_level_values(0) != 'Overall'
        ].index.get_level_values(0).unique().tolist()

        cohorts = table.columns.get_level_values(1).unique().tolist()

        # Define the legend handles (empty for now)
        legend_handles = []
        legend_labels = []

        # Iterate through Cohort number
        for c, coh in enumerate(cohorts):
            fig, axes = plt.subplots(1, 1, figsize=(4, 2), dpi=150)
            
            # Iterate through variables
            for v, var in enumerate(vars):
                
                if var in categorical:
                    values_names = table.loc[
                        table.index.get_level_values(0) == var
                    ].index.get_level_values(1)

                    cum_width = 0
                    for val in values_names:
                        value = table.loc[(var, val), ('Cohort', coh)]
                        if val != 'Missing':
                            bar = axes.barh(v, value, left=cum_width, height=.8, edgecolor='white')
                            textcolor = 'white'
                            if coh == 0:
                                legend_handles.append(bar[0])
                                if self.legend_with_vars:
                                    legend_labels.append(f"{var}: {val}")
                                else:
                                    legend_labels.append(val)
                        else:
                            bar = axes.barh(v, value, left=cum_width, height=.8, color='lightgray',
                                        hatch='///////', edgecolor='white')
                            textcolor = 'black'
                            if (coh == 0) & ('Missing' not in legend_labels):
                                legend_handles.append(bar[0])
                                legend_labels.append('Missing')

                        if value > 5:
                            axes.text(cum_width + value/2, v, '{:.1f}'.format(value),
                                    ha='center', va='center', color=textcolor, fontsize=8)
                        cum_width += value
                        
                    if (coh > 0) & (self.smds):
                        col_name = f"{coh-1} to {coh}"
                        var_smd = var.split(',')[0]
                        smd = table_smds.loc[var_smd, (col_name)]
                        axes.text(-1, v, f'{smd}', ha='right', va='center', fontsize=8, color='black', fontweight='normal')

                else:
                    val = ' '
                    value = table.loc[(var, val), ('Cohort', coh)]
                    axes.barh(v, 100, left=0, height=.8, color='lavender', edgecolor='white')
                    axes.text(50, v, f"{value}", ha='center', va='center', color='black', fontsize=8)
                    
                    if (coh > 0) & (val != 'Missing') & (self.smds):
                        col_name = f"{coh-1} to {coh}"
                        var_smd = var.split(',')[0]
                        smd = table_smds.loc[(var_smd), (col_name)]
                        axes.text(-1, v, f'{smd}', ha='right', va='center', fontsize=8, color='black', fontweight='normal')
                
                axes.text(101, v, var, ha='left', va='center', fontsize=8, color='black', fontweight='normal')


            if self.smds:
                if coh > 0:
                    color_smd = 'black'
                    text_smd = f'SMD ({coh-1}, {coh})'
                elif coh == 0:
                    color_smd = 'white'
                    text_smd = f'SMD (0, 0)'
                
                axes.text(-1, v + .75, text_smd, ha='right', va='center', fontsize=8, color=color_smd, fontweight='bold')

            axes.set_yticks([])
            axes.set_xticks([])
            for spine in axes.spines.values():
                spine.set_visible(False)

            if coh == 0:
                # move 'Missing' to the end of the legend
                missing_idx = legend_labels.index('Missing')
                legend_labels.append(legend_labels.pop(missing_idx))
                legend_handles.append(legend_handles.pop(missing_idx))

                # create a separate figure for the legend
                legend_fig, legend_ax = plt.subplots(figsize=(len(legend_labels)/4, 1))  # Adjust figsize as necessary
                legend_ax.axis('off')
                fig_legend = legend_ax.legend(legend_handles, legend_labels, loc='center', ncol=1,
                                            fontsize=8, frameon=False)

                # save the legend figure
                legend_fig.savefig('imgs/legend.svg', dpi=300, bbox_inches='tight')

                # close the legend figure
                plt.close(legend_fig)

            plt.savefig(f'imgs/part{c}.svg', dpi=300, bbox_inches='tight')
            plt.close()


    def view(self):

        # generate all auxiliary plots
        if self.plot_dists:
          self._plot_dists()
        
        dot = graphviz.Digraph(
            comment='Cohort Exclusion Process',
            format='svg',
            graph_attr={'fontname': 'Helvetica', 'splines': 'ortho'},
            node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': 'white', 'fixedsize': 'true',
                       'width': str(self.width), 'height': str(self.height), 'fontname': 'Helvetica'},  
            edge_attr={'dir': 'forward', 'arrowhead': 'vee', 'arrowsize': '0.5', 'minlen': '1'},
        )

        columns = self.table_flows.columns.tolist()
        num_columns = len(columns)

        # Add main cohort nodes with initial counts
        initial_counts = self.table_flows.loc['Initial, n']
        for i, (count, column) in enumerate(zip(initial_counts, columns)):
            node_label = self.cohort_labels[i].replace('___', f'{count}')
            dot.node(f'A{i}', node_label, shape='box', fontname='Helvetica')

        # Add final cohort node
        final_node_label = self.cohort_labels[-1]
        final_node_label = final_node_label.replace('___', f"{self.table_flows.loc['Result, n'].iloc[-1]}")
        dot.node(f'A{num_columns}', final_node_label, shape='box', fontname='Helvetica')

        if self.plot_dists:
          # Add final distribution plot node
          dot.node(f'plot_dist{num_columns}', label='',  image=f'part{num_columns}.svg',
                  imagepos='bc',  imagescale='true',
                  shape='box', color='transparent',
                  width=str(self.width+0.5),
                  height=str(self.height+0.2))

          with dot.subgraph() as s:
              s.attr(rank='same')
              s.node(f'A{num_columns}')
              s.node(f'plot_dist{num_columns}')

        # Add exclusion criteria nodes with removed counts
        removed_counts = self.table_flows.loc['Removed, n']
        for i, (count, column) in enumerate(zip(removed_counts, columns)):
            node_label = self.exclusion_labels[i].replace('___', f'{count}')
            dot.node(f'E{i}', node_label, shape='box', style='filled', fillcolor='floralwhite')

        # Add invisible nodes for positioning
        for i in range(num_columns + 1):
            dot.node(f'IA{i}', '', shape='point', height='0')

        # connect the main cohort nodes
        for i in range(num_columns):
            dot.edge(f'A{i}', f'IA{i}', arrowhead='none')
            dot.edge(f'IA{i}', f'A{i+1}', )

        # connect the exclusion nodes to the hidden nodes
        for i in range(num_columns):
            dot.edge(f'IA{i}', f'E{i}', constraint='false')
        
        # Adjust ranks to position nodes horizontally for exclusions
        for i in range(num_columns):
            with dot.subgraph() as s:
                s.attr(rank='same')
                s.node(f'IA{i}')
                s.node(f'E{i}')

        if self.plot_dists:
          # Add boxes for the distributions``
          for i in range(num_columns):
              dot.node(f'plot_dist{i}', label='', image=f'part{i}.svg',
                  imagepos='bc', imagescale='true',
                  shape='box', color='transparent',
                  width=str(self.width+0.75),
                  height=str(self.height+0.2))
              dot.edge(f'A{i}', f'plot_dist{i}', constraint='false', style='invis')
              with dot.subgraph() as s:
                  s.attr(rank='same')
                  s.node(f'A{i}')
                  s.node(f'plot_dist{i}')

        if self.legend:
          # Add a final node for the legend
          dot.node('legend', label='', image=f'legend.svg', imagescale='true',
                  shape='box', color='transparent',
                  imagepos='bl',
                  width=str(self.width),
                  height=str(self.height+0.2))

          # Connect the final cohort node to the legend from the first exclusion edge
          dot.edge(f'E0', 'legend', style='invis')
          with dot.subgraph() as s:
              s.attr(rank='same')
              s.node(f'E0')
              s.node('legend')
        
        # Save and render the graph
        dot.render(self.output_path, view=self.display, format='pdf')


from typing import Optional, Union, List, Dict, Any, Tuple, Callable
import pandas as pd
import numpy as np
import os
import inspect
import re

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
        from scipy import stats
        
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
                
            # For numeric columns, check normality (sample up to 5000 values)
            if self.data[col].dtype.kind in 'ifu':
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
                    else:  # Non-normal distribution
                        self._nonnormal_vars.append(col)
                except:
                    # If test fails, assume non-normal
                    self._nonnormal_vars.append(col)
        
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
    
    def exclude(self, mask, label: Optional[str] = None):
        """
        Add an exclusion step based on a boolean mask.
        
        Parameters:
        -----------
        mask : pandas.Series
            Boolean mask to select the subset of data to keep
        label : str, optional
            Label describing the exclusion. If not provided, tries to generate one.
            
        Returns:
        --------
        self : EasyFlow
            For method chaining
        """
        # Generate label if not provided
        if label is None:
            label = self._generate_exclusion_label(mask)
        
        # Apply the mask to the current data
        new_data = self._current_data[mask]
        
        # Store the step information
        self._exclusion_steps.append({
            'previous_data': self._current_data,
            'mask': mask,
            'new_data': new_data,
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
        
        # Add all exclusion steps
        for i, step in enumerate(self._exclusion_steps):
            # For the first step, pass the mask directly
            if i == 0:
                ef.add_exclusion(
                    mask=step['mask'],
                    exclusion_reason=step['label'],
                    new_cohort_label=f"Step {i+1}"
                )
            else:
                # For subsequent steps, we need to apply the mask to the current data
                # since the original mask was for a different dataframe
                indices = step['previous_data'][step['mask']].index
                next_mask = self.data.index.isin(indices)
                ef.add_exclusion(
                    mask=next_mask,
                    exclusion_reason=step['label'],
                    new_cohort_label=f"Step {i+1}"
                )
        
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
                  exclusions: List[Tuple[pd.Series, str]], 
                  output: str = "flow_diagram",
                  auto_detect: bool = True):
        """
        Quickly create a flow diagram with a list of exclusion steps.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Initial cohort data
        exclusions : List[Tuple[pd.Series, str]]
            List of (mask, label) pairs defining exclusions
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
        for mask, label in exclusions:
            flow.exclude(mask, label)
        
        # Generate the diagram
        flow.generate(output=output)
        
        return flow

class TablePValues:
    def __init__(
        self,
        dfs: list,
        categorical: Optional[list] = None,
        normal: Optional[list] = None,
        nonnormal: Optional[list] = None,
        alpha: float = 0.05,
        decimals: int = 3,
        min_n_expected: int = 5,
        min_samples: int = 30,
        rename: Optional[dict] = None
    ) -> None:
        """
        Calculate p-values between cohorts using appropriate statistical tests.
        """
        from scipy import stats
        
        if not isinstance(dfs, list) or len(dfs) < 2:
            raise ValueError("dfs must be a list with length ≥ 2")
        
        if (categorical is None) and (normal is None) and (nonnormal is None):
            raise ValueError("At least one of categorical, normal, or nonnormal must be provided")
        
        self._dfs = dfs
        self._categorical = [] if categorical is None else categorical
        self._normal = [] if normal is None else normal
        self._nonnormal = [] if nonnormal is None else nonnormal
        self._alpha = alpha
        self._decimals = decimals
        self._min_n_expected = min_n_expected
        self._min_samples = min_samples
        self._rename = {} if rename is None else rename
        
        # Create TableFlows object and get the DataFrame
        self._table_flows = TableFlows(
            dfs=self._dfs,
            label_suffix=False,
            thousands_sep=False,
        )
        
        # Make rename dict have the same keys as the original variable names if no rename
        for c in self._categorical + self._normal + self._nonnormal:
            if c not in self._rename.keys():
                self._rename[c] = c
    
    def _chi2_test(self, df1, df2, var):
        """Perform Chi-squared test for categorical variables."""
        from scipy import stats
        import numpy as np
        import pandas as pd
        
        # Check if we have enough non-missing values
        if df1[var].dropna().empty or df2[var].dropna().empty:
            return 1.0, False  # Return p=1.0 (no difference) if data is missing
        
        # Get all unique categories across both cohorts
        all_values = pd.concat([df1[var].dropna(), df2[var].dropna()]).unique()
        
        if len(all_values) < 2:
            return 1.0, False  # Return p=1.0 if there's only one category
        
        # Create contingency table
        table = np.zeros((2, len(all_values)))
        
        for i, val in enumerate(all_values):
            table[0, i] = (df1[var] == val).sum()
            table[1, i] = (df2[var] == val).sum()
        
        # Check if the table is valid for chi-square
        if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
            return 1.0, False  # Return p=1.0 if any row or column sums to zero
        
        try:
            # Check if expected counts are sufficient for Chi-square
            chi2_valid = True
            _, p, _, expected = stats.chi2_contingency(table, correction=True)
            
            if np.any(expected < self._min_n_expected):
                chi2_valid = False
                # Use Fisher's exact test for 2x2 tables
                if len(all_values) == 2:
                    try:
                        _, p = stats.fisher_exact(table)
                    except:
                        p = 1.0  # Default to no difference if test fails
                # For larger tables with small expected counts, chi-square is best we can do
            
            return p, chi2_valid
        except Exception as e:
            print(f"Error in chi2_test for {var}: {e}")
            return 1.0, False
    
    def _t_test(self, df1, df2, var):
        """Perform two-sample t-test for normally distributed variables."""
        from scipy import stats
        import numpy as np
        import pandas as pd
        
        try:
            # Convert data to numeric, coercing non-numeric values to NaN
            vals1 = pd.to_numeric(df1[var], errors='coerce').dropna()
            vals2 = pd.to_numeric(df2[var], errors='coerce').dropna()
            
            # Check if we have enough data
            if len(vals1) < 2 or len(vals2) < 2:
                return 1.0, False  # Return p=1.0 if not enough data
            
            # Check sample sizes
            small_sample = len(vals1) < self._min_samples or len(vals2) < self._min_samples
            
            if small_sample:
                # For small samples, use one-way ANOVA (equivalent to t-test for 2 groups)
                f_stat, p = stats.f_oneway(vals1, vals2)
            else:
                # For larger samples, use two-sample t-test with Welch's correction
                t_stat, p = stats.ttest_ind(vals1, vals2, equal_var=False)
            
            # Handle NaN p-value
            if np.isnan(p):
                return 1.0, False
                
            return p, small_sample
        except Exception as e:
            print(f"Error in t_test for {var}: {e}")
            return 1.0, False
    
    def _kruskal_test(self, df1, df2, var):
        """Perform Kruskal-Wallis test for non-normally distributed variables."""
        from scipy import stats
        import pandas as pd
        import numpy as np
        
        try:
            # Convert data to numeric, coercing non-numeric values to NaN
            vals1 = pd.to_numeric(df1[var], errors='coerce').dropna()
            vals2 = pd.to_numeric(df2[var], errors='coerce').dropna()
            
            # Check if we have enough data
            if len(vals1) < 2 or len(vals2) < 2:
                return 1.0, False
            
            # Kruskal-Wallis H-test (non-parametric test)
            h_stat, p = stats.kruskal(vals1, vals2)
            
            # Handle NaN p-value
            if np.isnan(p):
                return 1.0, False
                
            return p, True
        except Exception as e:
            print(f"Error in kruskal_test for {var}: {e}")
            return 1.0, False
    
    def _missingness_test(self, df1, df2, var):
        """Test difference in missingness proportion between cohorts using custom Z-test."""
        import numpy as np
        from scipy import stats
        
        try:
            # Calculate missing proportions
            n1 = len(df1)
            n2 = len(df2)
            
            if n1 == 0 or n2 == 0:
                return 1.0, False
            
            p1 = df1[var].isna().mean()
            p2 = df2[var].isna().mean()
            
            # If proportions are identical, no need for test
            if p1 == p2:
                return 1.0, True
            
            # Calculate pooled proportion
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            
            # Calculate standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            # If standard error is 0, avoid division by zero
            if se == 0:
                return 1.0, False
            
            # Calculate Z statistic
            z = (p1 - p2) / se
            
            # Calculate two-tailed p-value using normal distribution
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            return p_value, True
        except Exception as e:
            print(f"Error in missingness_test for {var}: {e}")
            return 1.0, False
    
    def view(self):
        """Generate a table of p-values between consecutive cohorts."""
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        # Get table flows as DataFrame
        table_flows_df = self._table_flows.view()
        cols = table_flows_df.columns
        
        # Create index for result table
        index_tuples = [('Overall', ' ')]
        
        # Add rows for main variables
        for var in self._categorical + self._normal + self._nonnormal:
            var_name = self._rename[var]
            index_tuples.append((var_name, ' '))
        
        # Add rows for missingness
        for var in self._categorical + self._normal + self._nonnormal:
            var_name = self._rename[var]
            index_tuples.append((var_name, 'Missing'))
        
        # Create index and empty DataFrame
        index = pd.MultiIndex.from_tuples(index_tuples, names=['Variable', 'Value'])
        table = pd.DataFrame('', index=index, columns=cols)
        
        # Fill in p-values between consecutive cohorts
        for col in cols:
            # Extract cohort indices from column name (e.g., "0 to 1" -> 0, 1)
            try:
                from_cohort, to_cohort = map(int, col.split(' to '))
                df1 = self._dfs[from_cohort]
                df2 = self._dfs[to_cohort]
                
                # Calculate p-values for categorical variables
                for var in self._categorical:
                    try:
                        # Main variable test
                        p_value, valid = self._chi2_test(df1, df2, var)
                        table.loc[(self._rename[var], ' '), col] = self._format_p_value(p_value, valid)
                        
                        # Missingness test
                        miss_p, miss_valid = self._missingness_test(df1, df2, var)
                        table.loc[(self._rename[var], 'Missing'), col] = self._format_p_value(miss_p, miss_valid)
                    except Exception as e:
                        print(f"Error processing {var}: {e}")
                        table.loc[(self._rename[var], ' '), col] = 'ERROR'
                        table.loc[(self._rename[var], 'Missing'), col] = 'ERROR'
                
                # Calculate p-values for normal variables
                for var in self._normal:
                    try:
                        # Main variable test
                        p_value, valid = self._t_test(df1, df2, var)
                        table.loc[(self._rename[var], ' '), col] = self._format_p_value(p_value, valid)
                        
                        # Missingness test
                        miss_p, miss_valid = self._missingness_test(df1, df2, var)
                        table.loc[(self._rename[var], 'Missing'), col] = self._format_p_value(miss_p, miss_valid)
                    except Exception as e:
                        print(f"Error processing {var}: {e}")
                        table.loc[(self._rename[var], ' '), col] = 'ERROR'
                        table.loc[(self._rename[var], 'Missing'), col] = 'ERROR'
                
                # Calculate p-values for non-normal variables
                for var in self._nonnormal:
                    try:
                        # Main variable test
                        p_value, valid = self._kruskal_test(df1, df2, var)
                        table.loc[(self._rename[var], ' '), col] = self._format_p_value(p_value, valid)
                        
                        # Missingness test
                        miss_p, miss_valid = self._missingness_test(df1, df2, var)
                        table.loc[(self._rename[var], 'Missing'), col] = self._format_p_value(miss_p, miss_valid)
                    except Exception as e:
                        print(f"Error processing {var}: {e}")
                        table.loc[(self._rename[var], ' '), col] = 'ERROR'
                        table.loc[(self._rename[var], 'Missing'), col] = 'ERROR'
                        
            except Exception as e:
                print(f"Error processing column {col}: {e}")
        
        # Remove rows with all empty values
        table = table.loc[~(table == '').all(axis=1)]
        
        # Add super header
        table = table.set_axis(
            pd.MultiIndex.from_product([['p-value'], table.columns]),
            axis=1
        )
        
        # Sort index with 'Overall' first
        table = table.sort_index(
            level=0, 
            key=lambda x: x == 'Overall',
            ascending=False, 
            sort_remaining=False
        )
        
        return table
    
    def _format_p_value(self, p_value, valid=True):
        """Format p-value with appropriate symbols and significance indicators."""
        import numpy as np
        
        # If p_value is nan or none, return empty string
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            return ''
        
        if not valid:
            prefix = "†"  # Mark small sample sizes or invalid tests
        else:
            prefix = ""
        
        # Format the p-value
        if p_value < 0.001:
            return f"{prefix}<0.001***"
        elif p_value < 0.01:
            return f"{prefix}{p_value:.{self._decimals}f}**"
        elif p_value < 0.05:
            return f"{prefix}{p_value:.{self._decimals}f}*"
        else:
            return f"{prefix}{p_value:.{self._decimals}f}"