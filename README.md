# Equiflow

[![Tests](https://github.com/MoreiraP12/equiflow-v2/actions/workflows/test.yml/badge.svg)](https://github.com/MoreiraP12/equiflow-v2/actions/workflows/test.yml)\
[![PyPI
version](https://badge.fury.io/py/equiflow.svg)](https://badge.fury.io/py/equiflow)\
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)\
[![Python
3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**equiflow** is a Python package for generating equity-focused cohort
selection flow diagrams. It facilitates transparent, reproducible
documentation of cohort curation in clinical and machine learning
research, helping investigators quantify and visualize demographic drift
introduced by exclusion criteria.

------------------------------------------------------------------------

## Features

-   **Cohort Flow Visualization** --- Generate publication-ready flow
    diagrams showing patient counts at each exclusion step\
-   **Distribution Tracking** --- Monitor categorical, normally
    distributed, and non-normal variables across selection steps\
-   **Demographic Drift Detection** --- Compute standardized mean
    differences (SMDs) between consecutive cohorts\
-   **Statistical Testing** --- Perform hypothesis testing with optional
    multiple testing correction (Bonferroni, Benjamini-Hochberg)\
-   **Flexible Interfaces** --- Use the fully customizable `EquiFlow`
    class or the streamlined `EasyFlow` API

------------------------------------------------------------------------

## Installation

``` bash
pip install equiflow
```

------------------------------------------------------------------------

## System Dependencies

For diagram generation, Graphviz must be installed:

``` bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

------------------------------------------------------------------------

## Python Dependencies

-   pandas\
-   numpy\
-   matplotlib\
-   graphviz\
-   scipy

------------------------------------------------------------------------

# Quick Start

## Using EquiFlow (Full Control)

``` python
from equiflow import *
import pandas as pd

flow = EquiFlow(
    data=your_dataframe,
    categorical=['sex', 'race', 'insurance_type'],
    normal=['age', 'weight', 'height'],
    nonnormal=['hospital_stay_days', 'num_previous_admissions']
)

flow.add_exclusion(
    keep=lambda df: df['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult patients"
)

flow.add_exclusion(
    keep=lambda df: df['has_complete_data'] == True,
    exclusion_reason="Incomplete data",
    new_cohort_label="Complete cases"
)

flow.plot_flows(
    output_file="patient_selection_flow",
    plot_dists=True,
    smds=True,
    legend=True,
    smd_decimals=1
)
```

## Using EasyFlow (Streamlined API)

``` python
from equiflow import *

flow = (
    EasyFlow(your_dataframe, title="Initial Cohort")
    .categorize(['sex', 'race', 'insurance_type'])
    .measure_normal(['age', 'weight', 'height'])
    .measure_nonnormal(['hospital_stay_days'])
    .exclude(lambda df: df['age'] >= 18, "Age < 18 years")
    .exclude(lambda df: df['has_complete_data'] == True, "Incomplete data")
    .generate(output="patient_flow", show=True)
)

print(flow.flow_table)
print(flow.drifts)
```

------------------------------------------------------------------------

# Standardized Mean Differences (SMDs)

SMDs quantify distribution changes between **each cohort and the
immediately preceding cohort** in the flow diagram.

-   **Categorical variables**: Cohen's h with Hedges' correction\
-   **Continuous variables**: Cohen's d with Hedges' correction\
-   **Interpretation guide**:
    -   SMD \> 0.1 → meaningful drift\
    -   SMD \> 0.2 → substantial change

------------------------------------------------------------------------

# Statistical Testing

`view_table_pvalues()` performs stepwise testing between consecutive
cohorts:

-   **Categorical variables**: Chi-square test (Fisher's exact for 2×2
    tables)\
-   **Normal continuous**: Welch's t-test\
-   **Non-normal continuous**: Kruskal-Wallis test\
-   **Missingness**: Two-proportion z-test comparing the proportion of
    missing values between consecutive cohorts

Multiple testing correction options:

-   `"none"` --- No correction (default)\
-   `"bonferroni"` --- Controls family-wise error rate\
-   `"fdr_bh"` --- Benjamini-Hochberg procedure

------------------------------------------------------------------------

# Citation

If you use EquiFlow in your research, please cite:

Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health
equity in AI. *J Biomed Inform*. 2024;152:104631.
https://doi.org/10.1016/j.jbi.2024.104631

------------------------------------------------------------------------

# License

MIT License
