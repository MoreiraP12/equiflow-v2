# equiflow

[![Tests](https://github.com/MoreiraP12/equiflow-v2/actions/workflows/test.yml/badge.svg)](https://github.com/MoreiraP12/equiflow-v2/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/equiflow.svg)](https://badge.fury.io/py/equiflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**equiflow** is a Python package for generating Equity-focused Cohort Selection Flow Diagrams. It facilitates transparent, reproducible documentation of cohort curation in clinical and machine learning research, helping researchers identify and quantify potential selection bias.

## Features

- **Cohort Flow Visualization**: Generate publication-ready flow diagrams showing patient counts at each exclusion step
- **Distribution Analysis**: Track categorical, normal, and non-normal continuous variables through the selection process
- **Demographic Drift Detection**: Calculate standardized mean differences (SMDs) to quantify how exclusion criteria affect variable distributions
- **Statistical Testing**: Compute p-values with optional multiple testing correction (Bonferroni, Benjamini-Hochberg)
- **Flexible Interfaces**: Use the detailed `EquiFlow` class or the streamlined `EasyFlow` API

## Installation

```bash
pip install equiflow
```

### System Dependencies

For flow diagram generation, you need Graphviz installed:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

### Python Dependencies

- pandas
- numpy
- matplotlib
- graphviz
- scipy

## Quick Start

### Using EquiFlow (Full Control)

```python
from equiflow import EquiFlow
import pandas as pd

# Initialize with your dataset
flow = EquiFlow(
    data=your_dataframe,
    categorical=['sex', 'race', 'insurance_type'],
    normal=['age', 'weight', 'height'],
    nonnormal=['hospital_stay_days', 'num_previous_admissions']
)

# Add exclusion steps (keep=True means KEEP the row)
flow.add_exclusion(
    keep=your_dataframe['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult patients"
)

flow.add_exclusion(
    keep=your_dataframe['has_complete_data'] == True,
    exclusion_reason="Incomplete data",
    new_cohort_label="Complete cases"
)

# View tables
flow_table = flow.view_table_flows()
characteristics_table = flow.view_table_characteristics()
drifts_table = flow.view_table_drifts()
pvalues_table = flow.view_table_pvalues(correction="fdr_bh")

# Generate flow diagram
flow.plot_flows(
    output_file="patient_selection_flow",
    plot_dists=True,
    smds=True,
    legend=True
)
```

### Using EasyFlow (Streamlined API)

```python
from equiflow import EasyFlow

# Chainable API for quick analysis
flow = (
    EasyFlow(your_dataframe, title="Initial Cohort")
    .categorize(['sex', 'race', 'insurance_type'])
    .measure_normal(['age', 'weight', 'height'])
    .measure_nonnormal(['hospital_stay_days'])
    .exclude(your_dataframe['age'] >= 18, "Age < 18 years")
    .exclude(lambda df: df['has_complete_data'] == True, "Incomplete data")
    .generate(output="patient_flow", show=True)
)

# Access results
print(flow.flow_table)
print(flow.characteristics)
print(flow.drifts)
```

## Core Classes

### EquiFlow

The main class for creating cohort flow diagrams with full customization:

| Method | Description |
|--------|-------------|
| `add_exclusion(keep, exclusion_reason, new_cohort_label)` | Add an exclusion step; rows where `keep=True` are retained |
| `view_table_flows()` | Get cohort sizes at each step |
| `view_table_characteristics()` | Get variable distributions for each cohort |
| `view_table_drifts()` | Get SMDs between consecutive cohorts |
| `view_table_pvalues(correction)` | Get p-values with optional multiple testing correction |
| `plot_flows()` | Generate the visual flow diagram |

### EasyFlow

A simplified, chainable interface for rapid analysis:

| Method | Description |
|--------|-------------|
| `categorize(variables)` | Set categorical variables |
| `measure_normal(variables)` | Set normally-distributed continuous variables |
| `measure_nonnormal(variables)` | Set non-normal continuous variables |
| `exclude(condition, label)` | Add exclusion step; rows where `condition=True` are kept |
| `generate(output, show)` | Create the flow diagram |

## Distribution Analysis

EquiFlow supports three variable types:

| Type | Display Format | Example |
|------|----------------|---------|
| Categorical | N (%), %, or N | Sex: Male 52.3% |
| Normal | Mean ± SD | Age: 45.2 ± 12.3 |
| Non-normal | Median [IQR] | LOS: 4.0 [2.0, 8.0] |

## Standardized Mean Differences (SMDs)

SMDs quantify distribution changes between consecutive cohorts:

- **Categorical variables**: Cohen's h with Hedges' correction
- **Continuous variables**: Cohen's d with Hedges' correction
- **Interpretation**: SMD > 0.1 suggests meaningful drift; SMD > 0.2 indicates substantial change

## Statistical Testing

The `view_table_pvalues()` method supports:

- **Categorical variables**: Chi-square test (Fisher's exact for 2×2 tables)
- **Normal continuous**: Welch's t-test
- **Non-normal continuous**: Kruskal-Wallis test
- **Missingness**: Two-proportion z-test

**Multiple testing correction options:**

| Option | Description |
|--------|-------------|
| `"none"` | No correction (default) |
| `"bonferroni"` | Bonferroni correction (controls FWER) |
| `"fdr_bh"` | Benjamini-Hochberg procedure (controls FDR) |

## Benefits for Research Equity

EquiFlow helps researchers:

- Make cohort selection decisions **transparent** and **reproducible**
- Identify when exclusion criteria **disproportionately affect** certain groups
- **Quantify demographic drift** at each selection step
- Document cohort curation in a **standardized format**
- Comply with **equity-focused reporting guidelines**

## Motivation

Selection bias can arise through many aspects of a study, including recruitment, inclusion/exclusion criteria, input-level exclusion and outcome-level exclusion, and often reflects the underrepresentation of populations historically disadvantaged in medical research. The effects of selection bias can be further amplified when non-representative samples are used in artificial intelligence (AI) and machine learning (ML) applications to construct clinical algorithms.

Building on the "Data Cards" initiative for transparency in AI research, we advocate for the addition of a **participant flow diagram for AI studies** detailing relevant sociodemographic and clinical characteristics of excluded participants across study phases, with the goal of identifying potential algorithmic biases before clinical implementation.

## Citation

If you use EquiFlow in your research, please cite our position paper:

> Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health equity in AI. *J Biomed Inform*. 2024;152:104631. https://doi.org/10.1016/j.jbi.2024.104631

```bibtex
@article{ellen2024participant,
  title={Participant flow diagrams for health equity in AI},
  author={Ellen, Jacob G and Matos, Jo{\~a}o and Viola, Matteo and others},
  journal={Journal of Biomedical Informatics},
  volume={152},
  pages={104631},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.jbi.2024.104631}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to:

- Set up a development environment
- Run tests
- Submit pull requests
- Report issues

## Related Tools

- [tableone](https://github.com/tompollard/tableone) - Summary statistics for patient populations
- [aequitas](https://github.com/dssg/aequitas) - Bias and fairness audit toolkit
