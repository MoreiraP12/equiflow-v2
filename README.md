# equiflow

[![Tests](https://github.com/joamats/equiflow/actions/workflows/test.yml/badge.svg)](https://github.com/joamats/equiflow/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/equiflow.svg)](https://badge.fury.io/py/equiflow)
[![Documentation Status](https://readthedocs.org/projects/equiflow/badge/?version=latest)](https://equiflow.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

***Under construction!***

*equiflow* is a package designed to generate "Equity-focused Cohort Selection Flow Diagrams". We hope to facilitate research, increase its reproducibility, and improve the transparency of the process of cohort curation in machine learning studies.

## Installation

```bash
pip install equiflow
```

## Quick Start

```python
import pandas as pd
from equiflow import EquiFlow

# Load your data
data = pd.read_csv("patients.csv")

# Initialize EquiFlow
ef = EquiFlow(
    data=data,
    categorical=['sex', 'race', 'ethnicity'],
    normal=['age'],
    nonnormal=['los_days', 'apache_score']
)

# Add exclusion steps (keep=True means KEEP the row)
ef.add_exclusion(
    keep=data['age'] >= 18,
    exclusion_reason="Age < 18",
    new_cohort_label="Adult patients"
)

ef.add_exclusion(
    keep=ef._dfs[-1]['los_days'] > 0,
    exclusion_reason="LOS ≤ 0 days",
    new_cohort_label="Valid LOS"
)

# Generate the flow diagram
ef.plot_flows(output_file="patient_flow")

# View summary tables
print(ef.view_table_flows())
print(ef.view_table_characteristics())
print(ef.view_table_drifts())
```

## EasyFlow: Simplified Interface

For quick analyses, use the `EasyFlow` class with method chaining:

```python
from equiflow import EasyFlow

flow = (EasyFlow(data, title="My Study")
    .categorize(['sex', 'race'])
    .measure_normal(['age'])
    .measure_nonnormal(['score'])
    .exclude(data['age'] >= 18, "Age < 18")
    .exclude(data['score'].notna(), "Missing score")
    .generate(output="my_flow"))
```

## Vision

*equiflow* provides tabular and visual representations of inclusion and exclusion criteria applied to a clinical dataset. Each patient exclusion step can depict the cohort composition across demographics and outcomes, to interrogate potential sampling selection bias.

This package is designed to enhance the transparency and reproducibility of research in the medical machine learning field. It complements other tools like [tableone](https://github.com/tompollard/tableone), which is used for generating summary statistics for patient populations.

## Features

- **Flow Tables**: Track cohort sizes through exclusion steps
- **Characteristics Tables**: Demographic breakdown at each step
- **Drift Tables**: Standardized Mean Differences (SMDs) to detect demographic drift
- **P-value Tables**: Statistical significance testing with multiple testing correction
- **Flow Diagrams**: Publication-ready visual flow diagrams with embedded distribution plots

## Citation

The concept was first introduced in our [position paper](https://www.sciencedirect.com/science/article/pii/S1532046424000492).

> Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health equity in AI. J Biomed Inform. 2024;152:104631. https://doi.org/10.1016/j.jbi.2024.104631

## Motivation

Selection bias can arise through many aspects of a study, including recruitment, inclusion/exclusion criteria, input-level exclusion and outcome-level exclusion, and often reflects the underrepresentation of populations historically disadvantaged in medical research. The effects of selection bias can be further amplified when non-representative samples are used in artificial intelligence (AI) and machine learning (ML) applications to construct clinical algorithms.

Building on the "Data Cards" initiative for transparency in AI research, we advocate for the addition of a **participant flow diagram for AI studies detailing relevant sociodemographic** and/or clinical characteristics of excluded participants across study phases, with the goal of identifying potential algorithmic biases before their clinical implementation.

## Development

```bash
# Clone the repository
git clone https://github.com/joamats/equiflow.git
cd equiflow

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.
