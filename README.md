# equiflow

***Under construction!***

*equiflow* is a package designed to generate "Equity-focused Cohort Selection Flow Diagrams". We hope to facilitate research, increase its reproducibility, and improve the transparency of the process of cohort curation in machine learning studies.


## Vision
*equiflow* will provide tabular and visual representations of inclusion and exclusion criteria applied to a clinical dataset. Each patient exclusion step can depict the cohort composition across demographics and outcomes, to interrogate potential sampling selection bias.

This package is designed to enhance the transparency and reproducibility of research in the medical machine learning field. It complements other tools like tableone, which is used for generating summary statistics for patient populations.


### EquiFlow Class

The main class that handles:
- Initializing with a dataset or multiple dataframes
- Adding exclusion criteria
- Viewing tables of flows, characteristics, and distribution shifts
- Plotting cohort flow diagrams

### Cohort Flow Visualization

EquiFlow generates flow diagrams showing:
- Initial cohort size
- Exclusion criteria with number of excluded patients
- Resulting cohort sizes at each step
- Optional visualizations of variable distributions at each step

### Distribution Analysis

The package can analyze:
- Categorical variables (showing percentages, counts, or both)
- Normally distributed continuous variables (showing mean ± SD)
- Non-normally distributed continuous variables (showing median [IQR])
- Missing data rates

### Distribution Drift Measurement

EquiFlow calculates standardized mean differences (SMDs) between consecutive cohorts to quantify:
- How much variable distributions change at each exclusion step
- Which variables are most affected by selection criteria
- Potential bias introduced in the selection process

## Installation

```bash
pip install equiflow
```

## Basic Usage

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

# Add exclusion steps
flow.add_exclusion(
    mask=your_dataframe['age'] >= 18,
    exclusion_reason="Age < 18 years",
    new_cohort_label="Adult patients"
)

flow.add_exclusion(
    mask=your_dataframe['has_complete_data'] == True,
    exclusion_reason="Incomplete data",
    new_cohort_label="Complete cases"
)

# View tables
flow_table = flow.view_table_flows()
characteristics_table = flow.view_table_characteristics()
drifts_table = flow.view_table_drifts()

# Generate flow diagram
flow.plot_flows(
    output_file="patient_selection_flow"
)
```

## Benefits for Research Equity

EquiFlow helps researchers:
- Make cohort selection decisions more transparent
- Identify when exclusion criteria may disproportionately affect certain groups
- Ensure research samples maintain representative distributions of key variables
- Document cohort selection in a standardized, reproducible way
- Comply with equity-focused reporting guidelines in research

## Citation

If you use EquiFlow in your research, please cite:
[Citation information would be here]

## Citation
The concept was first introuced in our [position paper](https://www.sciencedirect.com/science/article/pii/S1532046424000492).

> Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health equity in AI. J Biomed Inform. 2024;152:104631. [https://doi.org/10.1016/j.jbi.2024.104631](https://doi.org/10.1016/j.jbi.2024.104631)


## Motivation

Selection bias can arise through many aspects of a study, including recruitment, inclusion/exclusion criteria, input-level exclusion and outcome-level exclusion, and often reflects the underrepresentation of populations historically disadvantaged in medical research. The effects of selection bias can be further amplified when non-representative samples are used in artificial intelligence (AI) and machine learning (ML) applications to construct clinical algorithms. Building on the “Data Cards” initiative for transparency in AI research, we advocate for the addition of a **participant flow diagram for AI studies detailing relevant sociodemographic** and/or clinical characteristics of excluded participants across study phases, with the goal of identifying potential algorithmic biases before their clinical implementation. We include both a model for this flow diagram as well as a brief case study explaining how it could be implemented in practice. Through standardized reporting of participant flow diagrams, we aim to better identify potential inequities embedded in AI applications, facilitating more reliable and equitable clinical algorithms.

