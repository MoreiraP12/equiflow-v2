"""
EquiFlow: Equity-focused Cohort Selection Flow Diagrams.

A Python package for creating visual flow diagrams that track demographic
composition changes through sequential cohort exclusion steps in clinical
and epidemiological research.

Example
-------
>>> from equiflow import EquiFlow
>>> import pandas as pd
>>>
>>> # Load your data
>>> data = pd.read_csv("patients.csv")
>>>
>>> # Initialize EquiFlow
>>> ef = EquiFlow(
...     data=data,
...     categorical=['sex', 'race'],
...     normal=['age'],
...     nonnormal=['los_days']
... )
>>>
>>> # Add exclusion steps (keep=True means KEEP the row)
>>> ef.add_exclusion(
...     keep=data['age'] >= 18,
...     exclusion_reason="Age < 18",
...     new_cohort_label="Adult patients"
... )
>>>
>>> # Generate flow diagram
>>> ef.plot_flows(output_file="patient_flow")

For more information, see:
- Documentation: https://equiflow.readthedocs.io
- GitHub: https://github.com/MoreiraP12/equiflow-v2
- Paper: https://doi.org/10.1016/j.jbi.2024.104631
"""

from .equiflow import (
    EquiFlow,
    EasyFlow,
    TableFlows,
    TableCharacteristics,
    TableDrifts,
    TablePValues,
    FlowDiagram,
)

__author__ = "João Matos, Jacob Ellen, Pedro Moreira"
__version__ = "0.1.9"
__license__ = "MIT"

__all__ = [
    # Main classes
    "EquiFlow",
    "EasyFlow",
    # Table classes
    "TableFlows",
    "TableCharacteristics",
    "TableDrifts",
    "TablePValues",
    # Visualization
    "FlowDiagram",
]
