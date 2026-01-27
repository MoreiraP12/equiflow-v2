Usage
=====

.. _installation:

Installation
------------

To install equiflow, use pip:

.. code-block:: console

   pip install equiflow

Quick Start
-----------

Here's a simple example of how to use equiflow:

.. code-block:: python

   import pandas as pd
   from equiflow import EquiFlow

   # Load your data
   data = pd.read_csv("patients.csv")

   # Initialize EquiFlow with your demographic variables
   ef = EquiFlow(
       data=data,
       categorical=['sex', 'race', 'ethnicity'],
       normal=['age'],
       nonnormal=['los_days', 'apache_score']
   )

   # Add exclusion steps
   # Note: keep=True means KEEP the row (retain it in the cohort)
   ef.add_exclusion(
       keep=data['age'] >= 18,
       exclusion_reason="Age < 18",
       new_cohort_label="Adult patients"
   )

   ef.add_exclusion(
       keep=data['los_days'] > 0,
       exclusion_reason="LOS ≤ 0 days",
       new_cohort_label="Valid LOS"
   )

   # Generate the flow diagram
   ef.plot_flows(output_file="patient_flow")

   # View summary tables
   print(ef.view_table_flows())
   print(ef.view_table_characteristics())
   print(ef.view_table_drifts())

Understanding the ``keep`` Parameter
------------------------------------

The ``keep`` parameter in ``add_exclusion()`` specifies which rows to **retain** 
in the cohort. This follows a "positive selection" paradigm:

- ``keep=True`` → Row is **kept** (retained)
- ``keep=False`` → Row is **excluded** (removed)

For example, to exclude patients under 18:

.. code-block:: python

   # Keep patients who are 18 or older
   ef.add_exclusion(
       keep=data['age'] >= 18,
       exclusion_reason="Age < 18"
   )

EasyFlow: Simplified Interface
------------------------------

For quick analyses, use the ``EasyFlow`` class:

.. code-block:: python

   from equiflow import EasyFlow

   flow = (EasyFlow(data, title="My Study")
       .categorize(['sex', 'race'])
       .measure_normal(['age'])
       .measure_nonnormal(['score'])
       .exclude(data['age'] >= 18, "Age < 18")
       .exclude(data['score'].notna(), "Missing score")
       .generate(output="my_flow"))

Tables and Statistics
---------------------

EquiFlow provides several analysis tables:

**Flow Table** - Shows cohort sizes at each step:

.. code-block:: python

   ef.view_table_flows()

**Characteristics Table** - Shows demographic breakdown:

.. code-block:: python

   ef.view_table_characteristics()

**Drift Table** - Shows Standardized Mean Differences (SMDs):

.. code-block:: python

   ef.view_table_drifts()

**P-values Table** - Statistical significance testing:

.. code-block:: python

   ef.view_table_pvalues(correction="bonferroni")

Customization
-------------

**Variable ordering:**

.. code-block:: python

   ef = EquiFlow(
       data=data,
       categorical=['sex', 'race'],
       order_vars=['race', 'sex'],  # Display race first
       order_classes={'race': ['White', 'Black', 'Asian', 'Other']}
   )

**Limiting categories:**

.. code-block:: python

   ef = EquiFlow(
       data=data,
       categorical=['diagnosis'],
       limit={'diagnosis': 5}  # Show only top 5 diagnoses
   )

**Custom formatting:**

.. code-block:: python

   ef = EquiFlow(
       data=data,
       categorical=['sex'],
       format_cat="N (%)",      # Options: "N (%)", "N", "%"
       format_normal="Mean ± SD",
       decimals=2
   )

Citation
--------

If you use equiflow in your research, please cite:

   Ellen JG, Matos J, Viola M, et al. Participant flow diagrams for health 
   equity in AI. J Biomed Inform. 2024;152:104631. 
   https://doi.org/10.1016/j.jbi.2024.104631
