# SSEC-JHU bluephos

[![CI](https://github.com/ssec-jhu/bluephos/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/bluephos/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/ssec-jhu-bluephos/badge/?version=latest)](https://ssec-jhu-bluephos.readthedocs.io/en/latest/?badge=latest)
[![Security](https://github.com/ssec-jhu/bluephos/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/bluephos/actions/workflows/security.yml)
<!---[![DOI](https://zenodo.org/badge/<insert_ID_number>.svg)](https://zenodo.org/badge/latestdoi/<insert_ID_number>) --->


![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

BluePhos: An automated pipeline optimizing the synthesis and analysis of blue phosphorescent materials.

# BluePhos Pipeline Introduction

### Overview

The BluePhos pipeline is an automated computational tool streamlining the development and analysis of blue phosphorescent materials. It blends computational chemistry with machine learning to adeptly predict and hone the properties of essential compounds in light-emitting tech.

### Workflow Evolution

The BluePhos pipeline functions like an automated assembly line, with a structured yet adaptable workflow that distributes tasks efficiently across computing resources. It optimizes batch processing and resource allocation, processing molecules individually for streamlined operation.

The current version of the pipeline comprises the following sequential tasks:

* Ligand Generation Task: It commences by ingesting aromatic boronic acids and aromatic halides, generating ligand molecules via Suzuki coupling reactions.
* SMILES to SDF Conversion Task: Molecular structures encoded in SMILES strings are converted into SDF files, facilitating in-depth chemical data manipulation.
* Neural Network (NN) Task: This phase involves the extraction and engineering of features from each ligand. These features are processed through a trained graph neural network to predict the ligand's z-score, indicative of synthetic potential.

Planned enhancements include:

* Optimization Geometry Task: Aiming to optimize molecular geometries, ensuring that the ligands adopt energetically favorable conformations.
* Density Functional Theory (DFT) Calculation Task: Set to apply DFT calculations to optimized geometries for in-depth quantum mechanical insights into the ligands' electronic properties.

# Setup and Installation

Begin by cloning the BluePhos project repository to your local system:
* ``git clone https://github.com/ssec-jhu/bluephos.git`` 


After cloning, navigate to the project directory:
* ``cd bluephos``

# Running the Pipeline
Execute the BluePhos pipeline within a tox environment for a consistent and reproducible setup:

* ``tox -e run-pipeline -- --halide /path/to/aromatic_halides.csv --acid /path/to/aromatic_boronic_acids.csv --feature /path/to/element_features.csv --train /path/to/train_stats.csv --weight /path/to/model_weights.pt --o /path/to/output_dir/``

Replace /path/to/... with the actual paths to your datasets and parameter files.


## Example Usage with Test Data
To run the pipeline using example data provided in the repository:

* ``tox -e run-pipeline -- --halide ./tests/input/aromatic_halides_with_id.csv --acid ./tests/input/aromatic_boronic_acids_with_id.csv --feature ./bluephos/parameters/element_features.csv --train ./bluephos/parameters/train_stats.csv --weight ./bluephos/parameters/full_energy_model_weights.pt --o .``

This command uses test data to demonstrate the pipeline's functionality, ideal for initial testing and familiarization.

## Note:
The default output (--o) dataframe is stored in Parquet format due to its efficient storage, faster data access, and enhanced support for complex data structures. The Parquet file can be accessed in several ways:
### Using Pandas 
Pandas can be used to read and analyze Parquet files.

``import pandas as pd``

``df = pd.read_parquet('08ca147e-f618-11ee-b38f-eab1f408aca3-8.parquet')``

``print(df.describe())``

### Using DuckDB
DuckDB provides an efficient way to query Parquet files directly using SQL syntax.

``import duckdb as ddb``

``query_result = ddb.query('''SELECT * FROM '08ca147e-f618-11ee-b38f-eab1f408aca3-8.parquet' LIMIT 10''')``

``print(query_result.to_df())``






