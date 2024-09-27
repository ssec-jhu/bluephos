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

## Step 1: Clone the GitHub Repository
* ``git clone https://github.com/ssec-jhu/bluephos.git`` 

## Step 2: Set Up the Runtime Environment
Navigate to the Bluephos directory and create the blue_env environment using Conda:
```
cd bluephos
conda env create -f blue_env.yml
conda activate blue_env
```
After cloning, navigate to the project directory:
* ``cd bluephos``

# Running the Pipeline

## Usage

*    ``python bluephos_pipeline.py [options]``

## Command-Line Arguments

| **Argument**    | **Required** | **Type** | **Default** | **Description**                                                                                                                    |
|-----------------|--------------|----------|-------------|------------------------------------------------------------------------------------------------------------------------------------|
| --halides       | No           | String   | None        | Path to the CSV file containing halides data. Required when no input directory or ligand SMILES CSV file is specified.             |
| --acids         | No           | String   | None        | Path to the CSV file containing boronic acids data. Required when no input directory or ligand SMILES CSV file is specified.       |
| --features      | Yes          | String   | None        | Path to the element feature file used for neural network predictions.                                                              |
| --train         | Yes          | String   | None        | Path to the train stats file used to normalize input data.                                                                         |
| --weights       | Yes          | String   | None        | Path to the full energy model weights file for the neural network.                                                                 |
| --input_dir     | No           | String   | None        | Directory containing input parquet files for rerun mode. Used when mode 3 is not specified.                                        |
| --out-dir       | No           | String   | None        | Directory where the pipeline's output files will be saved. If not specified, defaults to the current directory.                    |
| --t_nn          | No           | Float    | 1.5         | Threshold for the neural network 'z' score. Candidates with an absolute 'z' score below this threshold will be considered.         |
| --t_ste         | No           | Float    | 1.9         | Threshold for 'ste' (Singlet-Triplet Energy gap). Candidates with an absolute 'ste' value below this threshold will be considered. |
| --t_dft         | No           | Float    | 2.0         | Threshold for 'dft' (dft_energy_diff). Candidates with an absolute 'dft' value below this threshold will be considered.            |
| --ligand_smiles | No           | String   | None        | Path to the ligand SMILE file containing ligand SMILES data. If provided, mode 3 is used.                                          |
| --no_xtb        | No           | Bool     | True        | Disable xTB optimization. Defaults (no this flag) to xTB optimization enabled; use --no_xtb to disable it.


## The BluePhos Discovery Pipeline now supports three modes of input data:
1.	Generate data from Halides and Acids CSV files: This mode is used when no input directory or ligand SMILES CSV file is specified. It generates ligand pairs from the provided halides and acids CSV files.
2.	Rerun data from parquet files: This mode is used when an input directory is specified. It reruns the pipeline using existing parquet files for ligand data.
3.	Input data from a ligand SMILES CSV file: This mode is prioritized if a ligand SMILES CSV file is provided. It directly processes ligands from the SMILES data.

### &nbsp;&nbsp;The priority order for these modes is 3 > 2 > 1, meaning:
&nbsp;&nbsp;-If a ligand SMILES CSV file (--ligand_smiles) is provided, the pipeline operates in mode 3.  
&nbsp;&nbsp;-If an input directory (--input_dir) is specified, and no ligand SMILES CSV file is provided, the pipeline operates in mode 2.  
&nbsp;&nbsp;-If neither a ligand SMILES CSV file nor an input directory is provided, the pipeline defaults to mode 1.

# Example Commands
1.	Generating Ligand Pairs and Running the Full Pipeline (Mode1):
If you want to generate ligand pairs from halides and acids files and run the full pipeline, you must specify the paths to the halides and acids files:
* ``python bluephos_pipeline.py --halides path/to/halides.csv --acids path/to/acids.csv --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5``
2.	Rerunning the Pipeline with Existing Parquet Files (Mode2):
If you have already run the pipeline for the ligands and want to rerun it for refiltering or recalculating the ligands based on previous results:
* ``python bluephos_pipeline.py --input_dir path/to/parquet_directory --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5``
3.	Using Ligand SMILES CSV File (Mode 3):
* ``python bluephos_pipeline.py --ligand_smiles path/to/ligand_smiles.csv --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5``
4.	Specifying Different Thresholds for NN and STE
You can adjust the thresholds for the neural network 'z' score and the xTB standard error (STE) as needed:
* ``python bluephos_pipeline.py --halides path/to/halides.csv --acids path/to/acids.csv --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5 --t_nn 2.0 --t_ste 2.5``
5.	Using a Different DFT Package
By default, the pipeline uses the ORCA DFT package, but you can switch to ASE (to be implemented later) if preferred:
* ``python bluephos_pipeline.py --halides path/to/halides.csv --acids path/to/acids.csv --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5 --dft_package ase``
6. Disable xTB optimiazation
By default, the geometries optimization task uses the xTB package.However you can disable it by running:
* ``python bluephos_pipeline.py --halides path/to/halides.csv --acids path/to/acids.csv --features path/to/features.csv --train path/to/train_stats.csv --weights path/to/model_weights.h5 --no_xtb``



## Execute the BluePhos pipeline within a tox environment for a consistent and reproducible setup:

* ``tox -e run-pipeline -- --halide /path/to/aromatic_halides.csv --acid /path/to/aromatic_boronic_acids.csv --feature /path/to/element_features.csv --train /path/to/train_stats.csv --weight /path/to/model_weights.pt -o /path/to/output_dir/``

Replace /path/to/... with the actual paths to your datasets and parameter files.


## Example Usage with Test Data
To run the pipeline using example data provided in the repository:

* ``tox -e run-pipeline -- --halide ./tests/input/aromatic_halides_with_id.csv --acid ./tests/input/aromatic_boronic_acids_with_id.csv --feature ./bluephos/parameters/element_features.csv --train ./bluephos/parameters/train_stats.csv --weight ./bluephos/parameters/full_energy_model_weights.pt -o .``

This command uses test data to demonstrate the pipeline's functionality, ideal for initial testing and familiarization.

# Result

## Note:
* The default output (--o) dataframe is stored in Parquet format due to its efficient storage, faster data access, and enhanced support for complex data structures. 
* The pipeline's results are organized by task, with filtered-out data stored in specific subdirectories within the /output directory. For example:
-The filtered-out data from the NN task is stored in /NN_filter_out.
-For the XTB task, the filtered-out data is saved in /XTB_filter_out.
-For the final DFT task, the results are divided into two directories: /DFT_filter_in for filtered-in data and /DFT_filter_out for filtered-out data.

## The Parquet file can be accessed in several ways:
### Using Pandas 
Pandas can be used to read and analyze Parquet files.
```py
import pandas as pd
df = pd.read_parquet('08ca147e-f618-11ee-b38f-eab1f408aca3-8.parquet')
print(df.describe())
```
### Using DuckDB
DuckDB provides an efficient way to query Parquet files directly using SQL syntax.
```py
import duckdb as ddb
query_result = ddb.query('''SELECT * FROM '08ca147e-f618-11ee-b38f-eab1f408aca3-8.parquet' LIMIT 10''')
print(query_result.to_df())
```





