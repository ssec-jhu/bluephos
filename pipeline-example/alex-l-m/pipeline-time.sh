#!/bin/bash

# Running generate_ligand_table.py
echo "Running generate_ligand_table.py..."
time python generate_ligand_table.py

# Shorten the combinatorial ligands file
echo "Running head command on combinatorial_ligands.csv..."
head -2 output_data/combinatorial_ligands.csv > output_data/combinatorial_ligands_short.csv

# Running smiles2mol.py
echo "Running smiles2mol.py..."
time python smiles2mol.py

# Compress the SDF file
echo "Running gzip on pipeline_example_mols.sdf..."
gzip pipeline_example_mols.sdf

# Running make_cauldron.py
echo "Running make_cauldron.py..."
time python make_cauldron.py

# Running join_features.R
echo "Running join_features.R..."
time Rscript join_features.R

# Running nn.py
echo "Running nn.py..."
time python nn.py

# Create directory for geometry optimization
echo "Creating directory triplet_xtb_geom..."
mkdir triplet_xtb_geom

# Running optimize_geometries.py
echo "Running optimize_geometries.py..."
time python optimize_geometries.py pipeline_example_mols.sdf.gz

# Create directory for XYZ files
echo "Creating directory xyzs..."
mkdir xyzs

# Running mol2xyz.py
echo "Running mol2xyz.py..."
time python mol2xyz.py

# Run quantum chemical calculations
echo "Running runall.sh..."
time ./runall.sh

# Running extract.py
echo "Running extract.py..."
time python extract.py

