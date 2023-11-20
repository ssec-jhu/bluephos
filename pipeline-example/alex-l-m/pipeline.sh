python generate_ligand_table.py 
head -2 output_data/combinatorial_ligands.csv > output_data/combinatorial_ligands_short.csv
python smiles2mol.py
gzip pipeline_example_mols.sdf
python make_cauldron.py
Rscript join_features.R
python nn.py
mkdir -p triplet_xtb_geom
python optimize_geometries.py pipeline_example_mols.sdf.gz
mkdir -p xyzs
python mol2xyz.py
./runall.sh
python extract.py
