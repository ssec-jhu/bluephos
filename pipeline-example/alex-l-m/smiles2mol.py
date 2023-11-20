from rdkit.Chem.rdmolfiles import MolFromSmiles, SDWriter
from rdkit.Chem.rdmolops import AddHs
from ligate import ligate
import pandas as pd

f = open("pipeline_example_mols.sdf", "w")
writer = SDWriter(f)
writer.SetForceV3000(True)
for i, row in enumerate(pd.read_csv("output_data/combinatorial_ligands_short.csv").itertuples()):
    ligand = MolFromSmiles(row.ligand_SMILES)
    if ligand is not None:
        mol = AddHs(ligate([ligand, ligand, ligand]))
        mol.SetProp("_Name", row.ligand_identifier)
        writer.write(mol)
writer.close()
f.close()
