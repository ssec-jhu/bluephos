from glob import glob
from rdkit.Chem import MolFromMolFile, MolToXYZFile
# Convert all molecules in the folder of otuput from XTB geometry optimization
# from .mol files to .xyz files for DFT input

inpath = "triplet_xtb_geom"
outdir = "xyzs"

for inpath in glob(f"{inpath}/*.mol"):
    mol = MolFromMolFile(inpath, removeHs = False)
    MolToXYZFile(mol, f"{outdir}/{mol.GetProp('_Name')}.xyz")
