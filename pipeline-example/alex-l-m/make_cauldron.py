import gzip
from rdkit.Chem.rdmolfiles import ForwardSDMolSupplier
import cauldronoid

rdkit_mols = ForwardSDMolSupplier(gzip.open("pipeline_example_mols.sdf.gz"), removeHs = False)

cauldron = [cauldronoid.rdkit2cauldronoid(rdkit_mol) \
            for rdkit_mol in rdkit_mols]

cauldronoid.mols2files(cauldron, "pipeline_example")
