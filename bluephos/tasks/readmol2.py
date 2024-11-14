'''Interpret the smiles column as input file names of mol2 files. Really I should rename the column but I'm being cautious in case other modules reference it by name'''

from functools import reduce
import pandas as pd
import bluephos.modules.log_config as log_config
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem, MolFromMol2File
from rdkit.Chem.rdmolops import CombineMols, Kekulize, SanitizeMol


# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)



def smiles_to_sdf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SMILES strings to SDF format and Add to Input dataframe"""

    for index, row in df.iterrows():
        # The column is labelled 'smiles' but it actually contains a path to a
        # mol2 file
        inpath = row['smiles']
        mol_nohs = MolFromMol2File(inpath)
        if mol_nohs is not None:
            mol = AddHs(mol_nohs)
            AllChem.Compute2DCoords(mol)
            mol.SetProp("_Name", row["mol_id"])
            mol_block = Chem.MolToMolBlock(mol)
            df.at[index, "structure"] = mol_block
        else:
            logger.warning(f"mol generation failed for index {index}, identifier {row['mol_id']}.")
    return df


Smiles2SDFTask = PipelineTask("smiles2sdf", smiles_to_sdf)
