from functools import reduce
import pandas as pd
import bluephos.modules.log_config as log_config
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem, MolFromSmiles
from rdkit.Chem.rdmolops import CombineMols, Kekulize, SanitizeMol


# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)


def ligate(ligands, metal_atom_element="Ir", metal_atom=None):
    """Ligate multiple ligands to a metal atom."""
    for ligand in ligands:
        ligand.RemoveAllConformers()
        Kekulize(ligand)
    if metal_atom is None:
        metal_atom = MolFromSmiles(f"[{metal_atom_element}]")
    mol = reduce(CombineMols, ligands, metal_atom)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    metal_atom_index = elements.index(metal_atom_element)
    editable = Chem.EditableMol(mol)
    editable.BeginBatchEdit()
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetSymbol() == "*":
            dummy_atom_index = bond.GetBeginAtomIdx()
            coordination_atom_index = bond.GetEndAtomIdx()
        elif bond.GetEndAtom().GetSymbol() == "*":
            dummy_atom_index = bond.GetEndAtomIdx()
            coordination_atom_index = bond.GetBeginAtomIdx()
        else:
            continue
        editable.AddBond(coordination_atom_index, metal_atom_index, bond.GetBondType())
        editable.RemoveBond(dummy_atom_index, coordination_atom_index)
    for i, element in reversed(list(enumerate(elements))):
        if element == "*":
            editable.RemoveAtom(i)
    editable.CommitBatchEdit()
    outmol = editable.GetMol()
    outmol.RemoveAllConformers()
    SanitizeMol(outmol)
    return outmol


def smiles_to_sdf(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SMILES strings to SDF format and Add to Input dataframe"""

    if "structure" not in df.columns:
        df["structure"] = None

    for index, row in df.iterrows():
        ligand = MolFromSmiles(row["ligand_SMILES"])
        if ligand is not None:
            ligated_mol = ligate([ligand, ligand, ligand])
            if ligated_mol is not None:
                mol = AddHs(ligated_mol)
                AllChem.Compute2DCoords(mol)
                mol.SetProp("_Name", row["ligand_identifier"])
                mol_block = Chem.MolToMolBlock(mol)
                df.at[index, "structure"] = mol_block
            else:
                logger.warning(f"Ligation failed for index {index}, identifier {row['ligand_identifier']}.")
        else:
            logger.warning(f"Invalid SMILES string for index {index}, identifier {row['ligand_identifier']}.")
    return df


Smiles2SDFTask = PipelineTask("smiles2sdf", smiles_to_sdf)
