# import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles, AllChem
# from rdkit.Chem.rdmolfiles import SDWriter
from rdkit.Chem.rdmolops import CombineMols, SanitizeMol, Kekulize
from functools import reduce
from dplutils.pipeline import PipelineTask
# import ray


def smiles_to_sdf(df: pd.DataFrame, output_folder) -> pd.DataFrame:

    # Adjusted 'ligate' function to include necessary imports
    def ligate(ligands, metal_atom_element="Ir", metal_atom=None):

        for ligand in ligands:
            ligand.RemoveAllConformers()
            Kekulize(ligand)
        if metal_atom is None:
            metal_atom = MolFromSmiles(f"[{metal_atom_element}]")
        mol = reduce(CombineMols, ligands, metal_atom)
        # Create a molecule that contains the metal atom as well as all the
        # ligands, but without bonds between the metal and the ligands
        mol = reduce(CombineMols, ligands, metal_atom)
        # Record the index of the metal atom, so that bonds can be created to it
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # This will not work if the ligands also contain the metal atom!
        metal_atom_index = elements.index(metal_atom_element)
        # Create an editable molecule and begin batch editing, which makes
        # functions available for adding and removing bonds as well as removing the
        # dummy atoms, and should keep the indexes stable as we do so
        editable = Chem.EditableMol(mol)
        editable.BeginBatchEdit()
        # Check each bond to see if it is a coordination site, and if so, bond the
        # atom to the metal
        for bond in mol.GetBonds():
            # Figure out which, if any, of the atoms in the bond is the coordinating atom
            if bond.GetBeginAtom().GetSymbol() == "*":
                dummy_atom_index = bond.GetBeginAtomIdx()
                coordination_atom_index = bond.GetEndAtomIdx()
            elif bond.GetEndAtom().GetSymbol() == "*":
                dummy_atom_index = bond.GetEndAtomIdx()
                coordination_atom_index = bond.GetBeginAtomIdx()
            else:
                continue
            # Add a bond to metal of the same type as the bond to the dummy atom
            # Has to have metal second for the dative bonds to work
            # Order matters for dative bonds
            # https://www.rdkit.org/docs/RDKit_Book.html#dative-bonds
            editable.AddBond(
                coordination_atom_index, metal_atom_index, bond.GetBondType()
            )
            # Remove the bond between the coordinating atom and the dummy atom
            editable.RemoveBond(dummy_atom_index, coordination_atom_index)
            # Remove all the dummy atoms
        for i, element in enumerate(elements):
            if element == "*":
                editable.RemoveAtom(i)
        # Apply the changes and get the molecule
        editable.CommitBatchEdit()
        outmol = editable.GetMol()
        # Shouldn't have any conformers but just in case
        outmol.RemoveAllConformers()
        # Probably already done by GetMol but just in case
        SanitizeMol(outmol)
        return outmol

    # Initialize an empty list to hold the RDKit molecule objects
    molecules = []
    # with SDWriter(os.path.join(output_folder, "pipeline_example_mols.sdf")) as writer:
    # writer.SetForceV3000(True)

    for index, row in df.iterrows():  # Assuming df is a DataFrame
        ligand = MolFromSmiles(row["ligand_SMILES"])
        if ligand is not None:
            ligated_mol = ligate([ligand, ligand, ligand])
            if ligated_mol is not None:
                mol = AddHs(ligated_mol)
                AllChem.Compute2DCoords(mol)  # Compute 2D coordinates for the molecule
                mol.SetProp("_Name", row["ligand_identifier"])
                # molecules.append(mol)
                molecules.append(Chem.MolToMolBlock(mol))
                # writer.write(mol)
            else:
                # Handle the case where ligate returns None
                print(
                    f"Ligation failed for index {index}, identifier {row['ligand_identifier']}."
                )
        #         molecules.append(None)
        # else:
        #     molecules.append(None)

    # Create a DataFrame to return
    result_df = pd.DataFrame(
        {
            "molecules": molecules  # Assuming you want to store the molecule objects directly; adjust as needed
        }
    )

    # For Verification
    # for rdkit_mol in result_df["molecules"]:
    #     mol_block = Chem.MolToMolBlock(rdkit_mol)
    #     print(mol_block)

    return result_df


Smiles2SDFTask = PipelineTask(
    "smiles2mol",
    smiles_to_sdf,
    context_kwargs={"output_folder": "output_folder"},
)
