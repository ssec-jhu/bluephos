from functools import reduce
from rdkit import Chem
from rdkit.Chem.rdmolops import CombineMols, SanitizeMol, Kekulize
from rdkit.Chem.rdmolfiles import MolFromSmiles

def ligate(ligands, metal_atom_element = "Ir", metal_atom = None):
    for ligand in ligands:
        ligand.RemoveAllConformers()
        Kekulize(ligand)
    if metal_atom is None:
        metal_atom = MolFromSmiles(f"[{metal_atom_element}]")
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
        editable.AddBond(coordination_atom_index, metal_atom_index, bond.GetBondType())
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
