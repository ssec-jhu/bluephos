from rdkit.Chem.rdMolTransforms import GetBondLength

# Function that checks bond length to verify geometries
elems_to_check = ["C", "H", "N", "O", "F", "Cl"]
def bond_lengths(mol):
    max_nonir_bond_length = 0
    max_ir_bond_length = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        start_elem = bond.GetBeginAtom().GetSymbol()
        end_elem = bond.GetEndAtom().GetSymbol()
        if start_elem in elems_to_check and end_elem in elems_to_check:
            max_nonir_bond_length = max([GetBondLength(mol.GetConformer(0), i, j),
                                         max_nonir_bond_length])
        if start_elem in "Ir" or end_elem in "Ir":
            max_ir_bond_length = max([GetBondLength(mol.GetConformer(0), i, j),
                                      max_ir_bond_length])
    return max_nonir_bond_length, max_ir_bond_length

def bonds_maintained(mol):
    max_nonir_bond_length, max_ir_bond_length = bond_lengths(mol)
    return max_nonir_bond_length < 2 and max_ir_bond_length < 2.5
