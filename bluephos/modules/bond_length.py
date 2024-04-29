from rdkit.Chem.rdMolTransforms import GetBondLength

# Constants for the elements to check and bond length thresholds
ELEMS_TO_CHECK = ["C", "H", "N", "O", "F", "Cl"]
NON_IR_BOND_THRESHOLD = 2.0
IR_BOND_THRESHOLD = 2.5


def bond_lengths(mol):
    """
    Calculate the maximum bond lengths for Ir and non-Ir bonds in a molecule.

    Args:
    mol (RDKit Mol): The molecule to check.

    Returns:
    tuple: max non-Ir bond length, max Ir bond length
    """
    max_nonir_bond_length = 0
    max_ir_bond_length = 0

    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Molecule has no conformers and bond lengths cannot be calculated."
        )

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        start_elem, end_elem = (
            bond.GetBeginAtom().GetSymbol(),
            bond.GetEndAtom().GetSymbol(),
        )

        bond_length = GetBondLength(mol.GetConformer(0), i, j)

        if start_elem in ELEMS_TO_CHECK and end_elem in ELEMS_TO_CHECK:
            max_nonir_bond_length = max(bond_length, max_nonir_bond_length)
        
        if start_elem == "Ir" or end_elem == "Ir":
            max_ir_bond_length = max(bond_length, max_ir_bond_length)

    return max_nonir_bond_length, max_ir_bond_length


def bonds_maintained(mol):
    """
    Check if all bond lengths are within acceptable thresholds for a molecule.

    Args:
    mol (RDKit Mol): The molecule to check.

    Returns:
    bool: True if all bonds are maintained within thresholds, else False.
    """
    max_nonir_bond_length, max_ir_bond_length = bond_lengths(mol)
    return (
        max_nonir_bond_length < NON_IR_BOND_THRESHOLD
        and max_ir_bond_length < IR_BOND_THRESHOLD
    )
