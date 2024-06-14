import os.path

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, rdChemReactions, rdmolfiles, rdmolops

CARBENE_SKELETON_SMARTS = (
    "[Ir]135(<-[CH0](~N(~*)~*~2)~N(~*~2)~c~c~1)(<-[CH0](~N(~*)~*~4)~N(~*~4)~c~c~3)(<-[CH0](~N(~*)~*~6)~N(~*~6)~c~c~5)"
)


def make_bonds_dative(mol, target_elem="Ir"):
    """
    Modifies a molecule to change specific bonds to dative based on the element targeting.

    Parameters:
        mol (Mol): The RDKit molecule object to modify.
        target_elem (str): Target element symbol for bond modifications.

    Returns:
        Mol: The modified RDKit molecule with dative bonds where applicable.
    """
    editable_mol = rdchem.RWMol(mol)

    # If you don't make a list, it loops infinitely over the bonds it's creating
    for bond in list(editable_mol.GetBonds()):
        iridium = None
        nitrogen = None
        carbene = None
        if (
            bond.GetBeginAtom().GetSymbol() == target_elem
            and bond.GetEndAtom().GetSymbol() in ["N", "P"]
            and bond.GetEndAtom().GetFormalCharge() == 1
        ):
            iridium = bond.GetBeginAtom()
            nitrogen = bond.GetEndAtom()
            start_idx = bond.GetEndAtomIdx()
            end_idx = bond.GetBeginAtomIdx()
        elif (
            bond.GetEndAtom().GetSymbol() == target_elem
            and bond.GetBeginAtom().GetSymbol() in ["N", "P"]
            and bond.GetBeginAtom().GetFormalCharge() == 1
        ):
            iridium = bond.GetEndAtom()
            nitrogen = bond.GetBeginAtom()
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
        if (
            bond.GetBeginAtom().GetSymbol() == target_elem
            and bond.GetEndAtom().GetSymbol() == "C"
            and bond.GetEndAtom().GetTotalValence() == 3
        ):
            iridium = bond.GetBeginAtom()
            carbene = bond.GetEndAtom()
            start_idx = bond.GetEndAtomIdx()
            end_idx = bond.GetBeginAtomIdx()
        elif (
            bond.GetEndAtom().GetSymbol() == target_elem
            and bond.GetBeginAtom().GetSymbol() == "C"
            and bond.GetBeginAtom().GetTotalValence() == 3
        ):
            iridium = bond.GetEndAtom()
            carbene = bond.GetBeginAtom()
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

        if nitrogen is not None:
            # Replace N+ - Ir with N -> Ir
            nitrogen.SetFormalCharge(0)

        if iridium is not None and (nitrogen is not None or carbene is not None):
            editable_mol.RemoveBond(start_idx, end_idx)
            editable_mol.AddBond(start_idx, end_idx, Chem.rdchem.BondType.DATIVE)

    outmol = editable_mol.GetMol()
    Chem.SanitizeMol(outmol)

    return outmol


def transfer_conformation(mol, substruct, conformer=0):
    """Given a molecule, and a second molecule which is a substructure of the
    first, assign coordinates to the substructure based on the matching part of
    the original molecule"""
    match = mol.GetSubstructMatch(substruct)
    substruct_conformation = rdchem.Conformer(substruct.GetNumAtoms())
    for i, index in enumerate(match):
        point = mol.GetConformer(conformer).GetAtomPosition(index)
        substruct_conformation.SetAtomPosition(i, point)
    substruct.AddConformer(substruct_conformation)


def run_three_times(mol, reaction):
    for i in range(3):
        mol = reaction.RunReactants([mol])[0][0]
    return mol


def get_directory_path():
    """Returns the directory path of the current script."""
    return os.path.dirname(os.path.abspath(__file__))


def skeleton_extraction(mol, template):
    """Extracts and returns the skeleton of a molecule based on the provided SMARTS template."""
    matches = mol.GetSubstructMatches(template)
    matching_indices = set(sum(matches, ()))
    editable_mol = rdchem.RWMol(mol)
    editable_mol.BeginBatchEdit()
    for atom in editable_mol.GetAtoms():
        if atom.GetIdx() not in matching_indices:
            editable_mol.RemoveAtom(atom.GetIdx())
        atom.SetFormalCharge(0)
    editable_mol.CommitBatchEdit()
    return editable_mol.GetMol()


def load_molecule(file_name):
    """Loads a molecule from a .mol2 file given a file name."""
    dir_path = get_directory_path()
    base_mol = rdmolfiles.MolFromMol2File(os.path.join(dir_path, file_name))
    dative_mol = make_bonds_dative(base_mol)
    rdmolops.RemoveStereochemistry(dative_mol)
    return dative_mol


def compute_skeletons(isomer):
    """Computes and returns all skeletons based on the isomer type."""

    if isomer == "fac":
        base_mol = load_molecule("OHUZEW.mol2")
        carbene_mol = load_molecule("MAXYIU.mol2")
    elif isomer == "mer":
        base_mol = load_molecule("OHUZIA.mol2")
        carbene_mol = load_molecule("MAXYOA.mol2")
    else:
        raise ValueError(f'Isomer should be "mer" or "fac", given {isomer}')

    template = rdmolfiles.MolFromSmarts("[Ir]1~n:[*]~[*]:c~1")

    skeleton = skeleton_extraction(base_mol, template)

    carbene_skeleton = rdmolfiles.MolFromSmarts(CARBENE_SKELETON_SMARTS)
    transfer_conformation(carbene_mol, carbene_skeleton)

    REACTION_PREFIX = "[Ir:1]1<-[n:2]:[n:3]~[c:4]:[c:5]~1>>[Ir:1]1<-[n:2]:"
    REACTION_SUFFIX = ":[c:5]~1"

    reactions = [
        rdChemReactions.ReactionFromSmarts(f"{REACTION_PREFIX}{core}{REACTION_SUFFIX}")
        for core in ["[c:3]~[n:4]", "[n:3]~[n:4]", "[c:3]~[c:4]"]
    ]
    skeletons = [skeleton] + [run_three_times(skeleton, reaction) for reaction in reactions] + [carbene_skeleton]

    return skeletons


def octahedral_embed(mol, isomer):
    """Embeds a molecule based on the skeletons for 'fac' or 'mer' isomers."""
    rdmolops.RemoveStereochemistry(mol)
    skeletons = compute_skeletons(isomer)

    finished = False
    for skeleton in skeletons:
        if len(mol.GetSubstructMatch(skeleton)) > 0:
            # Carbene embedding with a large template gives output "Could not
            # triangle bounds smooth molecule" and raises a ValueError. But
            # with a small template the imidazole is hroribly twisted, probably
            # because it thinks the atoms are aliphatic. Ignoring smoothing
            # failures with the large template, it works
            AllChem.ConstrainedEmbed(mol, skeleton, ignoreSmoothingFailures=True)
            finished = True
            # break
    if not finished:
        raise ValueError("Doesn't match templates")
