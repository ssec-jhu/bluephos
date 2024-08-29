import os.path
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Conformer
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromSmarts
from rdkit.Chem.AllChem import ConstrainedEmbed
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem.rdmolops import RemoveStereochemistry

def make_bonds_dative(mol, target_elem = "Ir"):
    editable_mol = RWMol(mol)

    # If you don't make a list, it loops infinitely over the bonds it's creating
    for bond in list(editable_mol.GetBonds()):
        iridium = None
        nitrogen = None
        carbene = None
        if bond.GetBeginAtom().GetSymbol() == target_elem and \
                bond.GetEndAtom().GetSymbol() in ["N", "P"] and \
                bond.GetEndAtom().GetFormalCharge() == 1:
            iridium = bond.GetBeginAtom()
            nitrogen = bond.GetEndAtom()
            start_idx = bond.GetEndAtomIdx()
            end_idx = bond.GetBeginAtomIdx()
        elif bond.GetEndAtom().GetSymbol() == target_elem and \
                bond.GetBeginAtom().GetSymbol() in ["N", "P"] and \
                bond.GetBeginAtom().GetFormalCharge() == 1:
            iridium = bond.GetEndAtom()
            nitrogen = bond.GetBeginAtom()
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
        if bond.GetBeginAtom().GetSymbol() == target_elem and \
                bond.GetEndAtom().GetSymbol() == "C" and \
                bond.GetEndAtom().GetTotalValence() == 3:
            iridium = bond.GetBeginAtom()
            carbene = bond.GetEndAtom()
            start_idx = bond.GetEndAtomIdx()
            end_idx = bond.GetBeginAtomIdx()
        elif bond.GetEndAtom().GetSymbol() == target_elem and \
                bond.GetBeginAtom().GetSymbol() == "C" and \
                bond.GetBeginAtom().GetTotalValence() == 3:
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

def transfer_conformation(mol, substruct, conformer = 0):
    '''Given a molecule, and a second molecule which is a substructure of the
    first, assign coordinates to the substructure based on the matching part of
    the original molecule'''
    match = mol.GetSubstructMatch(substruct)
    substruct_conformation = Conformer(substruct.GetNumAtoms())
    for i, index in enumerate(match):
        point = mol.GetConformer(conformer).GetAtomPosition(index)
        substruct_conformation.SetAtomPosition(i, point)
    substruct.AddConformer(substruct_conformation)

fac = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "OHUZEW.mol2")))
RemoveStereochemistry(fac)
mer = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "OHUZIA.mol2")))
RemoveStereochemistry(mer)

template = MolFromSmarts("[Ir]1~n:[*]~[*]:c~1")

carbene_fac = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "MAXYIU.mol2")))
RemoveStereochemistry(carbene_fac)
carbene_mer = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "MAXYOA.mol2")))
RemoveStereochemistry(carbene_mer)

# Extract skeletons of a molecule based on a template, keeping coordinates
# Multiple skeletons because I don't know how to do wildcards
def skeleton(template, mol):
    template_matches = mol.GetSubstructMatches(template)
    matching_indices = set(sum(template_matches, ()))
    editable_mol = RWMol(mol)
    editable_mol.BeginBatchEdit()
    for atom in editable_mol.GetAtoms():
        # Remove all atoms except the matching ones
        if atom.GetIdx() not in matching_indices:
            editable_mol.RemoveAtom(atom.GetIdx())
        # All the atoms should have formal charge 0, not sure why they don't
        atom.SetFormalCharge(0)
    editable_mol.CommitBatchEdit()
    skeleton_mol = editable_mol.GetMol()
    return skeleton_mol

fac_skeleton = skeleton(template, fac)
mer_skeleton = skeleton(template, mer)

# Making the carbene skeletons in a completely different way
# I probably am going to want to do this for all of them
carbene_skeleton_smarts = "[Ir]135(<-[CH0](~N(~*)~*~2)~N(~*~2)~c~c~1)(<-[CH0](~N(~*)~*~4)~N(~*~4)~c~c~3)(<-[CH0](~N(~*)~*~6)~N(~*~6)~c~c~5)"
carbene_fac_skeleton = MolFromSmarts(carbene_skeleton_smarts)
transfer_conformation(carbene_fac, carbene_fac_skeleton)
carbene_mer_skeleton = MolFromSmarts(carbene_skeleton_smarts)
transfer_conformation(carbene_mer, carbene_mer_skeleton)

def run_three_times(mol, reaction):
    for i in range(3):
        mol = reaction.RunReactants([mol])[0][0]
    return mol
reactions = [
    ReactionFromSmarts("[Ir:1]1<-[n:2]:[n:3]~[c:4]:[c:5]~1>>[Ir:1]1<-[n:2]:[c:3]~[n:4]:[c:5]~1"),
    ReactionFromSmarts("[Ir:1]1<-[n:2]:[n:3]~[c:4]:[c:5]~1>>[Ir:1]1<-[n:2]:[n:3]~[n:4]:[c:5]~1"),
    ReactionFromSmarts("[Ir:1]1<-[n:2]:[n:3]~[c:4]:[c:5]~1>>[Ir:1]1<-[n:2]:[c:3]~[c:4]:[c:5]~1")
]
fac_skeletons = [fac_skeleton] + [run_three_times(fac_skeleton, reaction) for reaction in reactions] + [carbene_fac_skeleton]
mer_skeletons = [mer_skeleton] + [run_three_times(mer_skeleton, reaction) for reaction in reactions] + [carbene_mer_skeleton]

# Skeletons for tridentate carbenes
# I may have to remake these later if I want to control the isomers
# For now I think it doesn't matter because all the carbene ligands are symmetric?
# For homoleptic:
biplet = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "BIPLET.mol2")))
biplet_skeleton = MolFromSmarts('[Ir]1234(~[#6](~[#7](~[#6])~[#6])~[#7]~c~c~1~c~[#7]~[#6](~[#7](~[#6])~[#6])~2)~[#6](~[#7](~[#6])~[#6])~[#7]~c~c~3~c~[#7]~[#6](~[#7](~[#6])~[#6])~4')
transfer_conformation(biplet, biplet_skeleton)
# For heteroleptic, three with counterligands of different size
soynom = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "SOYNOM.mol2")))
RemoveStereochemistry(soynom)
soynom_skeleton = MolFromSmarts('[Ir]1234(~*~*~*~*~1~*~*~*~2)~[#6](~[#7](~[#6])~[#6])~[#7]~c~c~3~c~[#7]~[#6](~[#7](~[#6])~[#6])~4')
transfer_conformation(soynom, soynom_skeleton)
uyokur = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "UYOKUR.mol2")))
uyokur_skeleton = MolFromSmarts('[Ir]1234(~*~*~*~*~*~1~*~*~*~2)~[#6](~[#7](~[#6])~[#6])~[#7]~c~c~3~c~[#7]~[#6](~[#7](~[#6])~[#6])~4')
transfer_conformation(uyokur, uyokur_skeleton)
egufiz = make_bonds_dative(MolFromMol2File(os.path.join(__path__[0], "EGUFIZ.mol2")))
egufiz_skeleton = MolFromSmarts('[Ir]1234(~*~*~*~*~*~1~*~*~*~*~2)~[#6](~[#7](~[#6])~[#6])~[#7]~c~c~3~c~[#7]~[#6](~[#7](~[#6])~[#6])~4')
transfer_conformation(egufiz, egufiz_skeleton)
# Homoleptic has to go first, since a later pattern can cover it
tridentate_skeletons = [biplet_skeleton, soynom_skeleton, uyokur_skeleton, egufiz_skeleton]

def octahedral_embed(mol, isomer):
    # Needed for some of the mol2 files I got from CSD
    # Will not be able to embed with stereochemistry
    RemoveStereochemistry(mol)
    if isomer == "fac":
        skeletons = fac_skeletons
    elif isomer == "mer":
        skeletons = mer_skeletons
    elif isomer == "tridentate":
        skeletons = tridentate_skeletons
    else:
        raise ValueError(f"Isomer should be \"mer\" or \"fac\", given {isomer}")
    finished = False
    for skeleton in skeletons:
        if len(mol.GetSubstructMatch(skeleton)) > 0:
            # Carbene embedding with a large template gives output "Could not
            # triangle bounds smooth molecule" and raises a ValueError. But
            # with a small template the imidazole is hroribly twisted, probably
            # because it thinks the atoms are aliphatic. Ignoring smoothing
            # failures with the large template, it works
            ConstrainedEmbed(mol, skeleton, ignoreSmoothingFailures = True)
            finished = True
    if not finished:
        raise ValueError("Doesn't match templates")
