import sys
import numpy as np
import re
import rdkit.Chem

def min_angle(mol):
    coordlist = metal_neighbor_coords(mol)
    distances = []
    for i in range(len(coordlist)):
        for j in range(i, len(coordlist)):
            if i != j:
                distances.append(angle_3d(coordlist[i], coordlist[j]))
    distances.sort()
    #return np.max(distances[:12])-np.min(distances[:12])
    return np.min(distances[:12])

def dist_3d(a,b):
    return np.sqrt(np.sum((np.array(a)-np.array(b))**2, axis = 0))
    
def angle_3d(a,b):
    v1_u = unit_vector(a)
    v2_u = unit_vector(b)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
    
def metal_neighbor_coords(mol, metal = "Ir", conformer_index = 0):
    coordlist = []
    indices_seen = set()
    conformer = mol.GetConformer(conformer_index)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == metal:
            center = conformer.GetAtomPosition(atom.GetIdx())
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                # This if probably not necessary, there shouldn't be duplicates
                if neighbor_index not in indices_seen:
                    point = conformer.GetAtomPosition(neighbor_index)
                    coordlist.append([point.x - center.x, point.y - center.y, point.z - center.z])
                    indices_seen.add(neighbor_index)
    return coordlist

# Boolean function based on an arbitrary cutoff for angle
def isoctahedral(mol):
    return min_angle(mol) > 70
    
if __name__ == "__main__":
    with open(sys.argv[1], "r") as sdf_file:
        print("loading sdf src")
        sdf_text = sdf_file.read()
        sdf_list = sdf_text.split("$$$$\n")
        print("extracting names")
        with rdkit.Chem.rdmolfiles.SDWriter("trainset.sdf") as sdf:
            sdf.SetKekulize(False)
            for i in sdf_list[:-1]:
                name = i.strip().split()[0].strip()
                mol = rdkit.Chem.rdmolfiles.MolFromMolBlock(i, sanitize=False)
                if "Ir" in [i.GetSymbol() for i in mol.GetAtoms()]:
                    print(name, isoctahedral(mol), sep = ",")
                    if isoctahedral(mol):
                        continue
                else:
                    print(name, "NA", sep = ",")
                sdf.write(mol)
