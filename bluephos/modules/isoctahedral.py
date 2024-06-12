import numpy as np


def min_angle(mol):
    """Calculate the minimum angle between the first 12 nearest neighbors of a metal atom."""
    coordlist = metal_neighbor_coords(mol)

    if len(coordlist) < 2:
        return float("inf")  # Avoid calculation if there are fewer than two neighbors

    distances = []
    for i in range(len(coordlist)):
        for j in range(i + 1, len(coordlist)):
            distances.append(angle_3d(coordlist[i], coordlist[j]))
    distances.sort()
    # return np.max(distances[:12])-np.min(distances[:12])
    return np.min(distances[:12])


def dist_3d(a, b):
    """Calculate the Euclidean distance between two points in 3D space."""
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2, axis=0))


def angle_3d(a, b):
    """Calculate the angle between two vectors in 3D space."""
    v1_u = unit_vector(a)
    v2_u = unit_vector(b)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def recenter_point(point, center):
    """Calculate the relative coordinates from the center to the point."""
    return [point.x - center.x, point.y - center.y, point.z - center.z]


def metal_neighbor_coords(mol, metal="Ir", conformer_index=0):
    """
    Extract coordinates of nearest neighbors of specified metal atoms in a molecule.

    Args:
        mol (rdchem.Mol): RDKit molecule object.
        metal (str): The symbol of the metal atom to find neighbors for, default is "Ir".
        conformer_index (int): The index of the conformer whose coordinates are used, default is 0.

    Returns:
        list: A list of coordinates [dx, dy, dz] relative to each metal atom found.

    Raises:
        ValueError: If the specified conformer index does not exist.
        AttributeError: If the molecule does not have conformers.
    """
    coord_list = []
    indices_seen = set()
    conformer = mol.GetConformer(conformer_index)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == metal:
            center = conformer.GetAtomPosition(atom.GetIdx())

            # Retrieve all neighbor indices for the current metal atom
            neighbor_indexes = set(neighbor.GetIdx() for neighbor in atom.GetNeighbors())

            # Fetch positions of all neighbors using their indices
            # Filter neighbors that have already been processed using indices_seen
            points = [conformer.GetAtomPosition(idx) for idx in neighbor_indexes - indices_seen]

            # Calculate relative coordinates and extend the coord_list
            coord_list.extend(recenter_point(p, center) for p in points)

            # Update the indices_seen set with new indices to prevent reprocessing
            indices_seen.update(neighbor_indexes)

    return coord_list


# Boolean function based on an arbitrary cutoff for angle
def isoctahedral(mol):
    return min_angle(mol) > 70
