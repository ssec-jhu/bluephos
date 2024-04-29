import os
import string
import tempfile
import random
import numpy as np
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Geometry.rdGeometry import Point3D
from ase import Atoms
from ase.optimize import BFGS


def rdkit_to_ase(mol_rdkit, conformation_index, charge, uhf):
    """Convert an RDKit molecule to an ASE Atoms object.

    Args:
        mol_rdkit (Mol): RDKit molecule object.
        conformation_index (int): Index of the conformation to use.
        charge (float): Total molecular charge.
        spin_multiplicity (float): Total spin multiplicity.

    Returns:
        Atoms: ASE Atoms object initialized with element symbols and positions.
    """
    elements = [atom.GetSymbol() for atom in mol_rdkit.GetAtoms()]
    positions = mol_rdkit.GetConformer(conformation_index).GetPositions()
    mol_ase = Atoms(elements, positions)

    # XTB uses total charge and total magnetic moment to decide charge and multiplicity
    # Assign initial charges to have the right sum
    num_atoms = mol_ase.get_global_number_of_atoms()
    mol_ase.set_initial_charges(np.array([charge] + [0.0] * (num_atoms - 1)))
    mol_ase.set_initial_magnetic_moments(np.array([uhf] + [0.0] * (num_atoms - 1)))

    return mol_ase


def annotate_molecule_property(
    mol_rdkit,
    property_name,
    ase_calculator,
    property_function,
    conformation_index=0,
    charge=0,
    uhf=0,
):
    """Annotate an RDKit molecule with a calculated property using ASE.

    Args:
        mol_rdkit (Mol): RDKit molecule object.
        property_name (str): Name of the property to annotate.
        calculator (object): ASE calculator for computing properties.
        property_func (function): Function that computes the property using ASE.
        conformation_index (int, optional): Conformation index. Defaults to 0.
        charge (int, optional): Molecular charge. Defaults to 0.
        uhf (int, optional): Unpaired electrons. Defaults to 0.
    """
    ase_molecule = rdkit_to_ase(mol_rdkit, conformation_index, charge, uhf)
    ase_molecule.calc = ase_calculator
    property_value = property_function(ase_molecule)
    mol_rdkit.SetDoubleProp(property_name, property_value)


def annotate_atom_property(
    mol_rdkit,
    property_name,
    ase_calculator,
    property_function,
    conformation_index=0,
    charge=0,
    uhf=0,
):
    """Annotate an RDKit molecule with atomic properties calculated using ASE.

    Args:
        mol_rdkit (Mol): RDKit molecule object.
        property_name (str): Name of the atomic property to annotate.
        calculator (object): ASE calculator for computing properties.
        property_func (function): Function that computes properties for each atom using ASE.
        conformation_index (int, optional): Conformation index. Defaults to 0.
        charge (int, optional): Molecular charge. Defaults to 0.
        uhf (int, optional): Unpaired electrons. Defaults to 0.
    """
    ase_molecule = rdkit_to_ase(mol_rdkit, conformation_index, charge, uhf)
    ase_molecule.calc = ase_calculator
    property_values = property_function(ase_molecule)

    for atom, property_value in zip(mol_rdkit.GetAtoms(), property_values):
        atom.SetDoubleProp(property_name, property_value)


def optimize_geometry(
    mol_rdkit,
    ase_calculator,
    conformation_index=None,
    constraints=None,
    charge=0,
    uhf=0,
    fmax=0.05,
):
    """Optimize the geometry of an RDKit molecule using an ASE calculator.

    Args:
        molecule (Mol): RDKit molecule.
        calculator (object): ASE calculator for geometry optimization.
        conformation_index (int, optional): Index of the conformation to optimize. Defaults to None.
        constraints (list, optional): Constraints to apply during optimization. Defaults to None.
        charge (int, optional): Molecular charge. Defaults to 0.
        uhf (int, optional): Unpaired electrons. Defaults to 0.
        fmax (float, optional): Maximum force convergence criterion. Defaults to 0.05.

    Returns:
        int: Index of the optimized conformation.

    Raises:
        ValueError: If conformation generation fails.
    """

    if conformation_index is None:
        # Generate initial conformer
        mol_rdkit.RemoveAllConformers()
        conformation_index = EmbedMolecule(mol_rdkit)

    if conformation_index != -1:
        """
        Create a random sequence of characters for the temp files. This way, jobs
        running on different threads don't try to access the same file
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            noise = "".join(random.choices(string.ascii_uppercase + string.digits, k=20))
            # Name of the trajectory file
            traj_filename = os.path.join(tmp_dir, f"tmp_opt_{noise}.traj")

            # Optimize the geometry
            mol_opt_ase = rdkit_to_ase(mol_rdkit, conformation_index, charge, uhf)
            if constraints is not None:
                for constraint in constraints:
                    mol_opt_ase.set_constraint(constraint)
            mol_opt_ase.calc = ase_calculator
            opt = BFGS(mol_opt_ase, trajectory=traj_filename, logfile=None)
            opt.run(fmax=fmax)

        # Set the optimized geometry as the conformer
        positions = mol_opt_ase.get_positions()
        target_conformer = mol_rdkit.GetConformer(conformation_index)
        for i, row in enumerate(positions):
            target_conformer.SetAtomPosition(i, Point3D(*row[:3]))

    else:
        raise ValueError("Failed to generate conformation")

    return conformation_index
