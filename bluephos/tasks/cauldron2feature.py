import pandas as pd
import os
from rdkit.Chem import AddHs, MolFromMolBlock
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data
import torch as t


# Define a class to encapsulate molecule-related information and operations
class Molecule:
    default_names = {
        "mol_id",
        "atom_id",
        "symbol",
        "x",
        "y",
        "z",
        "bond_id",
        "start_atom",
        "end_atom",
        "formal_charge",
        "n_hydrogen",
        "bond_type",
    }

    def __init__(
        self,
        molecule_table=None,
        one_atom_table=None,
        two_atom_table=None,
        name=None,
        special_colnames=None,
    ):
        self.special_colnames = dict((i, i) for i in self.default_names)
        if special_colnames is not None:
            # Add or replace special column names according to the special_colnames argument
            self.special_colnames.update(special_colnames)

            to_remove = [
                key for key, value in self.special_colnames.items() if value is None
            ]
            for key in to_remove:
                del self.special_colnames[key]
        if molecule_table is not None:
            assert molecule_table.shape[0] == 1
            self.molecule_table = molecule_table
        else:
            self.molecule_table = None

        if name is not None:
            self.name = str(name)
        elif (
            molecule_table is not None
            and self.special_colnames["mol_id"] in molecule_table.columns
        ):
            mol_id_col = molecule_table[self.special_colnames["mol_id"]]
            self.name = str(mol_id_col.item())
        else:
            self.name = None
        self.one_atom_table = one_atom_table
        self.two_atom_table = two_atom_table

    # Retrieving the tables
    def get_molecule_table(self):
        return self.molecule_table

    def get_one_atom_table(self):
        return self.one_atom_table

    def get_two_atom_table(self):
        return self.two_atom_table

    # Standard properties
    def get_name(self):
        return self.name

    def _derive_name_from_table(molecule_table):
        if molecule_table is not None and "mol_id" in molecule_table.columns:
            return str(molecule_table["mol_id"].iloc[0])
        return None

    def get_torch_geom_data(self):
        return tables_to_torch_geom_data(
            self.name, self.molecule_table, self.one_atom_table, self.two_atom_table
        )


# Function to convert molecule tables to PyTorch Geometric Data
def tables_to_torch_geom_data(mol_id, molecule_table, atom_table, bond_table):
    data = Data()
    data.mol_id = mol_id

    if molecule_table is not None:
        molecule_properties = molecule_table.drop(columns=["mol_id"]).to_numpy()
        data.y = t.tensor(molecule_properties, dtype=t.float32)

    if atom_table is not None:
        atom_properties = atom_table.drop(columns=["mol_id", "atom_id"]).to_numpy()
        data.x = t.tensor(atom_properties, dtype=t.float32)

    if bond_table is not None:
        bond_table = pd.concat(
            [
                bond_table,
                bond_table.rename(
                    columns={"start_atom": "end_atom", "end_atom": "start_atom"}
                ),
            ]
        )
        edges = bond_table[["start_atom", "end_atom"]].to_numpy().T
        data.edge_index = t.tensor(edges, dtype=t.long)

    return data


# Utility function for setting RDKit properties based on the type of the value
def set_rdkit_prop(obj, dtype, name, value):
    if dtype.kind == "b":
        obj.SetBoolProp(name, value)
    elif dtype.kind == "i":
        obj.SetIntProp(name, value)
    elif dtype.kind == "u":
        # RDKit does not directly support unsigned int properties, treat as int
        obj.SetIntProp(name, int(value))
    elif dtype.kind == "f":
        obj.SetDoubleProp(name, value)
    elif dtype.kind == "S" or dtype.kind == "U":
        obj.SetProp(name, value)
    elif dtype.kind == "O":
        obj.SetProp(name, str(value))
    else:
        raise ValueError(f"dtype.kind {dtype.kind} not recognized")


def add_default_props(mol: Mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("atom_id", str(i))
        atom.SetProp("formal_charge", str(atom.GetFormalCharge()))
        atom.SetProp("n_hydrogen", str(atom.GetTotalNumHs()))
    for i, bond in enumerate(mol.GetBonds()):
        bond.SetProp("bond_id", str(i))
        bond.SetProp("start_atom", str(bond.GetBeginAtomIdx()))
        bond.SetProp("end_atom", str(bond.GetEndAtomIdx()))
        bond.SetProp("bond_type", bond.GetBondType().name)


def tables2data(mol_id, molecule_table=None, atom_table=None, bond_table_oneway=None):
    output = Data()
    output.mol_id = mol_id
    if molecule_table is not None:
        molecule_properties = molecule_table.drop(columns="mol_id")
        molecule_property_value = molecule_properties.to_numpy()
        output.y = t.tensor(molecule_property_value, dtype=t.float32)

    if atom_table is not None:
        atom_table_sorted = atom_table.sort_values("atom_id")
        atom_properties = atom_table_sorted.drop(columns=["mol_id", "atom_id"])
        output.x = t.tensor(atom_properties.to_numpy(), dtype=t.float32)
    if bond_table_oneway is not None:
        bond_table_reversed = bond_table_oneway.rename(
            columns={"start_atom": "end_atom", "end_atom": "start_atom"}
        )
        bond_table_twoway = pd.concat([bond_table_oneway, bond_table_reversed])
        edges = bond_table_twoway[["start_atom", "end_atom"]]
        # 2 by 2*b matrix containing the adjacency list
        output.edge_index = t.tensor(edges.to_numpy().transpose(), dtype=t.long)
        bond_properties = bond_table_twoway.drop(
            columns=["mol_id", "bond_id", "start_atom", "end_atom"]
        )
        if bond_properties.shape[1] != 0:
            # 2*b by j matrix of one-hot bond types
            output.edge_attr = t.tensor(bond_properties.to_numpy(), dtype=t.float32)
    return output


def add_positions(rdkit_mol, conf_id=0):
    for pos_vec, atom in zip(
        list(rdkit_mol.GetConformer(conf_id).GetPositions()), rdkit_mol.GetAtoms()
    ):
        atom.SetDoubleProp("x", pos_vec[0])
        atom.SetDoubleProp("y", pos_vec[1])
        atom.SetDoubleProp("z", pos_vec[2])


# Convert RDKit molecule to Molecule class instance
def rdkit_to_cauldronoid(rdkit_mol):

    add_default_props(rdkit_mol)

    if rdkit_mol.GetNumConformers() > 0:
        add_positions(rdkit_mol)

    for atom in rdkit_mol.GetAtoms():
        atom.SetProp("symbol", atom.GetSymbol())

    name = (
        rdkit_mol.GetProp("_Name")
        if rdkit_mol.HasProp("_Name") and rdkit_mol.GetProp("_Name") != ""
        else None
    )
    molecule_data = pd.DataFrame(
        [rdkit_mol.GetPropsAsDict(includePrivate=False, includeComputed=False)]
    )
    atom_data = pd.DataFrame(
        [
            atom.GetPropsAsDict(includePrivate=False, includeComputed=False)
            for atom in rdkit_mol.GetAtoms()
        ]
    )
    bond_data = pd.DataFrame(
        [
            bond.GetPropsAsDict(includePrivate=False, includeComputed=False)
            for bond in rdkit_mol.GetBonds()
        ]
    )
    return Molecule(molecule_data, atom_data, bond_data, name)


# Helper function to combine multiple Molecule instances into tables
def bind_rows_map(table_fun, name_fun, id_colname, xs):

    names = map(name_fun, xs)
    tables = map(table_fun, xs)
    return (
        pd.concat(tables, axis=0, keys=names)
        .reset_index(level=0)
        .rename(columns={"level_0": id_colname})
    )


def mols_to_tables(mols):
    mol_id_colname = mols[0].special_colnames["mol_id"]
    assert all(mol.special_colnames["mol_id"] == mol_id_colname for mol in mols)
    molecule_table = bind_rows_map(
        Molecule.get_molecule_table, Molecule.get_name, mol_id_colname, mols
    )

    one_atom_table = bind_rows_map(
        Molecule.get_one_atom_table, Molecule.get_name, mol_id_colname, mols
    )

    two_atom_table = bind_rows_map(
        Molecule.get_two_atom_table, Molecule.get_name, mol_id_colname, mols
    )
    return molecule_table, one_atom_table, two_atom_table


# Function to create Molecule instances from tables
def tables2mols(molecule_table, one_atom_table, two_atom_table, special_colnames=None):
    if special_colnames is None:
        mol_id_colname = "mol_id"
    else:
        mol_id_colname = special_colnames["mol_id"]
    mol_ids = set(molecule_table[mol_id_colname])
    molecule_dict = dict(iter(molecule_table.groupby(mol_id_colname)))
    one_atom_dict = dict(iter(one_atom_table.groupby(mol_id_colname)))
    two_atom_dict = dict(iter(two_atom_table.groupby(mol_id_colname)))
    return [
        Molecule(molecule_dict[mol_id], one_atom_dict[mol_id], two_atom_dict[mol_id])
        for mol_id in mol_ids
    ]


def join_features(
    molecule_table, one_atom_table, two_atom_table, para_folder, element_feature, train_stats
) -> pd.DataFrame:
    element_features = pd.read_csv(os.path.join(para_folder, element_feature))
    train_stats = pd.read_csv(os.path.join(para_folder, train_stats))

    # Join in element features for atoms
    synthetic_one_atom = (
        one_atom_table[["mol_id", "atom_id", "symbol"]]
        .merge(element_features, on="symbol", how="left")
        .drop(columns=["symbol"])
    )

    synthetic_two_atom = two_atom_table[["mol_id", "bond_id", "start_atom", "end_atom"]]

    # Scale features
    for feature in train_stats["feature"]:
        synthetic_one_atom[feature] = (
            synthetic_one_atom[feature]
            - train_stats.loc[train_stats["feature"] == feature, "center"].values[0]
        ) / train_stats.loc[train_stats["feature"] == feature, "scale"].values[0]

    mol_features = tables2mols(molecule_table, synthetic_one_atom, synthetic_two_atom)

    feature_df = pd.DataFrame(mol_features, columns=["Molecule"])
    return feature_df


# Main function to create features from molecule data
def feature_create(mol_df: pd.DataFrame, para_folder, element_feature, train_stats) -> pd.DataFrame:

    # Convert the structures in mol_df to RDKit molecules and then to Molecule instances
    rdkit_mols = mol_df["structure"].apply(MolFromMolBlock).apply(AddHs)
    cauldron = [rdkit_to_cauldronoid(mol) for mol in rdkit_mols]

    # Extract tables from Molecule instances
    molecule_table, one_atom_table, two_atom_table = mols_to_tables(cauldron)

    # Process features and return the result DataFrame
    result_df = join_features(
        molecule_table, one_atom_table, two_atom_table, para_folder, element_feature, train_stats
    )

    return result_df
