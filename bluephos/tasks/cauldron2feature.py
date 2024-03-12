import pandas as pd
import os
from tasks.cauldronoid import rdkit2cauldronoid, mols2files, mols2tables, tables2mols
from rdkit.Chem import AddHs
from dplutils.pipeline import PipelineTask


def make_cauldron(df_mols: pd.DataFrame) -> dict:

    # Get RDKit Mol objects
    rdkit_mols = df_mols["molecules"]

    # Convert RDKit Mol objects to cauldronoid objects
    cauldron = [rdkit2cauldronoid(AddHs(rdkit_mol)) for rdkit_mol in rdkit_mols]

    # Use cauldronoid to create files from the processed molecule objects
    mols2files(cauldron, "pipeline_example")

    # Convert Cauldronoid objects back to tables
    molecule_table, one_atom_table, two_atom_table = mols2tables(cauldron)

    cauldron_df = {
        "molecule_table": molecule_table,
        "one_atom_table": one_atom_table,
        "two_atom_table": two_atom_table,
    }

    return cauldron_df


def join_feature(cauldron_df: dict, input_folder, output_folder) -> pd.DataFrame:

    element_features = pd.read_csv(
        os.path.join(input_folder, "element_features.csv"), dtype={"symbol": str}
    )
    train_stats = pd.read_csv(
        os.path.join(input_folder, "train_stats.csv"), dtype={"feature": str}
    )

    synthetic_mols, synthetic_one_atom, synthetic_two_atom = cauldron_df.values()

    synthetic_mols = cauldron_df["molecule_table"]

    # Join in element features for atoms
    synthetic_one_atom = (
        cauldron_df["one_atom_table"][["mol_id", "atom_id", "symbol"]]
        .merge(element_features, on="symbol", how="left")  # Join in element features
        .drop(columns=["symbol"])  # Remove the 'symbol' column
        # Further processing as required
    )

    synthetic_two_atom = cauldron_df["two_atom_table"][
        ["mol_id", "bond_id", "start_atom", "end_atom"]
    ]

    # Scale features
    for feature in train_stats["feature"]:
        synthetic_one_atom[feature] = (
            synthetic_one_atom[feature]
            - train_stats.loc[train_stats["feature"] == feature, "center"].values[0]
        ) / train_stats.loc[train_stats["feature"] == feature, "scale"].values[0]

    mol_features = tables2mols(synthetic_mols, synthetic_one_atom, synthetic_two_atom)

    feature_df = pd.DataFrame(mol_features, columns=["Molecule"])

    # Write featurized molecule table
    synthetic_mols.to_csv(
        os.path.join(output_folder, "featurized_mol_tbl.csv.gz"), index=False
    )

    # Write featurized atom table
    synthetic_one_atom.to_csv(
        os.path.join(output_folder, "featurized_one_tbl.csv.gz"), index=False
    )

    # Write featurized bond table
    synthetic_two_atom[["mol_id", "bond_id", "start_atom", "end_atom"]].to_csv(
        os.path.join(output_folder, "featurized_two_tbl.csv.gz"), index=False
    )

    return feature_df


def feature_create(mol_df: pd.DataFrame, input_folder, output_folder) -> pd.DataFrame:

    cauldron_tables = make_cauldron(mol_df)

    result_df = join_feature(cauldron_tables, input_folder, output_folder)

    return result_df


Cauldron2FeatureTask = PipelineTask(
    "cauldron2feature",
    feature_create,
    context_kwargs={"input_folder": "input_folder", "output_folder": "output_folder"},
    num_gpus=0,
    batch_size=1,
)
