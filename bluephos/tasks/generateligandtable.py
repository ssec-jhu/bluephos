import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdqueries
import os
from pathlib import Path
from dplutils.pipeline import PipelineTask


def suzuki_couple(boronic_acid_smiles, halide_smiles):
    # Define SMILES strings
    boronic_acid = Chem.MolFromSmiles(boronic_acid_smiles)
    halide = Chem.MolFromSmiles(halide_smiles)

    # Define Suzuki coupling reaction using reaction SMARTS
    suzuki_coupling_rxn_smarts = "[n:11][c:1]([X:4])[a:10].[c:2][B:5]([O:6])[O:7]>>[n:11][c:1]([c:2])[a:10].[B:5][X:4][O:6][O:7]"
    suzuki_coupling_rxn = AllChem.ReactionFromSmarts(suzuki_coupling_rxn_smarts)

    # Perform the reaction
    product_set = suzuki_coupling_rxn.RunReactants((halide, boronic_acid))

    # Collect the products and convert them to SMILES
    products = []
    for product in product_set:
        products.append(Chem.MolToSmiles(product[0], isomericSmiles=True))

    return set(products)


def add_dummy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # Define the reaction SMARTS to identify the Ir bonding sites
    rxn_smarts = "[c:1][c:2]-[c:3][n:4]>>*[c:1][c:2]-[c:3][n:4]->*"
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    # Apply the reaction to the molecule
    products = rxn.RunReactants((mol,))

    return list(set([Chem.MolToSmiles(x[0]) for x in products]))


def generate_ligand_table(
    inputs: pd.DataFrame,
    halides_file_name,
    acids_file_name,
    input_folder,
    output_folder,
) -> pd.DataFrame:

    halide_df = pd.read_csv(os.path.join(input_folder, halides_file_name))
    acid_df = pd.read_csv(os.path.join(input_folder, acids_file_name))

    print("read halide and acid successfully")

    result_data = []  # List to store the output data

    for _, halide_row in tqdm(halide_df.iterrows(), total=len(halide_df)):
        for _, acid_row in acid_df.iterrows():
            ligand_set = suzuki_couple(
                acid_row["acid_SMILES"], halide_row["halide_SMILES"]
            )

            ligands_with_binding_sites = []
            for ligand_smiles in ligand_set:
                ligands_with_binding_sites = (
                    ligands_with_binding_sites + add_dummy_atoms(ligand_smiles)
                )

            # If the set is empty, do nothing
            if not ligands_with_binding_sites:
                continue

            # Convert each ligand in the set to a row in the DataFrame and append to the list
            for i, ligand in enumerate(ligands_with_binding_sites):
                ligand_identifier = (
                    halide_row["halide_identifier"]
                    + acid_row["acid_identifier"]
                    + "L"
                    + hex(i)[2:]
                )
                result_data.append(
                    {
                        "ligand_identifier": ligand_identifier,
                        "ligand_SMILES": ligand,
                        "halide_identifier": halide_row["halide_identifier"],
                        "halide_SMILES": halide_row["halide_SMILES"],
                        "acid_identifier": acid_row["acid_identifier"],
                        "acid_SMILES": acid_row["acid_SMILES"],
                    }
                )

    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Create the DataFrame using the list of data
    result_df = pd.DataFrame(result_data)

    result_df.to_csv(
        os.path.join(output_folder, "combinatorial_ligands.csv"), index=False
    )

    return result_df


GenerateLigandTableTask = PipelineTask(
    "generate_ligand_table",
    generate_ligand_table,
    context_kwargs={
        "halides_file_name": "halides_file_name",
        "acids_file_name": "acids_file_name",
        "input_folder": "input_folder",
        "output_folder": "output_folder",
    },
)
