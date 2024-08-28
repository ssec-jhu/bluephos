import pandas as pd
import bluephos.modules.log_config as log_config
from dplutils.pipeline import PipelineTask
from rdkit.Chem import AllChem, MolFromSmiles, MolToSmiles

# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)


def suzuki_couple(boronic_acid_smiles, halide_smiles):
    # Define SMILES strings
    boronic_acid = MolFromSmiles(boronic_acid_smiles)
    halide = MolFromSmiles(halide_smiles)

    # Define Suzuki coupling reaction using reaction SMARTS
    suzuki_coupling_rxn_smarts = (
        "[n:11][c:1]([X:4])[a:10].[c:2][B:5]([O:6])[O:7]>>[n:11][c:1]([c:2])[a:10].[B:5][X:4][O:6][O:7]"
    )
    suzuki_coupling_rxn = AllChem.ReactionFromSmarts(suzuki_coupling_rxn_smarts)

    # Perform the reaction
    product_set = suzuki_coupling_rxn.RunReactants((halide, boronic_acid))

    # Collect the products and convert them to SMILES
    products = []
    for product in product_set:
        products.append(MolToSmiles(product[0], isomericSmiles=True))

    return set(products)


def add_dummy_atoms(smiles):
    mol = MolFromSmiles(smiles)

    # Define the reaction SMARTS to identify the Ir bonding sites
    rxn_smarts = "[c:1][c:2]-[c:3][n:4]>>*[c:1][c:2]-[c:3][n:4]->*"
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    # Apply the reaction to the molecule
    products = rxn.RunReactants((mol,))

    return list(set([MolToSmiles(x[0]) for x in products]))


def generate_ligand_table(
    ligand_pair_df: pd.DataFrame,
) -> pd.DataFrame:
    result_data = []  # List to store the output data

    for _, ligand_pair in ligand_pair_df.iterrows():
        ligand_set = suzuki_couple(ligand_pair["acid_SMILES"], ligand_pair["halide_SMILES"])

        ligands_with_binding_sites = []
        for ligand_smiles in ligand_set:
            ligands_with_binding_sites = ligands_with_binding_sites + add_dummy_atoms(ligand_smiles)

        # If the set is empty, do nothing
        if not ligands_with_binding_sites:
            continue

        # Convert each ligand in the set to a row in the DataFrame and append to the list
        for i, ligand in enumerate(ligands_with_binding_sites):
            ligand_identifier = ligand_pair["halide_identifier"] + ligand_pair["acid_identifier"] + "L" + hex(i)[2:]
            result_data.append(
                {
                    "ligand_identifier": ligand_identifier,
                    "ligand_SMILES": ligand,
                    "halide_identifier": ligand_pair["halide_identifier"],
                    "halide_SMILES": ligand_pair["halide_SMILES"],
                    "acid_identifier": ligand_pair["acid_identifier"],
                    "acid_SMILES": ligand_pair["acid_SMILES"],
                }
            )

    logger.info("Ligand Generate Task complete")
    # Create the DataFrame using the list of data
    return pd.DataFrame(result_data)


GenerateLigandTableTask = PipelineTask(
    "generate_ligand_table",
    generate_ligand_table,
    batch_size=1000,
)
