__doc__ = """"BluePhos Discovery Pipeline"""

from pathlib import Path
import pandas as pd
import numpy as np
from dplutils import cli
from dplutils.pipeline.ray import RayStreamGraphExecutor
from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.smiles2sdf import Smiles2SDFTask
from bluephos.tasks.dft import DFTTask


def ligand_pair_generator(halides_file, acids_file):
    """
    Generate ligand pairs from halides and acids files.

    Args:
        halides_file (str): Path to the CSV file containing halides data.
        acids_file (str): Path to the CSV file containing acids data.

    Yields:
        DataFrame: A DataFrame with a single row representing a ligand pair.
    """
    halides_df = pd.read_csv(halides_file)
    acids_df = pd.read_csv(acids_file)
    for _, halide in halides_df.iterrows():
        for _, acid in acids_df.iterrows():
            ligand_pair = {
                "halide_identifier": halide["halide_identifier"],
                "halide_SMILES": halide["halide_SMILES"],
                "acid_identifier": acid["acid_identifier"],
                "acid_SMILES": acid["acid_SMILES"],
            }
            yield pd.DataFrame([ligand_pair])


def rerun_candidate_generator(input_dir, t_nn, t_ste):
    """
    Generates candidate DataFrames from parquet files in the input directory.

    Core Algorithm:
    - If the absolute value of 'z' is less than t_nn,
    - and 'ste' is None or its absolute value is less than t_ste,
    - and 'dft_energy_diff' is None,
    This row is then added to a new DataFrame and yielded for re-run.

    Additional Context:
    1. All valid ligand pairs should already have run through the NN process and have a 'z' score.
    2. If a row's 'ste' is None, then it's 'dft_energy_diff' should also be None.

    Args:
        input_dir (str): Directory containing input parquet files.
        t_nn (float): Threshold for 'z' score.
        t_ste (float): Threshold for 'ste'.

    Yields:
        DataFrame: A single-row DataFrame containing candidate data.
    """
    for file in Path(input_dir).glob("*.parquet"):
        print(file.name)
        df = pd.read_parquet(file)
        
        df["ste"] = df["ste"].replace({None: np.nan})

        filtered = df[
            (df["z"].notnull())
            & (df["z"].abs() < t_nn)
            & ((df["ste"].isnull()) | (df["ste"].abs() < t_ste))
            & (df["dft_energy_diff"].isna())
        ]
        for _, row in filtered.iterrows():
            yield row.to_frame().transpose()


def ligand_smiles_reader_generator(ligand_smiles):
    ligand_df = pd.read_csv(ligand_smiles)
    for _, row in ligand_df.iterrows():
        yield row.to_frame().transpose()


def get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste):
    """
    Get the appropriate generator based on the input directory presence.
    """
    if ligand_smiles:
        return lambda: ligand_smiles_reader_generator(ligand_smiles)
    elif not input_dir:
        return lambda: ligand_pair_generator(halides, acids)
    return lambda: rerun_candidate_generator(input_dir, t_nn, t_ste)


def get_pipeline(
    ligand_smiles,  # Path to the ligands CSV file
    halides,  # Path to the halides CSV file
    acids,  # Path to the acids CSV file
    element_features,  # Path to the element features file
    train_stats,  # Path to the train stats file
    model_weights,  # Path to the model weights file
    input_dir=None,  # Directory containing input parquet files(rerun). Defaults to None.
    dft_package="orca",  # DFT package to use. Defaults to "orca".
    t_nn=1.5,  # Threshold for 'z' score. Defaults to None
    t_ste=1.9,  # Threshold for 'ste'. Defaults to None
):
    """
    Set up and return the BluePhos discovery pipeline executor
    Returns:
        RayStreamGraphExecutor: An executor for the BluePhos discovery pipeline
    """
    steps = (
        [
            GenerateLigandTableTask,
            Smiles2SDFTask,
            NNTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]
        if not (input_dir or ligand_smiles)  # input as halides and acids CSV files
        else [
            NNTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]
        if not ligand_smiles  # input as parquet files
        else [
            Smiles2SDFTask,
            NNTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]  # input as ligand smiles CSV file
    )
    generator = get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste)
    pipeline_executor = RayStreamGraphExecutor(graph=steps, generator=generator)

    context_dict = {
        "ligand_smiles": ligand_smiles,
        "halides": halides,
        "acids": acids,
        "element_features": element_features,
        "train_stats": train_stats,
        "model_weights": model_weights,
        "dft_package": dft_package,
        "t_nn": t_nn,
        "t_ste": t_ste,
    }

    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


if __name__ == "__main__":
    ap = cli.get_argparser(description=__doc__)
    ap.add_argument("--ligand_smiles", required=False, help="CSV file containing ligand SMILES data")
    ap.add_argument("--halides", required=False, help="CSV file containing halides data")
    ap.add_argument("--acids", required=False, help="CSV file containing boronic acids data")
    ap.add_argument("--features", required=True, help="Element feature file")
    ap.add_argument("--train", required=True, help="Train stats file")
    ap.add_argument("--weights", required=True, help="Full energy model weights")
    ap.add_argument("--input_dir", required=False, help="Directory containing input parquet files")
    ap.add_argument("--t_nn", type=float, required=False, default=1.5, help="Threshold for 'z' score (default: 1.5)")
    ap.add_argument("--t_ste", type=float, required=False, default=1.9, help="Threshold for 'ste' (default: 1.9)")

    ap.add_argument(
        "--dft_package",
        required=False,
        default="orca",
        choices=["orca", "ase"],
        help="DFT package to use (default: orca)",
    )
    args = ap.parse_args()

    # Run the pipeline with the provided arguments
    cli.cli_run(
        get_pipeline(
            args.ligand_smiles,
            args.halides,
            args.acids,
            args.features,
            args.train,
            args.weights,
            args.input_dir,
            args.dft_package,
            args.t_nn,
            args.t_ste,
        ),
        args,
    )
