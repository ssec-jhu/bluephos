__doc__ = """"BluePhos Discovery Pipeline"""

from pathlib import Path

import numpy as np
import pandas as pd
from dplutils import cli
from dplutils.pipeline.ray import RayStreamGraphExecutor
from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.readmol2 import Smiles2SDFTask
from bluephos.tasks.dft import DFTTask
from bluephos.tasks.filter_pipeline import (
    FilterDFTInTask,
    FilterDFTOutTask,
    FilterNNInTask,
    FilterNNOutTask,
    FilterXTBInTask,
    FilterXTBOutTask,
)
from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.smiles2sdf import Smiles2SDFTask


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


def check_column(df, column_name, condition_func, default=True):
    """
    Helper function to check if a column exists and apply a condition.

    """
    if column_name not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    else:
        return condition_func(df[column_name])


def rerun_candidate_generator(input_dir, t_nn, t_ste, t_dft):
    """
    Generates candidate DataFrames from parquet files in the input directory.

    Core Algorithm:
    - A row is selected for re-run if:
        1. 'z' is None or its absolute value is less than t_nn,
        2. and 'ste' is None or its absolute value is less than t_ste,
        3. and 'dft_energy_diff' is None.

    Args:
        input_dir (str): Directory containing input parquet files.
        t_nn (float): Threshold for 'z' score.
        t_ste (float): Threshold for 'ste'.
        t_dft (float): (Optional) Threshold for 'dft_energy_diff' (not currently used)

    Yields:
        DataFrame: A single-row DataFrame containing candidate data.
    """
    for file in Path(input_dir).glob("*.parquet"):
        df = pd.read_parquet(file)
        df["ste"] = df["ste"].replace({None: np.nan})

        filtered = df[
            check_column(df, "z", lambda col: col.isnull() | (col.abs() < t_nn))
            & check_column(df, "ste", lambda col: col.isnull() | (col.abs() < t_ste))
            & check_column(df, "dft_energy_diff", lambda col: col.isna())
        ]
        for _, row in filtered.iterrows():
            yield row.to_frame().transpose()


def ligand_smiles_reader_generator(ligand_smiles):
    ligand_df = pd.read_csv(ligand_smiles)
    for _, row in ligand_df.iterrows():
        yield row.to_frame().transpose()


def get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste, t_dft):
    """
    Get the appropriate generator based on the input directory presence.
    """
    if ligand_smiles:
        return lambda: ligand_smiles_reader_generator(ligand_smiles)
    elif not input_dir:
        return lambda: ligand_pair_generator(halides, acids)
    return lambda: rerun_candidate_generator(input_dir, t_nn, t_ste, t_dft)


def build_pipeline_graph(input_dir: str, ligand_smiles: str):
    """
    Construct the pipeline graph based on input conditions.

    Args:
        input_dir (str): Directory containing input parquet files.
        ligand_smiles (str): Path to the ligand SMILES CSV file.

    Returns:
        list: A list of task tuples representing the pipeline graph.
    """
    full_pipeline = [
        (GenerateLigandTableTask, Smiles2SDFTask),  # Generate ligands, then convert SMILES to SDF
        (Smiles2SDFTask, NNTask),  # Use SMILES to run NN prediction
        (NNTask, FilterNNOutTask),  # NN filter "out" goes to sink
        (NNTask, FilterNNInTask),  # NN filter "in" continues to the next task
        (FilterNNInTask, OptimizeGeometriesTask),  # Optimize geometries for filtered ligands
        (OptimizeGeometriesTask, FilterXTBOutTask),  # XTB filter "out" goes to sink
        (OptimizeGeometriesTask, FilterXTBInTask),  # XTB filter "in" continues to the next task
        (FilterXTBInTask, DFTTask),  # Run DFT calculation for filtered ligands
        (DFTTask, FilterDFTOutTask),  # DFT filter "out" goes to sink
        (DFTTask, FilterDFTInTask),  # DFT filter "in" could be processed further
    ]

    if ligand_smiles:
        return full_pipeline[1:]  # from NNTask (Case 1: Input as ligand SMILES CSV file)

    if input_dir:
        return full_pipeline[2:]  # from Smiles2SDFTask (Case 2: Use parquet files for rerun)

    return full_pipeline  # from GenerateLigandTableTask (Case 3: Input as halides and acids CSV files)


def get_pipeline(
    ligand_smiles,  # Path to the ligands CSV file
    halides,  # Path to the halides CSV file
    acids,  # Path to the acids CSV file
    element_features,  # Path to the element features file
    train_stats,  # Path to the train stats file
    model_weights,  # Path to the model weights file
    input_dir=None,  # Directory containing input parquet files(rerun). Defaults to None.
    dft_package="orca",  # DFT package to use. Defaults to "orca".
    xtb=True,  # Enable xTb optimize geometries task. Defaults to True.
    t_nn=1.5,  # Threshold for 'z' score.
    t_ste=1.9,  # Threshold for 'ste'.
    t_dft=2.5,  # Threshold for 'dft'.
):
    """
    Set up and return the BluePhos discovery pipeline executor
    Returns:
        RayStreamGraphExecutor: An executor for the BluePhos discovery pipeline
    """

    generator = get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste, t_dft)
    pipeline_graph = build_pipeline_graph(input_dir, ligand_smiles)
    pipeline_executor = RayStreamGraphExecutor(graph=pipeline_graph, generator=generator)

    context_dict = {
        "ligand_smiles": ligand_smiles,
        "halides": halides,
        "acids": acids,
        "element_features": element_features,
        "train_stats": train_stats,
        "model_weights": model_weights,
        "dft_package": dft_package,
        "xtb": xtb,
        "t_nn": t_nn,
        "t_ste": t_ste,
        "t_dft": t_dft,
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
    ap.add_argument("--t_dft", type=float, required=False, default=2.5, help="Threshold for 'dft' (default: 2.5)")
    ap.add_argument(
        "--no_xtb", action="store_false", dest="xtb", default=True, help="Disable xTB optimization (default: enabled)"
    )

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
            args.xtb,
            args.t_nn,
            args.t_ste,
            args.t_dft,
        ),
        args,
    )
