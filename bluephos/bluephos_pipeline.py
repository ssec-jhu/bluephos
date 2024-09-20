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
from bluephos.tasks.filter_pipeline import (
    FilterNNInTask, FilterNNOutTask,
    FilterXTBInTask, FilterXTBOutTask,
    FilterDFTInTask, FilterDFTOutTask
)


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


def get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste, t_dft):
    """
    Get the appropriate generator based on the input directory presence.
    """
    if ligand_smiles:
        return lambda: ligand_smiles_reader_generator(ligand_smiles)
    elif not input_dir:
        return lambda: ligand_pair_generator(halides, acids)
    return lambda: rerun_candidate_generator(input_dir, t_nn, t_ste, t_dft)


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
    t_dft=2.0,  # Threshold for 'dft'.  
):
    """
    Set up and return the BluePhos discovery pipeline executor
    Returns:
        RayStreamGraphExecutor: An executor for the BluePhos discovery pipeline
    """
    
    # Define tasks in the pipeline graph
    t1 = GenerateLigandTableTask()
    t2 = Smiles2SDFTask()
    t3 = NNTask()
    t4 = OptimizeGeometriesTask()
    t5 = DFTTask()

    
    # Construct the pipeline graph
    pipeline_graph = [
        (t1, t2),  # Generate ligands only if necessary
        (t2, t3),  # Convert SMILES to SDF if SMILES input exists
        (t3, FilterNNOutTask),  # NN filter "in" continues to next task
        (t3, FilterNNInTask),  # NN filter "out" goes to sink
        (FilterNNInTask, t4),
        (t4, FilterXTBOutTask),  # XTB filter "in" continues to next task
        (t4, FilterXTBInTask),  # XTB filter "out" goes to sink
        (FilterXTBInTask, t5),
        (t5, FilterDFTOutTask),  # DFT filter "in" could be processed further
        (t5, FilterDFTInTask),  # DFT filter "out" goes to sink
    ]


    # # Define filter configurations dynamically
    # filter_context = {
    #     "t_nn": t_nn,
    #     # "filter_in_nn": True,
    #     "t_ste": t_ste,
    #     # "filter_in_xtb": True,
    #     "t_dft": t_dft,
    #     # "filter_in_dft": True
    # }

    generator = get_generator(ligand_smiles, halides, acids, input_dir, t_nn, t_ste, t_dft)
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
        # **filter_context  # Merge the filter context into the main context
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
    ap.add_argument("--t_dft", type=float, required=False, default=2.0, help="Threshold for 'ste' (default: 2.0)")
    ap.add_argument("--no_xtb", action="store_false", dest="xtb", help="Disable xTB optimization (default: enabled)")
    
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
