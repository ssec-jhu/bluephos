__doc__ = """"BluePhos Discovery Pipeline"""

import pandas as pd
import json
from dplutils import cli
from dplutils.pipeline.ray import RayStreamGraphExecutor
from pathlib import Path
from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.smiles2sdf import Smiles2SDFTask
from bluephos.tasks.dft import DFTTask


def initialize_dataframe():
    """Initialize a DataFrame with the required columns."""
    columns = [
        "ligand_identifier",
        "ligand_SMILES",
        "halide_identifier",
        "halide_SMILES",
        "acid_identifier",
        "acid_SMILES",
        "structure",
        "z",
        "xyz",
        "ste",
        "dft_energy_diff",
    ]
    return pd.DataFrame(columns=columns)


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
            ligand_pair_df = initialize_dataframe()

            ligand_pair_df.at[0, "halide_identifier"] = halide["halide_identifier"]
            ligand_pair_df.at[0, "halide_SMILES"] = halide["halide_SMILES"]
            ligand_pair_df.at[0, "acid_identifier"] = acid["acid_identifier"]
            ligand_pair_df.at[0, "acid_SMILES"] = acid["acid_SMILES"]

            yield ligand_pair_df


def rerun_candidate_generator(input_dir, t_nn, t_ste, t_ed):
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
        t_ed (float): Threshold for 'dft_energy_diff'.

    Yields:
        DataFrame: A single-row DataFrame containing candidate data.
    """
    for file in Path(input_dir).glob("*.parquet"):
        df = pd.read_parquet(file)

        filtered = df[
            (df["z"].notnull())
            & (df["z"].abs() < t_nn)
            & ((df["ste"].isnull()) | (df["ste"].abs() < t_ste))
            & (df["dft_energy_diff"].isna())
        ]
        for _, row in filtered.iterrows():
            yield row.to_frame().transpose()


def get_generator(halides, acids, input_dir, t_nn, t_ste, t_ed):
    if not input_dir:
        return lambda: ligand_pair_generator(halides, acids)
    else:
        return lambda: rerun_candidate_generator(input_dir, t_nn, t_ste, t_ed)


def get_pipeline(
    halides,  # Path to the halides CSV file
    acids,  # Path to the acids CSV file
    element_features,  # Path to the element features file
    train_stats,  # Path to the train stats file
    model_weights,  # Path to the model weights file
    input_dir=None,  # Directory containing input parquet files(rerun). Defaults to None.
    dft_package="orca",  # DFT package to use. Defaults to "orca".
    t_nn=None,  # Threshold for 'z' score. Defaults to None
    t_ste=None,  # Threshold for 'ste'. Defaults to None
    t_ed=None,  # Threshold for 'dft_energy_diff'. Defaults to None
):
    steps = (
        [
            GenerateLigandTableTask,
            Smiles2SDFTask,
            NNTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]
        if not input_dir
        else [
            NNTask,
            OptimizeGeometriesTask,
            DFTTask,
        ]
    )
    generator = get_generator(halides, acids, input_dir, t_nn, t_ste, t_ed)
    pipeline_executor = RayStreamGraphExecutor(graph=steps, generator=generator)

    context_dict = {
        "halides": halides,
        "acids": acids,
        "element_features": element_features,
        "train_stats": train_stats,
        "model_weights": model_weights,
        "dft_package": dft_package,
        "t_nn": t_nn,
        "t_ste": t_ste,
        "t_ed": t_ed,
    }

    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


if __name__ == "__main__":
    ap = cli.get_argparser(description=__doc__)
    ap.add_argument("--halides", required=False, help="CSV file containing halides data")
    ap.add_argument("--acids", required=False, help="CSV file containing boronic acids data")
    ap.add_argument("--features", required=True, help="Element feature file")
    ap.add_argument("--train", required=True, help="Train stats file")
    ap.add_argument("--weights", required=True, help="Full energy model weights")
    ap.add_argument("--out_dir", required=False, help="Directory for output parquet files")
    ap.add_argument("--input_dir", required=False, help="Directory containing input parquet files")
    ap.add_argument("--threshold_file", required=False, help="JSON file containing t_nn, t_ste, and t_ed threshold")
    ap.add_argument(
        "--dft_package",
        required=False,
        default="orca",
        choices=["orca", "ase"],
        help="DFT package to use (default: orca)",
    )
    args = ap.parse_args()

    # Initialize thresholds
    t_nn, t_ste, t_ed = None, None, None

    # Load thresholds from the provided JSON file if available
    if args.threshold_file:
        with open(args.threshold_file, "r") as f:
            config = json.load(f)

            thresholds = config["thresholds"]
            t_nn = thresholds["t_nn"]
            t_ste = thresholds["t_ste"]
            t_ed = thresholds["t_ed"]

    # Run the pipeline with the provided arguments
    cli.cli_run(
        get_pipeline(
            args.halides,
            args.acids,
            args.features,
            args.train,
            args.weights,
            args.input_dir,
            args.dft_package,
            t_nn,
            t_ste,
            t_ed,
        ),
        args,
    )
