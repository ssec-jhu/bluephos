__doc__ = """"BluePhos Discovery Pipeline"""

import pandas as pd
from dplutils import cli
from dplutils.pipeline.ray import RayStreamGraphExecutor

from bluephos.tasks.generateligandtable import GenerateLigandTableTask
from bluephos.tasks.nn import NNTask
from bluephos.tasks.optimizegeometries import OptimizeGeometriesTask
from bluephos.tasks.smiles2sdf import Smiles2SDFTask


def ligand_pair_generator(halides_file, acids_file):
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
            yield (pd.DataFrame([ligand_pair]))


def get_pipeline(
    halides,
    acids,
    element_features,
    train_stats,
    model_weights,
):
    steps = [
        GenerateLigandTableTask,
        Smiles2SDFTask,
        NNTask,
        OptimizeGeometriesTask,
        # Add additional tasks here as needed
    ]
    pipeline_executor = RayStreamGraphExecutor(
        graph=steps,
        generator=lambda: ligand_pair_generator(halides, acids),
    )
    context_dict = {
        "halides": halides,
        "acids": acids,
        "element_features": element_features,
        "train_stats": train_stats,
        "model_weights": model_weights,
    }
    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


if __name__ == "__main__":
    ap = cli.get_argparser(description=__doc__)
    ap.add_argument("--halides", required=True, help="CSV file containing halides data")
    ap.add_argument("--acids", required=True, help="CSV file containing boronic acids data")
    ap.add_argument("--features", required=True, help="Element feature file")
    ap.add_argument("--train", required=True, help="Train stats file")
    ap.add_argument("--weights", required=True, help="Full energy model weights")
    args = ap.parse_args()

    cli.cli_run(
        get_pipeline(
            args.halides,
            args.acids,
            args.features,
            args.train,
            args.weights,
        ),
        args,
    )
