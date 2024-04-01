"""
BluePhos Discovery Pipeline
"""

import os
import pandas as pd
import argparse
from dplutils.pipeline.ray import RayStreamGraphExecutor
from bluephos.tasks.generateligandtable import (
    generate_ligand_table,
    GenerateLigandTableTask,
)
from tasks.smiles2sdf import smiles_to_sdf, Smiles2SDFTask
from tasks.nn import nn, NNTask

# Toggle execution mode: True for Ray's distributed processing, False for local testing and debugging.
ray_enabled = True


def ligand_pair_generator(halides_file_name, acids_file_name, input_folder):
    halides_df = pd.read_csv(os.path.join(input_folder, halides_file_name))
    acids_df = pd.read_csv(os.path.join(input_folder, acids_file_name))
    for _, halide in halides_df.iterrows():
        for _, acid in acids_df.iterrows():
            ligand_pair = {
                "halide_identifier": halide["halide_identifier"],
                "halide_SMILES": halide["halide_SMILES"],
                "acid_identifier": acid["acid_identifier"],
                "acid_SMILES": acid["acid_SMILES"],
            }
            ligand_pair_df = pd.DataFrame([ligand_pair])
            yield ligand_pair_df


def get_pipeline(halides_file_name, acids_file_name, input_folder, output_folder, element_feature, train_stats):
    steps = [
        GenerateLigandTableTask,
        Smiles2SDFTask,
        NNTask,
        # Add additional tasks here as needed
    ]
    pipeline_executor = RayStreamGraphExecutor(
        graph=steps,
        generator=lambda: ligand_pair_generator(
            halides_file_name, acids_file_name, input_folder
        ),
    )
    context_dict = {
        "halides_file_name": halides_file_name,
        "acids_file_name": acids_file_name,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "element_feature":element_feature,
        "train_stats":train_stats,
        
    }
    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


def test_pipeline(halides_file_name, acids_file_name, input_folder, output_folder, element_feature, train_stats):

    # Initialize an empty DataFrame to collect generator output
    ligand_pairs = pd.DataFrame()

    # Consume the generator and append each DataFrame to `ligand_pairs`
    for ligand_pair_df in ligand_pair_generator(
        halides_file_name, acids_file_name, input_folder
    ):
        ligand_pairs = pd.concat([ligand_pairs, ligand_pair_df], ignore_index=True)

    # Run tasks sequentially, feeding the output of one as the input to the next
    ligand_table_df = generate_ligand_table(ligand_pairs)
    print("GenerateLigandTable Task completed!")

    smiles_to_sdf_df = smiles_to_sdf(ligand_table_df, output_folder)
    print("Smiles2SDF Task completed!")

    nn_score_df = nn(smiles_to_sdf_df, input_folder, output_folder, element_feature, train_stats)
    print("NN Task completed!")

    # Optionally, return the results for further processing or verification
    return nn_score_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BluePhos Discovery Pipeline")
    parser.add_argument(
        "--halides", required=True, help="CSV file containing halides data"
    )
    parser.add_argument(
        "--acids", required=True, help="CSV file containing boronic acids data"
    )
    parser.add_argument("--input", required=True, help="Input folder for the results")
    parser.add_argument("--output", required=True, help="Output folder for the results")
    parser.add_argument("--feature", required=True, help="element feature for the results")
    parser.add_argument("--train", required=True, help="train stats folder for the results")
    args = parser.parse_args()

    if ray_enabled:
        pipeline = get_pipeline(args.halides, args.acids, args.input, args.output, args.feature, args.train)

        output = pipeline.run()

        for batch in output:
            print("Batch")
            print(batch.columns)
            print(batch)

    else:
        nn_score = test_pipeline(args.halides, args.acids, args.input, args.output, args.feature, args.train)
        print(nn_score)
