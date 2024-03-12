"""
BluePhos Discovery Pipeline
"""

import pandas as pd
import argparse
from dplutils.pipeline.ray import RayDataPipelineExecutor
from tasks.generateligandtable import generate_ligand_table, GenerateLigandTableTask
from tasks.smiles2mol import smiles_to_mols, Smiles2MolTask
from tasks.cauldron2feature import feature_create, Cauldron2FeatureTask
from tasks.nn import apply_nn, NNTask

# Toggle execution mode: True for Ray's distributed processing, False for local testing and debugging.
ray_enabled = False


def get_pipeline(halides_file_name, acids_file_name, input_folder, output_folder):
    steps = [
        GenerateLigandTableTask,
        Smiles2MolTask,
        Cauldron2FeatureTask,
        NNTask,
        # Add additional tasks here as needed
    ]
    pipeline_executor = RayDataPipelineExecutor(steps)
    context_dict = {
        "halides_file_name": halides_file_name,
        "acids_file_name": acids_file_name,
        "input_folder": input_folder,
        "output_folder": output_folder,
    }
    for key, value in context_dict.items():
        pipeline_executor.set_context(key, value)
    return pipeline_executor


def test_pipeline(halides_file_name, acids_file_name, input_folder, output_folder):

    # Assuming inputs is an empty DataFrame for this example; adjust as necessary
    inputs = pd.DataFrame()

    # Run tasks sequentially, feeding the output of one as the input to the next
    ligand_table_df = generate_ligand_table(
        inputs, halides_file_name, acids_file_name, input_folder, output_folder
    )
    print("GenerateLigandTable Task completed!")

    smiles_to_mol_df = smiles_to_mols(ligand_table_df, output_folder)
    print("Smiles2Mol Task completed!")

    feature_df = feature_create(smiles_to_mol_df, input_folder, output_folder)
    print("Feature Task completed!")

    nn_score_df = apply_nn(feature_df, input_folder, output_folder)
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

    args = parser.parse_args()

    if ray_enabled:
        pipeline = get_pipeline(args.halides, args.acids, args.input, args.output)

        # pipeline.run()
        for batch in pipeline.run():
            print("Batch")
            # print(batch)
    else:
        nn_score = test_pipeline(args.halides, args.acids, args.input, args.output)
        print(nn_score.head(5))

