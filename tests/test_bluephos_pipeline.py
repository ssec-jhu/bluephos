import pytest
from dplutils.pipeline.ray import RayStreamGraphExecutor

from bluephos.bluephos_pipeline import get_pipeline


@pytest.fixture
def mock_args():
    # Create dummy files for testing
    halides = "tests/input/aromatic_halides_with_id.csv"
    acids = "tests/input/aromatic_boronic_acids_with_id.csv"
    features = "bluephos/parameters/element_features.csv"
    train = "bluephos/parameters/train_stats.csv"
    weights = "bluephos/parameters/full_energy_model_weights.pt"
    ligands = "tests/input/ligandlist.csv"

    return {
        "halides": halides,
        "acids": acids,
        "element_features": features,
        "train_stats": train,
        "model_weights": weights,
        "ligand_smiles": ligands,
    }


def test_get_pipeline(mock_args):
    pipeline_executor = get_pipeline(**mock_args)

    # Assert that the pipeline_executor is an instance of RayStreamGraphExecutor
    assert isinstance(pipeline_executor, RayStreamGraphExecutor)
