import pytest
import pandas as pd
from unittest.mock import patch
from bluephos.bluephos_pipeline import get_pipeline
from dplutils.pipeline.ray import RayStreamGraphExecutor
from bluephos.tasks.optimizegeometries import optimize

@pytest.fixture
def mock_args():
    # Create dummy files for testing
    halides = 'tests/input/aromatic_halides_with_id.csv'
    acids = 'tests/input/aromatic_boronic_acids_with_id.csv'
    features = 'bluephos/parameters/element_features.csv'
    train = 'bluephos/parameters/train_stats.csv'
    weights = 'bluephos/parameters/full_energy_model_weights.pt'
      
    return {
        "halides": halides,
        "acids": acids,
        "element_features": features,
        "train_stats": train,
        "model_weights": weights,
    }


def test_get_pipeline(mock_args):

    pipeline_executor = get_pipeline(**mock_args)
    
    # Assert that the pipeline_executor is an instance of RayStreamGraphExecutor
    assert isinstance(pipeline_executor, RayStreamGraphExecutor)
    

@pytest.fixture
def setup_dataframe():
    mol_block = """
  MJ201100                      

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.9500 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9266    0.0000   -0.3333 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
    """
    data = {'structure': [mol_block]}
    df = pd.DataFrame(data)
    return df

@patch('bluephos.tasks.optimizegeometries.octahedral_embed')
@patch('bluephos.tasks.optimizegeometries.optimize_geometry')
def test_optimize(mock_optimize_geometry, mock_octahedral_embed, setup_dataframe):
    # Assume these functions do not throw an error and behave as expected
    mock_octahedral_embed.return_value = None  # Does not need to return anything
    mock_optimize_geometry.return_value = None  # Does not need to return anything
    
    # Run optimize
    optimize(setup_dataframe, 0, 'fac')
    
    # Check if XYZ data was added or set to failed
    assert setup_dataframe.loc[0, 'xyz'] is not None