import pytest
import pandas as pd
from unittest.mock import patch
from bluephos.tasks.optimizegeometries import optimize_geometries


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
    data = {"ligand_identifier": "H11A1L0", "structure": [mol_block]}
    df = pd.DataFrame(data)
    return df


@patch("bluephos.tasks.optimizegeometries.octahedral_embed")
@patch("bluephos.tasks.optimizegeometries.optimize_geometry")
def test_optimize(mock_optimize_geometry, mock_octahedral_embed, setup_dataframe):
    # Assume these functions do not throw an error and behave as expected
    mock_octahedral_embed.return_value = None  # Does not need to return anything
    mock_optimize_geometry.return_value = None  # Does not need to return anything

    # Run optimize
    output_dataframe = optimize_geometries(setup_dataframe)

    # Check if XYZ data was added or set to failed
    assert output_dataframe.loc[0, "xyz"] is not None
