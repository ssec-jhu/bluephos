import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from bluephos.tasks.dft_orca import dft_run, remove_second_row

@pytest.fixture
def setup_dataframe():
    data = {
        "ligand_identifier": ["H11A1L9"],
        "xyz": [
            """3
H11A1L9
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9500
O -0.9266 0.0000 -0.3333"""
        ]
    }
    df = pd.DataFrame(data)
    return df

@patch("bluephos.tasks.dft_orca.tempfile.mkdtemp")
@patch("bluephos.tasks.dft_orca.run_dft_calculation")
@patch("bluephos.tasks.dft_orca.os.getenv")
@patch("bluephos.tasks.dft_orca.extract")
@patch("subprocess.run")
def test_dft_run(mock_subprocess, mock_extract, mock_getenv, mock_run_dft_calculation, mock_mkdtemp, setup_dataframe):
    mock_getenv.return_value = "mock/path/to/orca"
    mock_mkdtemp.return_value = "/mock/temp/dir"
    mock_run_dft_calculation.return_value = ("/mock/path/triplet_output.txt", "/mock/path/base_output.txt")
    mock_extract.return_value = 0.1234
    mock_subprocess.return_value = MagicMock()
    
    df = dft_run(setup_dataframe)
    
    assert "Energy Diff" in df.columns
    assert df.loc[0, "Energy Diff"] == 0.1234

def test_remove_second_row(setup_dataframe):
    xyz = setup_dataframe.loc[0, "xyz"]
    expected_output = """3
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9500
O -0.9266 0.0000 -0.3333
*"""
    assert remove_second_row(xyz) == expected_output
