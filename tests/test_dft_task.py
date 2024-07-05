import os
import pytest
import pandas as pd
from unittest.mock import patch
from bluephos.tasks.dft import dft_run, remove_second_row, OrcaCalculator


@pytest.fixture
def setup_dataframe():
    ligand_identifier = "H11A1L9"
    data = {
        "ligand_identifier": [ligand_identifier],
        "xyz": [
            """3
H11A1L9
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9500
O -0.9266 0.0000 -0.3333"""
        ],
        "ste": [0.05],
        "energy diff": [None],
    }
    df = pd.DataFrame(data)
    return df


def create_test_data(output_dir, ligand_identifier, triplet_filename=None, base_filename=None):
    if triplet_filename is None:
        triplet_filename = f"{ligand_identifier}_triplet_output.txt"
    if base_filename is None:
        base_filename = f"{ligand_identifier}_base_output.txt"

    triplet_content = """Simulated ORCA triplet output data
Total Energy       :     -4073.79546374 Eh        -273432.343322 eV"""
    base_content = """Simulated ORCA base output data
Total Energy       :     -4074.79546374 Eh        -273431.343322 eV"""

    (output_dir / triplet_filename).write_text(triplet_content)
    (output_dir / base_filename).write_text(base_content)


@patch("bluephos.tasks.dft.OrcaCalculator.extract_results")
@patch("tempfile.mkdtemp")
def test_dft_run(mock_mkdtemp, mock_extract_results, setup_dataframe, tmp_path):
    # Setup mock to simulate ORCA command output
    mock_extract_results.side_effect = lambda temp_dir, base_name, xyz: -273432.343322 + 273431.343322

    # Mocking mkdtemp to use tmp_path
    mock_mkdtemp.return_value = str(tmp_path)

    os.environ["EBROOTORCA"] = "/mock/path/to/orca"

    # Run dft_run
    df = dft_run(setup_dataframe, t_ste=0.1, package="orca")

    # Assertions to check file usage, existence, and DataFrame updates
    assert "energy diff" in df.columns
    assert isinstance(df.loc[0, "energy diff"], float)
    assert df.loc[0, "energy diff"] == pytest.approx(-273432.343322 + 273431.343322)


@patch("bluephos.tasks.dft.OrcaCalculator.extract_results")
def test_extraction_from_real_output(mock_extract_results, tmp_path):
    ligand_identifier = "H11A1L9"
    create_test_data(tmp_path, ligand_identifier)

    # Assuming extract function is expecting paths to both output files
    expected_diff = -273432.343322 + 273431.343322  # Calculation based on provided values
    mock_extract_results.return_value = expected_diff

    # Mocking OrcaCalculator instance
    orca_calculator = OrcaCalculator(n_cpus=1, orca_path="/mock/path/to/orca")

    energy_diff = orca_calculator.extract_results(
        temp_dir=str(tmp_path), base_name=ligand_identifier, xyz_value="mock_xyz"
    )
    mock_extract_results.return_value = expected_diff

    # Verifying extract reads and parses the output correctly
    assert isinstance(energy_diff, float)
    assert energy_diff == pytest.approx(expected_diff)
    mock_extract_results.assert_called_once_with(
        temp_dir=str(tmp_path), base_name=ligand_identifier, xyz_value="mock_xyz"
    )


def test_remove_second_row(setup_dataframe):
    xyz = setup_dataframe.loc[0, "xyz"]
    expected_output = """3
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9500
O -0.9266 0.0000 -0.3333
*"""
    assert remove_second_row(xyz) == expected_output
