import pytest
import pandas as pd
from unittest.mock import patch
from bluephos.tasks.dft_orca import dft_run, extract, remove_second_row


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
        ],
    }
    df = pd.DataFrame(data)
    return df


def create_test_data(output_dir, triplet_filename="triplet_output.txt", base_filename="base_output.txt"):
    triplet_content = """Simulated ORCA triplet output data
Total Energy       :     -4073.79546374 Eh        -273432.343322 eV"""
    base_content = """Simulated ORCA base output data
Total Energy       :     -4074.79546374 Eh        -273431.343322 eV"""
    (output_dir / triplet_filename).write_text(triplet_content)
    (output_dir / base_filename).write_text(base_content)


@patch("bluephos.tasks.dft_orca.run_orca_command")
def test_dft_run(mock_run_orca_command, setup_dataframe, tmp_path):
    # Setup mock to simulate ORCA command output
    mock_run_orca_command.side_effect = lambda *args, **kwargs: create_test_data(tmp_path)
    triplet_output_file = tmp_path / "triplet_output.txt"
    base_output_file = tmp_path / "base_output.txt"

    # Assume dft_run now accepts an output file path as an argument and reads from it
    df = dft_run(setup_dataframe)

    # Assertions to check file usage, existence, and DataFrame updates
    assert triplet_output_file.exists() and base_output_file.exists(), "Output file was not created as expected"
    assert "Energy Diff" in df.columns
    assert isinstance(df.loc[0, "Energy Diff"], float)


def test_extraction_from_real_output(tmp_path):
    create_test_data(tmp_path)
    triplet_output_path = tmp_path / "triplet_output.txt"
    base_output_path = tmp_path / "base_output.txt"

    # Assuming extract function is expecting paths to both output files
    expected_diff = -273432.343322 + 273431.343322  # Calculation based on provided values
    energy_diff = extract(triplet_output_path, base_output_path)

    # Verifying extract reads and parses the output correctly
    assert isinstance(energy_diff, float)
    assert energy_diff == pytest.approx(expected_diff)


def test_remove_second_row(setup_dataframe):
    xyz = setup_dataframe.loc[0, "xyz"]
    expected_output = """3
H 0.0000 0.0000 0.0000
H 0.0000 0.0000 0.9500
O -0.9266 0.0000 -0.3333
*"""
    assert remove_second_row(xyz) == expected_output
