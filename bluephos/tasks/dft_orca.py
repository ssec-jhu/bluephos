import os
import pandas as pd
import numpy as np
import subprocess
import logging
import tempfile
from bluephos.modules.extract import extract
from dplutils.pipeline import PipelineTask

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# DEBUG = True retains DFT details for review; False only keeps final results.
DEBUG = False

# Constants and configuration for ORCA input files, need to be customized for specific calculations
ORCA_INPUT_OPT = """!B3LYP LANL2DZ OPT
%PAL NPROCS 48 END
%geom
MaxIter 200
TolE 4e-5 # Energy change (a.u.) (about 1e-3 eV)
TolRMSG 2e-4 # RMS gradient (a.u.)
TolMaxG 4e-4 # Max. element of gradient (a.u.) (about 0.02 ev/A)
TolRMSD 4e-2 # RMS displacement (a.u.)
TolMaxD 8e-2 # Max. displacement (a.u.)
END
* xyz 0 3 {0}
"""

TRIPLET_ORCA_INPUT = """!B3LYP LANL2DZ
%PAL NPROCS 4 END
* xyzfile 0 3 {0}
"""

BASE_ORCA_INPUT = """!B3LYP LANL2DZ
%PAL NPROCS 4 END
* xyzfile 0 1 {0}
"""


def create_orca_input_files(temp_dir, xyz_value):

    """Generate ORCA input files based on the template and provided molecular structure."""

    relax_input_file = os.path.join(temp_dir, "relax.inp")
    triplet_input_file = os.path.join(temp_dir, "triplet.inp")
    base_input_file = os.path.join(temp_dir, "base.inp")
    relax_xyz_file = os.path.join(temp_dir, "relax.xyz")

    with open(relax_input_file, 'w') as file:
        file.write(ORCA_INPUT_OPT.format(xyz_value))

    with open(triplet_input_file, 'w') as file:
        file.write(TRIPLET_ORCA_INPUT.format(relax_xyz_file))

    with open(base_input_file, 'w') as file:
        file.write(BASE_ORCA_INPUT.format(relax_xyz_file))

    return relax_input_file, triplet_input_file, base_input_file


def run_orca_command(input_file, output_file, orca_path):

    """Execute the ORCA software command using subprocess to perform quantum chemical calculations."""

    command = [orca_path, input_file]
    with open(output_file, 'w') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=True, text=True)


def run_dft_calculation(temp_dir, orca_path, xyz_value, base_name):

    """Run DFT calculations for the specified molecule using ORCA."""

    relax_input, triplet_input, base_input = create_orca_input_files(temp_dir, xyz_value)
    relax_output = os.path.join(temp_dir, f"{base_name}_relax_output.txt")
    triplet_output = os.path.join(temp_dir, f"{base_name}_triplet_output.txt")
    base_output = os.path.join(temp_dir, f"{base_name}_base_output.txt")

    try:
        run_orca_command(relax_input, relax_output, orca_path)
        run_orca_command(triplet_input, triplet_output, orca_path)
        run_orca_command(base_input, base_output, orca_path)
    except subprocess.CalledProcessError as e:
        logging.error(f"DFT calculation failed: {e}")
        raise

    return triplet_output, base_output


def remove_second_row(xyz):
    """Remove the second row from the xyz data to comply with ORCA input requirements."""

    lines = xyz.split('\n')
    if len(lines) > 1:
        lines.pop(1)
    lines.append('*')  # Add a new line with '*'
    return '\n'.join(lines)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process each valid entry in the DataFrame for DFT calculations."""

    orca_path = os.path.join(os.getenv('EBROOTORCA'), 'orca')

    Run = False
    for index, row in df.iterrows():
        if Run:
            break
        Run = True
        base_name = None
        if row["xyz"] not in ["failed", None]:
            base_name = row["ligand_identifier"]

        if base_name:
            
            if DEBUG:
                temp_dir = tempfile.mkdtemp()
            else:
                temp_dir = tempfile.TemporaryDirectory()

            try:
                xyz_value = remove_second_row(row["xyz"])
                logging.info(f"Starting DFT calculation for {base_name}...")
                triplet_output, base_output = run_dft_calculation(temp_dir, orca_path, xyz_value, base_name)

                # Extract energy difference from output and add it to dataframe
                df.at[index, "Energy Diff"] = extract(triplet_output, base_output)
            finally:
                if DEBUG:
                    logging.info(f"Temporary files for {base_name} kept at {temp_dir} for debugging.")
                else:
                    if isinstance(temp_dir, tempfile.TemporaryDirectory):
                        temp_dir.cleanup()

    return df


def dft_run(df: pd.DataFrame) -> pd.DataFrame:
    df["Energy Diff"] = np.nan
    return process_dataframe(df)


DFTTask = PipelineTask(
    "dft_run",
    dft_run,
    batch_size=1,
)