import os
import pandas as pd
import numpy as np
import subprocess
import multiprocessing
import logging
import tempfile
from bluephos.modules.dft_extract import extract
from dplutils.pipeline import PipelineTask

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# DEBUG = True retains DFT details for review; False only keeps final results.
DEBUG = True

# Constants
MAX_DEFAULT_CPUS = 48  # Maximum default CPUs to use if not specified by environment


def get_orca_templates(n_cpus, xyz_value, relax_xyz_file):
    ORCA_INPUT_OPT = f"""!B3LYP LANL2DZ OPT
%PAL NPROCS {n_cpus} END
%geom
MaxIter 1
TolE 4e-5 # Energy change (a.u.) (about 1e-3 eV)
TolRMSG 2e-4 # RMS gradient (a.u.)
TolMaxG 4e-4 # Max. element of gradient (a.u.) (about 0.02 ev/A)
TolRMSD 4e-2 # RMS displacement (a.u.)
TolMaxD 8e-2 # Max. displacement (a.u.)
END
* xyz 0 3 {xyz_value}
"""

    TRIPLET_ORCA_INPUT = f"""!B3LYP LANL2DZ
%PAL NPROCS {n_cpus} END
* xyzfile 0 3 {relax_xyz_file}
"""

    BASE_ORCA_INPUT = f"""!B3LYP LANL2DZ
%PAL NPROCS {n_cpus} END
* xyzfile 0 1 {relax_xyz_file}
"""

    return ORCA_INPUT_OPT, TRIPLET_ORCA_INPUT, BASE_ORCA_INPUT


def create_orca_input_files(temp_dir, xyz_value):
    """
    Generates ORCA input files with appropriate configurations based on the given parameters.

    Parameters:
    - temp_dir (str): The directory where the ORCA input files will be saved.
    - xyz_value (str): The XYZ coordinates for the molecule, formatted as required by ORCA.

    Returns:
    - tuple: Paths to the generated ORCA input files (relax.inp, triplet.inp, base.inp).
    """

    # Determine the number of CPUs to use based on system capabilities and environment settings
    n_cpu_custom = min(multiprocessing.cpu_count(), MAX_DEFAULT_CPUS)
    n_cpus = os.getenv("OMP_NUM_THREADS", str(n_cpu_custom))
    logging.info(f"Using {n_cpus} CPUs for computation")

    # Prepare file paths for the ORCA input files
    relax_input_file = os.path.join(temp_dir, "relax.inp")
    triplet_input_file = os.path.join(temp_dir, "triplet.inp")
    base_input_file = os.path.join(temp_dir, "base.inp")
    relax_xyz_file = os.path.join(temp_dir, "relax.xyz")

    # Generate ORCA input configurations using the provided templates
    ORCA_INPUT_OPT, TRIPLET_ORCA_INPUT, BASE_ORCA_INPUT = get_orca_templates(n_cpus, xyz_value, relax_xyz_file)

    # Write the configurations to the respective files
    with open(relax_input_file, "w") as file:
        file.write(ORCA_INPUT_OPT)
    with open(triplet_input_file, "w") as file:
        file.write(TRIPLET_ORCA_INPUT)
    with open(base_input_file, "w") as file:
        file.write(BASE_ORCA_INPUT)

    return relax_input_file, triplet_input_file, base_input_file


def run_orca_command(input_file, output_file, orca_path):
    """Execute the ORCA software command using subprocess to perform quantum chemical calculations."""

    command = [orca_path, input_file]
    with open(output_file, "w") as output:
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

    lines = xyz.split("\n")
    if len(lines) > 1:
        lines.pop(1)
    lines.append("*")  # Add a new line with '*'
    return "\n".join(lines)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process each valid entry in the DataFrame for DFT calculations."""

    orca_path = os.path.join(os.getenv("EBROOTORCA"), "orca")

    for index, row in df.iterrows():
        base_name = None
        if row["xyz"] not in ["failed", None]:
            base_name = row["ligand_identifier"]
        else:
            continue

        if DEBUG:
            temp_dir = tempfile.mkdtemp()
        else:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

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
    num_cpus=48,
)
