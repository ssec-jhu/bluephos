import os
import tempfile
import multiprocessing
import pandas as pd
import bluephos.modules.log_config as log_config
from bluephos.modules.dft_calculators import OrcaCalculator, ASECalculator, remove_second_row
from dplutils.pipeline import PipelineTask

# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)

# DEBUG = True retains DFT details for review; False only keeps final results
DEBUG = True

# Constants
MAX_DEFAULT_CPUS = 48  # Maximum default CPUs to use if not specified by environment


def get_dft_calculator(dft_package, n_cpus):
    """
    Get the appropriate DFT calculator based on the specified package.

    Args:
        dft_package (str): The name of the DFT package to use ('orca' or 'ase').
        n_cpus (int): Number of CPUs to use for the calculations.

    Returns:
        An instance of the specified DFT calculator.

    Raises:
        ValueError: If an unsupported DFT package is specified.
    """

    if dft_package == "orca":
        orca_path = os.path.join(os.getenv("EBROOTORCA"), "orca")
        return OrcaCalculator(n_cpus, orca_path)
    elif dft_package == "ase":
        return ASECalculator(n_cpus)
    else:
        raise ValueError("Unsupported DFT package")


# Process each row of the DataFrame to perform DFT calculations
def process_dataframe(row, t_ste, dft_calculator):
    mol_id = row["ligand_identifier"]
    ste = row["ste"]
    energy_diff = row["dft_energy_diff"]

    if ste is None or abs(ste) >= t_ste or energy_diff is not None:
        logger.info(f"Skipping DFT on molecule {mol_id} based on z or t_ste conditions.")
        return row

    if row["xyz"] not in ["failed", None]:
        base_name = row["ligand_identifier"]
    else:
        return row

    if DEBUG:
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

    try:
        xyz_value = remove_second_row(row["xyz"])
        logger.info(f"Starting DFT calculation for {base_name}...")
        energy_diff = dft_calculator.extract_results(temp_dir, base_name, xyz_value)
        row["dft_energy_diff"] = energy_diff
        return row
    finally:
        if DEBUG:
            logger.info(f"Temporary files for {base_name} kept at {temp_dir} for debugging.")
        else:
            if isinstance(temp_dir, tempfile.TemporaryDirectory):
                temp_dir.cleanup()


# Run DFT calculations on the DataFrame
def dft_run(df: pd.DataFrame, t_ste: float, dft_package: str) -> pd.DataFrame:
    n_cpu_custom = min(multiprocessing.cpu_count(), MAX_DEFAULT_CPUS)
    n_cpus = int(os.getenv("OMP_NUM_THREADS", str(n_cpu_custom)))

    dft_calculator = get_dft_calculator(dft_package, n_cpus)

    if 'dft_energy_diff' not in df.columns:
        df['dft_energy_diff'] = None
    df = df.apply(process_dataframe, axis=1, t_ste=t_ste, dft_calculator=dft_calculator)
    return df


# Define the PipelineTask for the DFT run
DFTTask = PipelineTask(
    "dft_run",
    dft_run,
    context_kwargs={
        "t_ste": "t_ste",
        "dft_package": "dft_package",  #  Either "ase" or "orca"
    },
    batch_size=1,
    num_cpus=32,
)
