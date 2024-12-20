import logging
import multiprocessing
import os
import subprocess
from textwrap import dedent
import socket

from bluephos.modules.dft_extract import extract

# Constants
MAX_DEFAULT_CPUS = 48  # Maximum default CPUs to use if not specified by environment


class DFTCalculator:
    """
    Base class for DFT Calculators.
    """

    def __init__(self, n_cpus):
        self.n_cpus = n_cpus

    def prepare_input_files(self, temp_dir, xyz_value):
        raise NotImplementedError

    def run_calculation(self, temp_dir, base_name):
        raise NotImplementedError

    def extract_results(self, temp_dir, base_name):
        raise NotImplementedError


class OrcaCalculator(DFTCalculator):
    """
    ORCA implementation of the DFTCalculator
    """

    def __init__(self, n_cpus, orca_path):
        super().__init__(n_cpus)
        self.orca_path = orca_path

    def prepare_input_files(self, temp_dir, xyz_value):
        return create_orca_input_files(temp_dir, xyz_value)

    def run_calculation(self, temp_dir, base_name, xyz_value):
        relax_input, triplet_input, base_input = self.prepare_input_files(temp_dir, xyz_value)
        relax_output = os.path.join(temp_dir, f"{base_name}_relax_output.txt")
        triplet_output = os.path.join(temp_dir, f"{base_name}_triplet_output.txt")
        base_output = os.path.join(temp_dir, f"{base_name}_base_output.txt")

        try:
            run_orca_command(relax_input, relax_output, self.orca_path)
            run_orca_command(triplet_input, triplet_output, self.orca_path)
            run_orca_command(base_input, base_output, self.orca_path)
        except subprocess.CalledProcessError as e:
            logging.error(f"DFT calculation failed for {base_name}: {e}")
            raise

        return triplet_output, base_output

    def extract_results(self, temp_dir, base_name, xyz_value):
        triplet_output, base_output = self.run_calculation(temp_dir, base_name, xyz_value)
        return extract(triplet_output, base_output)


def create_orca_input_files(temp_dir, xyz_value):
    """
    Create ORCA input files for different calculation types.
    """
    n_cpu_custom = min(multiprocessing.cpu_count(), MAX_DEFAULT_CPUS)
    n_cpus = os.getenv("OMP_NUM_THREADS", str(n_cpu_custom))
    logging.info(f"Using {n_cpus} CPUs for computation")

    relax_input_file = os.path.join(temp_dir, "relax.inp")
    triplet_input_file = os.path.join(temp_dir, "triplet.inp")
    base_input_file = os.path.join(temp_dir, "base.inp")
    relax_xyz_file = os.path.join(temp_dir, "relax.xyz")

    ORCA_INPUT_OPT = dedent(f"""\
    !B3LYP LANL2DZ OPT
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
    """)

    TRIPLET_ORCA_INPUT = dedent(f"""\
    !B3LYP LANL2DZ
    %PAL NPROCS {n_cpus} END
    * xyzfile 0 3 {relax_xyz_file}
    """)

    BASE_ORCA_INPUT = dedent(f"""\
    !B3LYP LANL2DZ
    %PAL NPROCS {n_cpus} END
    * xyzfile 0 1 {relax_xyz_file}
    """)

    with open(relax_input_file, "w") as file:
        file.write(ORCA_INPUT_OPT)
    with open(triplet_input_file, "w") as file:
        file.write(TRIPLET_ORCA_INPUT)
    with open(base_input_file, "w") as file:
        file.write(BASE_ORCA_INPUT)

    return relax_input_file, triplet_input_file, base_input_file


def run_orca_command(input_file, output_file, orca_path):
    """
    Run the ORCA command and log outputs.
    """
    this_host = socket.gethostname()
    mpi_options = f"--oversubscribe --host {this_host}"
    command = [orca_path, input_file, mpi_options]
    logging.info(f"Running ORCA command: {command}")
    with open(output_file, "w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=True, text=True)
        if result.stderr:  # Check if stderr is not empty
            logging.error(f"ORCA command error output: {result.stderr}")
        else:
            logging.info(f"ORCA command completed successfully for {input_file}")


def remove_second_row(xyz):
    """Remove the second row of an XYZ string and add an asterisk at the end"""

    lines = xyz.split("\n")
    if len(lines) > 1:
        lines.pop(1)
    lines.append("*")  # Add a new line with '*'
    return "\n".join(lines)


# Placeholder for future implementation of ASECalculator
class ASECalculator(DFTCalculator):
    """
    ASE implementation of the DFTCalculator.
    (To be implemented later)
    """

    def __init__(self, n_cpus):
        super().__init__(n_cpus)

    def prepare_input_files(self, temp_dir, xyz_value):
        # Prepare ASE input files
        pass

    def run_calculation(self, temp_dir, base_name):
        # Run ASE
        pass

    def extract_results(self, temp_dir, base_name):
        # Extract results from ASE output files
        pass
