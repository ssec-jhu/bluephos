import re
from glob import iglob

import numpy as np

import bluephos.modules.log_config as log_config

# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)


def get_total_energy(filename):
    """Extract the total energy from the given file.

    Args:
        filename (str): Path to the file.

    Returns:
        float: The total energy in eV.
    """
    try:
        with open(filename) as f:
            content = f.read()
        energy = re.findall(r"Total Energy\s*:\s*[\-0-9\.]+ Eh\s*([\-0-9\.]+) eV", content)[0]
        return float(energy)
    except (IndexError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error reading total energy from {filename}: {e}")
        raise


def get_gap(filename):
    """Extract the energy gap from the given file.

    Args:
        filename (str): Path to the file.

    Returns:
        float or str: The energy gap in eV, or "FAILED" if the gap cannot be found.
    """
    try:
        with open(filename) as f:
            content = f.read()

        occ = re.findall(
            r"^\s*NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*^((?:\s*[\-0-9\.]+\s*[\-0-9\.]+\s*" r"[\-0-9\.]+\s*[\-0-9\.]+\s*\n)*)",
            content,
            re.MULTILINE,
        )

        if not occ:
            return "FAILED"

        occ = np.array([[float(value) for value in line.split()] for line in occ[0].rstrip("\n").split("\n")])

        fl = np.flatnonzero(occ[:, 1])[-1]

        return occ[fl + 1, 3] - occ[fl, 3]
    except (IndexError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error reading energy gap from {filename}: {e}")
        return "FAILED"


def get_kr_wav(filename):
    """Extract the transition electric dipole moments and calculate k_r and wavelength.

    Args:
        filename (str): Path to the file.

    Returns:
        tuple: The wavelength and k_r values.
    """
    try:
        with open(filename) as f:
            content = f.read()

        dipole = re.findall(
            r"^SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE "
            r"MOMENTS\s*^.*\n^.*\n^.*\n^.*\n^((?:\s*[0-9]+\s*[0-9]+\s*[\-0-9\.]+\s*"
            r"[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*"
            r"[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*\n)+)",
            content,
            re.MULTILINE,
        )

        if not dipole:
            return None, None

        dipole = np.array([[float(value) for value in line.split()] for line in dipole[0].rstrip("\n").split("\n")])

        wav = dipole[0, 3]

        C = 4.33919882e7
        Kb = 8.617e-5
        T = 300

        K = np.zeros(3)
        E = np.zeros(3)
        for i in range(3):
            E[i] = 1240 / dipole[i, 3]
            K[i] = C * dipole[i, 4] * (1240 / dipole[i, 3])

        k_r = np.sum(K * np.exp(-E / (Kb * T))) / np.sum(np.exp(-E / (Kb * T)))

        return wav, k_r
    except (IndexError, ValueError, FileNotFoundError) as e:
        logger.error(f"Error reading k_r and wavelength from {filename}: {e}")
        return None, None


def extract(triplet_output, base_output):
    """Calculate the energy difference between triplet and base states.

    Args:
        triplet_output (str): Path to the triplet output file.
        base_output (str): Path to the base output file.

    Returns:
        float: The energy difference in eV.
    """
    try:
        Ediff = get_total_energy(triplet_output) - get_total_energy(base_output)
        return Ediff
    except Exception as e:
        logger.error(f"Error processing files: {triplet_output}, {base_output}. Error: {e}")
        return None


def process_files(file_pattern):
    """Process the files matching the given pattern.

    Args:
        file_pattern (str): Pattern to match the files.

    Prints:
        The directory name, energy difference, and k_r for each matched file.
    """
    for filename in iglob(file_pattern):
        triplet_filename = filename.rstrip("output.txt") + "triplet/output.txt"
        Ediff = extract(triplet_filename, filename)
        if Ediff is not None:
            wav_TDDFT, kr = get_kr_wav(filename)
            logger.info(f"{filename.split('/')[-2]}: Ediff={Ediff}, kr={kr}")


if __name__ == "__main__":
    process_files("*/output.txt")
