import re
import numpy as np
from glob import iglob


def get_total_energy(filename):
    with open(filename) as f:
        s = f.read()
    return float(re.findall("Total Energy\s*:\s*[\-0-9\.]+ Eh\s*([\-0-9\.]+) eV", s)[0])


def get_gap(filename):
    with open(filename) as f:
        s = f.read()

    occ = re.findall(
        "^\s*NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*^((?:\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*\n)*)",
        s,
        re.MULTILINE,
    )

    if len(occ) == 0:
        return "FAILED"

    occ = np.array([[float(l) for l in line.split()] for line in occ[0].rstrip("\n").split("\n")])

    fl = np.flatnonzero(occ[:, 1])[-1]

    return occ[fl + 1, 3] - occ[fl, 3]


def get_kr_wav(filename):
    with open(filename) as f:
        s = f.read()

    dipole = re.findall(
        "^SPIN ORBIT CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\s*^.*\n^.*\n^.*\n^.*\n^((?:\s*[0-9]+\s*[0-9]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*[\-0-9\.]+\s*\n)+)",
        s,
        re.MULTILINE,
    )

    if len(dipole) == 0:
        return None, None

    dipole = np.array([[float(l) for l in line.split()] for line in dipole[0].rstrip("\n").split("\n")])

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


def extract(triplet_output, base_output):
    try:
        Ediff = get_total_energy(triplet_output) - get_total_energy(base_output)
    except Exception as e:
        print(f"Error processing files: {triplet_output}, {base_output}. Error: {e}")
        return None

    return Ediff


# Example usage
for filename in iglob("*/output.txt"):
    triplet_filename = filename.rstrip("output.txt") + "triplet/output.txt"
    Ediff = extract(triplet_filename, filename)
    if Ediff is not None:
        wav_TDDFT, kr = get_kr_wav(filename)
        print(filename.split("/")[-2], Ediff, kr)
