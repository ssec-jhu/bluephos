import sys
import re
from time import time, sleep
from glob import glob
import gzip
from concurrent.futures import ThreadPoolExecutor
from os.path import basename
from rdkit.Chem.rdmolfiles import ForwardSDMolSupplier, MolToMolFile
from rdkit.Chem.rdmolops import AddHs
from octahedral_embed import octahedral_embed
from annotate_rdkit_with_ase import optimize_geometry
from xtb.ase.calculator import XTB
from bond_length import bonds_maintained
from isoctahedral import isoctahedral
from ase.calculators.calculator import InputError

mols_path = sys.argv[1]
failures_path = "failures.txt"
outpath = "triplet_xtb_geom"

# Make a list of molecule IDs for for which geometry has already been calculated
already_ran = set(re.match(r"(.*).mol", basename(path)).group(1) \
                  for path in glob(f"{outpath}/*.mol"))

print(f"{len(already_ran)} already ran")

# Make a list of molecule IDs for which geometry generation was attempted, but
# failed
failed = set(line.strip() \
             for line in open(failures_path))
print(f"{len(failed)} previously failed")

def optimize_and_write(mol, isomer):
    mol_id_raw = mol.GetProp("_Name")
    mol_id = mol_id_raw + f"_{isomer}"
    mol.SetProp("_Name", mol_id)
    start = time()
    for attempt in range(3):
        try:
            octahedral_embed(mol, isomer)
            optimize_geometry(XTB(method="GFN2-xTB"), mol, uhf = 2, conformation_index = 0)
            end = time()
            print(f"Optimized {mol_id} in {end-start} seconds")
        except InputError:
            # This error seems to occur when it can't read the parameter file
            # Just a niagara error I think? The parameter file is there
            # These ones tend to succeed after rerunning once or twice
            print(f"InputError for {mol_id}, will attempt {3 - (attempt+1)} more times")
            # These errors seem to happen all at once at the beginning. Maybe
            # we need to wait them out?
            sleep(10)
            continue 
        except ValueError:
            # ConstrainedEmbed gives a value error
            print(f"ValueError, probably because ConstrainedEmbed couldn't embed {mol_id}")
            with open(failures_path, "a") as f:
                f.write(mol_id + "\n")
            break
        except:
            print(f"Other problem with {mol_id}")
            with open(failures_path, "a") as f:
                f.write(mol_id + "\n")
            break
        if bonds_maintained(mol) and isoctahedral(mol):
            MolToMolFile(mol, f"{outpath}/{mol_id}.mol")
            break
        else:
            print(f"{mol_id} failed geometry check")
            # In my experience the ones that fail the geometry check fail it on
            # every attempt, so don't try again
            with open(failures_path, "a") as f:
                f.write(mol_id + "\n")
            break

def check_and_run(mol):
    if mol.GetNumAtoms() > 200:
        print("More than 200 atoms")
        return None
    mol_id_raw = mol.GetProp("_Name")
    mol_id = mol_id_raw + "_fac"
    if mol_id in already_ran:
        print(f"{mol_id} has already run")
        return None
    if mol_id in failed:
        print(f"{mol_id} previously failed")
        return None
    optimize_and_write(AddHs(mol),  "fac")

#    mol_id_raw = mol.GetProp("_Name")
#    mol_id = mol_id_raw + "_mer"
#    if mol_id in already_ran:
#        print(f"{mol_id} has already run")
#        return None
#    if mol_id in failed:
#        print(f"{mol_id} previously failed")
#        return None
#    optimize_and_write(AddHs(mol), "mer")
    return None

#with ThreadPoolExecutor(max_workers = 80) as pool:
#    pool.map(check_and_run, ForwardSDMolSupplier(\
#            gzip.open(mols_path, "rb"),
#        removeHs = False))
list(map(check_and_run, ForwardSDMolSupplier(\
        gzip.open(mols_path, "rb"),
        removeHs = False)))
