import logging
import pandas as pd
from time import time, sleep
from rdkit import Chem
from rdkit.Chem import AddHs, MolToXYZBlock
from bluephos.modules.octahedral_embed import octahedral_embed
from bluephos.modules.annotate_rdkit_with_ase import optimize_geometry
from bluephos.modules.bond_length import bonds_maintained
from bluephos.modules.isoctahedral import isoctahedral
# from xtb.ase.calculator import XTB
from ase.calculators.calculator import InputError
from dplutils.pipeline import PipelineTask

try:
    from xtb.ase.calculator import XTB
except ImportError:
    XTB = None
    print("xtb not installed. Limited functionality available.")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def optimize(df, row_index, isomer):

    start = time()
    mol = AddHs(Chem.MolFromMolBlock(df.loc[row_index, "structure"]))
    mol_id = mol.GetProp("_Name") + f"_{isomer}"
    mol.SetProp("_Name", mol_id)

    for attempt in range(3):
        try:
            octahedral_embed(mol, isomer)
            if XTB is not None:
                optimize_geometry(XTB(method="GFN2-xTB"), mol, uhf=2, conformation_index=0)
            else:
                print("Proceeding without xTB functionality.")
            end = time()
            print(f"Optimized {mol_id} in {end-start} seconds")

            if bonds_maintained(mol) and isoctahedral(mol):
                # Store the molecule in XYZ format in the DataFrame
                print("Go through check and will write to xyz")
                df.at[row_index, "xyz"] = MolToXYZBlock(mol)
                break
            else:
                print(f"{mol_id} failed geometry check")
                df.at[row_index, "xyz"] = "failed"
                break

        except InputError:
            print(f"InputError for {mol_id}, will attempt {3 - (attempt+1)} more times")
            sleep(10)
            continue
        except ValueError:
            print(
                f"ValueError, probably because ConstrainedEmbed couldn't embed {mol_id}"
            )
            df.at[row_index, "xyz"] = "failed"
            break
        except Exception as e:
            print(f"Other problem with {mol_id}: {str(e)}")
            df.at[row_index, "xyz"] = "failed"
            break


def optimize_geometries(df: pd.DataFrame) -> pd.DataFrame:

    if "xyz" not in df.columns:
        df["xyz"] = None
    for index, row in df.iterrows():
        mol_id = row["ligand_identifier"]
        mol = row["structure"]

        if mol is None or Chem.MolFromMolBlock(mol).GetNumAtoms() > 200:
            logging.warning(f"Molecule {mol_id} is invalid or too large.")
            df.at[index, "xyz"] = "failed"
            continue

        if df.at[index, "xyz"] is not None:
            logging.info(f"Molecule {mol_id} already processed or previously failed.")
            continue

        optimize(df, index, "fac")

    return df


OptimizeGeometriesTask = PipelineTask(
    "optimize_geometries",
    optimize_geometries,
    batch_size=10,
)
