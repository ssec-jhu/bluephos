import logging
from time import sleep

import pandas as pd
from ase.calculators.calculator import InputError
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, MolToXYZBlock

from bluephos.modules.annotate_rdkit_with_ase import optimize_geometry
from bluephos.modules.bond_length import bonds_maintained
from bluephos.modules.isoctahedral import isoctahedral
from bluephos.modules.octahedral_embed import octahedral_embed

try:
    from xtb.ase.calculator import XTB
except ImportError:
    XTB = None
    logging.warning("xtb not installed. Limited functionality available.")


logger = logging.getLogger(__name__)


# Timer code left blank pending Alexander's clarification on timing requirements
def optimize(row):
    mol_id = row["ligand_identifier"]
    existing_xyz = row.get("xyz")

    # Check if already processed or failed
    # TODO: Reconsider/define data source for tracking processed/failed items post-pipeline.
    if existing_xyz is not None and existing_xyz != "":
        if existing_xyz == "failed":
            logger.info(f"Molecule {mol_id} previously failed. Skipping optimization.")
        else:
            logger.info(f"Molecule {mol_id} already processed. Skipping optimization.")
        return existing_xyz

    mol = row["structure"]

    if pd.isna(mol) or Chem.MolFromMolBlock(mol) is None or Chem.MolFromMolBlock(mol).GetNumAtoms() > 200:
        logger.warning(f"Molecule {mol_id} is invalid or too large.")
        return "failed"

    mol = AddHs(Chem.MolFromMolBlock(mol))
    isomer = "fac"  # Example isomer, can be dynamically set if needed
    mol_id = mol.GetProp("_Name") + f"_{isomer}"
    mol.SetProp("_Name", mol_id)

    for attempt in range(3):
        try:
            octahedral_embed(mol, isomer)
            if XTB is not None:
                optimize_geometry(mol, XTB(method="GFN2-xTB"), conformation_index=0, uhf=2)
            else:
                logger.error("Proceeding without xTB functionality.")

            if bonds_maintained(mol) and isoctahedral(mol):
                # Store the molecule in XYZ format in the DataFrame
                logger.info("Go through check and will write to xyz")
                return MolToXYZBlock(mol)
            else:
                logger.error(f"{mol_id} failed geometry check on attemp {attempt + 1}")
                return "failed"

        except InputError:
            logger.error(f"InputError for {mol_id}, will attempt {3 - (attempt+1)} more times")
            sleep(10)
            continue
        except ValueError:
            logger.error(f"ValueError, probably because ConstrainedEmbed couldn't embed {mol_id}")
            return "failed"
        except Exception as e:
            logger.exception(f"Unhandled exception for {mol_id}: {str(e)}")
            return "failed"


def optimize_geometries(df: pd.DataFrame) -> pd.DataFrame:
    if "xyz" not in df.columns:
        df["xyz"] = None
    return df.assign(xyz=df.apply(optimize, axis=1))


OptimizeGeometriesTask = PipelineTask(
    "optimize_geometries",
    optimize_geometries,
    batch_size=10,
)
