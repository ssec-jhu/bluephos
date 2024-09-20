import bluephos.modules.log_config as log_config
from time import sleep
import pandas as pd
from ase.calculators.calculator import InputError
from dplutils.pipeline import PipelineTask
from rdkit import Chem
from rdkit.Chem import AddHs, MolToXYZBlock
from bluephos.modules.annotate_rdkit_with_ase import optimize_geometry, annotate_molecule_property
from bluephos.modules.bond_length import bonds_maintained
from bluephos.modules.isoctahedral import isoctahedral
from bluephos.modules.octahedral_embed import octahedral_embed

# Setup logging and get a logger instance
logger = log_config.setup_logging(__name__)

try:
    from xtb.ase.calculator import XTB
except ImportError:
    XTB = None
    logger.warning("xtb not installed. Limited functionality available.")


def calculate_ste(mol):
    try:
        calc = XTB(method="GFN2-xTB")
        annotate_molecule_property(mol, "singlet_energy", calc, lambda x: x.get_potential_energy(), uhf=0)
        annotate_molecule_property(mol, "triplet_energy", calc, lambda x: x.get_potential_energy(), uhf=2)
        return mol.GetDoubleProp("triplet_energy") - mol.GetDoubleProp("singlet_energy")
    except Exception:
        logger.exception("Failed to calculate singlet-triplet energy gap ")
        return None


def optimize(row, xtb):
    mol_id = row["ligand_identifier"]
    z = row["z"]
    ste = row["ste"]

    # Log the values of z and ste for debugging
    logger.info(f"Processing molecule {mol_id} ...")

    # # Skip processing based on conditions
    # if z is None or abs(z) >= t_nn or ste is not None:
    #     logger.info(f"Skipping xTB optimization on molecule {mol_id} based on z or t_ste conditions.")
    #     return row  # Return the row unchanged

    mol = row["structure"]

    mol_obj = None if pd.isna(mol) else Chem.MolFromMolBlock(mol)
    if not mol_obj or mol_obj.GetNumAtoms() > 200:
        logger.warning(f"Molecule {mol_id} is invalid or too large.")
        row["xyz"] = "failed"
        row["ste"] = None
        return row  # Return the updated row

    mol = AddHs(Chem.MolFromMolBlock(mol))
    isomer = "fac"  # Example isomer, can be dynamically set if needed
    mol_id = mol.GetProp("_Name") + f"_{isomer}"
    mol.SetProp("_Name", mol_id)

    for attempt in range(3):
        try:
            octahedral_embed(mol, isomer)
            if xtb:
                if XTB is not None:
                    logger.info("Proceeding with xTB functionality...")
                    optimize_geometry(mol, XTB(method="GFN2-xTB"), conformation_index=0, uhf=2)
                else:
                    logger.error("Cannot proceed with xTB functionality because xtb is not installed.")
            else:
                logger.info("Proceeding without xTB functionality...")

            if bonds_maintained(mol) and isoctahedral(mol):
                # Store the molecule in XYZ format in the DataFrame
                logger.info(f"{mol_id} Go through check and will write to xyz")
                xyz_block = MolToXYZBlock(mol)
                ste_score = calculate_ste(mol)
                row["xyz"] = xyz_block
                row["ste"] = ste_score
                return row  # Return the updated row
            else:
                logger.error(f"{mol_id} failed geometry check on attempt {attempt + 1}")
                row["xyz"] = "failed"
                row["ste"] = None
                return row  # Return the updated row

        except InputError:
            logger.error(f"InputError for {mol_id}, will attempt {3 - (attempt+1)} more times")
            sleep(10)
            continue
        except ValueError:
            logger.error(f"ValueError, probably because ConstrainedEmbed couldn't embed {mol_id}")
            row["xyz"] = "failed"
            row["ste"] = None
            return row  # Return the updated row
        except Exception as e:
            logger.exception(f"Unhandled exception for {mol_id}: {str(e)}")
            row["xyz"] = "failed"
            row["ste"] = None
            return row  # Return the updated row


def optimize_geometries(df: pd.DataFrame, xtb: bool) -> pd.DataFrame:
    for col in ["xyz", "ste"]:
        if col not in df.columns:
            df[col] = None

    # Apply the optimize function to each row
    df = df.apply(optimize, axis=1, xtb=xtb)

    return df


OptimizeGeometriesTask = PipelineTask(
    "optimize_geometries",
    optimize_geometries,
    context_kwargs={
        "xtb": "xtb",
    },
    batch_size=1,
    num_cpus=1,
)
