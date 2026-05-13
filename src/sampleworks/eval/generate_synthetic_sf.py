"""Generate synthetic structure factor amplitudes via SFcalculator-torch.

Produces an MTZ file of |F_protein| (no bulk solvent, no scaling) for each input
PDB/mmCIF structure. v1 scope: protein-only structure factors, with optional
per-row override of unit cell + space group and optional R-free flag column.
"""

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray
from loguru import logger
from sampleworks.utils.atom_array_utils import (
    assign_occupancies,
    BLANK_ALTLOC_IDS,
    detect_altlocs,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    remove_hydrogens,
)
from SFC_Torch import SFcalculator
from SFC_Torch.io import array2hier, PDBParser


@dataclass
class BatchRow:
    """A row from the batch processing CSV file.

    Attributes
    ----------
    filename
        Path to the structure file (relative to base_dir)
    mtzfile
        Optional custom output filename for the MTZ
    unit_cell
        Optional unit cell to override the one in the structure file
    space_group
        Optional space group (in Hermann-Mauguin string format) to override the
        one in the structure file.
    occ_values
        Custom list of occupancy values for altlocs, must be in range [0.0, 1.0]
    """

    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".pdb", ".cif", ".mmcif", ".ent"})

    filename: str
    mtzfile: str | None = None
    unit_cell: gemmi.UnitCell | None = None
    space_group: str | None = None
    occ_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        ext = Path(self.filename).suffix.lower()
        if ext not in self.VALID_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension '{ext}' for '{self.filename}'. "
                f"Expected one of: {', '.join(sorted(self.VALID_EXTENSIONS))}"
            )
        for i, v in enumerate(self.occ_values):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Occupancy value {v} at index {i} is out of range [0.0, 1.0]")

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "BatchRow":
        """Create a BatchRow from a CSV row dictionary.

        CSV columns: filename (required), mtzfile, unit_cell (six floats
        separated by ':' — a:b:c:alpha:beta:gamma), space_group (space group number or
        Hermann-Mauguin string), occ_values (colon-separated, e.g. '0.3:0.7').
        """
        if "filename" not in row:
            raise KeyError("CSV is missing required 'filename' column")

        unit_cell: gemmi.UnitCell | None = None
        if row.get("unit_cell"):
            parts = [float(v.strip()) for v in row["unit_cell"].split(":")]
            if len(parts) != 6:
                raise ValueError(
                    f"unit_cell must be 6 colon-separated values (a:b:c:alpha:beta:gamma), "
                    f"got {len(parts)}: {row['unit_cell']!r}"
                )
            unit_cell = gemmi.UnitCell(*parts)

        space_group: str | None = None
        if row.get("space_group"):
            space_group = row["space_group"]
            if space_group.isdigit():
                space_group = gemmi.SpaceGroup(int(space_group)).hm

        occ_values: list[float] = []
        if row.get("occ_values"):
            occ_values = [float(v.strip()) for v in row["occ_values"].split(":")]

        return cls(
            filename=row["filename"],
            mtzfile=row.get("mtzfile") or None,
            unit_cell=unit_cell,
            space_group=space_group,
            occ_values=occ_values,
        )


def atomarray_to_gemmi(
    atom_array: AtomArray,
    unit_cell: gemmi.UnitCell | None = None,
    space_group: str | None = None,
) -> gemmi.Structure:
    """Convert a biotite AtomArray to a gemmi.Structure for SFcalculator.

    Anisotropic B-factors are set to zero since biotite does not store them.
    Blank altloc labels are converted from biotite's '' to gemmi's '\\x00'.

    Parameters
    ----------
    atom_array
        Input structure with occupancy and b_factor annotations
    unit_cell
        Crystallographic unit cell for the structure. If None, gemmi defaults
        to (1.0, 1.0, 1.0, 90.0, 90.0, 90.0).
    space_group
        Space group (in Hermann-Mauguin string format) for the structure. If
        empty or invalid, SFcalculator defaults to P1.

    Returns
    -------
    gemmi.Structure
        Structure ready to be wrapped by SFC_Torch.io.PDBParser
    """
    n = len(atom_array)
    cra_names = [
        f"{atom_array.chain_id[i]}-0-{atom_array.res_name[i]}-{atom_array.atom_name[i]}"
        for i in range(n)
    ]
    # gemmi uses '\x00' for blank altloc; biotite uses ''
    atom_altloc = ["\x00" if a in BLANK_ALTLOC_IDS else a for a in atom_array.altloc_id]
    structure = array2hier(
        atom_pos=atom_array.coord,
        atom_b_aniso=np.zeros((n, 3, 3), dtype=np.float64),
        atom_b_iso=atom_array.b_factor,
        atom_occ=atom_array.occupancy,
        atom_name=atom_array.element,
        cra_name=cra_names,
        atom_altloc=atom_altloc,
        res_id=atom_array.res_id,
    )
    if unit_cell is not None:
        structure.cell = unit_cell
    if space_group is not None:
        structure.spacegroup_hm = space_group
    return structure


def write_amplitudes_to_mtz(
    sfc: SFcalculator,
    output_path: Path,
    test_fraction: float = 0.05,
    seed: int | None = None,
    hkl_attr: str = "Hasu_array",
    f_attr: str = "Ftotal_asu",
    ccp4_convention: bool = False,
    sigf_scale: float = 0.2,
) -> None:
    """Build an rs.DataSet from SFcalculator outputs and write it as MTZ.

    Parameters
    ----------
    sfc: SFcalculator
        SFcalculator instance
    output_path: Path
        Path to the output MTZ file
    test_fraction: float
        Fraction of reflections to mark as R-free test set (0 disables)
    seed: int | None
        Optional seed for reproducible R-free flag assignment
    hkl_attr: str
        Attribute name in SFcalculator for hkl indices
    f_attr: str
        Attribute name in SFcalculator for structure factors
    ccp4_convention: bool
        If True, use CCP4 convention for R-free flag assignment. Default
        is False, which uses Phenix convention (1 = test, 0 = working).
    sigf_scale: float
        Scale factor to make a fake SIGFP column from FP values so that
        SFcalculator can load the output MTZ file without errors. Default
        is 0.2. The actual SIGFP values only matter when computing R-factor.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = sfc.prepare_dataset(hkl_attr, f_attr)
    dataset.rename(columns={"FMODEL": "FP"}, inplace=True)
    dataset["SIGFP"] = dataset["FP"] * sigf_scale
    dataset["SIGFP"] = dataset["SIGFP"].astype(rs.StandardDeviationDtype())
    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset,
            ccp4_convention=ccp4_convention,
            fraction=test_fraction,
            seed=seed,
        )
    dataset.write_mtz(str(output_path))
    logger.info(f"Saved structure factors to {output_path}")


def _process_single_row(
    row: BatchRow,
    base_dir: Path,
    output_dir: Path,
    dmin: float,
    mode: str,
    anomalous: bool,
    occ_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    simulate_solvent_and_scale: bool = False,
) -> None:
    """Compute synthetic protein structure factors for a single structure.

    Parameters
    ----------
    row
        BatchRow describing the input structure and optional per-row overrides
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output MTZ files will be written
    dmin
        High-resolution limit in Angstroms
    mode
        SFcalculator mode: "xray" or "cryoem"
    anomalous
        If True, include anomalous scattering (keeps Friedel pairs separate)
    occ_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform'
        (1/n_altlocs each), or 'custom' (use per-row occ_values from BatchRow)
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables)
    seed
        Optional seed for reproducible R-free flag assignment
    device
        PyTorch device for SFcalculator
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors. Default is False.
    strip_waters
        If True, remove water molecules before computing structure factors. Default is False.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
        Equivalent to strip_ligands in generate_synthetic_density.py.
    """
    structure_path = base_dir / row.filename
    if not structure_path.exists():
        logger.error(f"Structure not found: {structure_path}")
        return

    # Load structure and strip off unwanted atoms
    try:
        atom_array = load_structure_with_altlocs(structure_path)
    except Exception as e:
        logger.error(
            f"Failed to load {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return
    atom_array = remove_hydrogens(atom_array) if strip_hydrogens else atom_array
    atom_array = remove_waters(atom_array) if strip_waters else atom_array
    atom_array = keep_polymer(keep_amino_acids(atom_array)) if strip_ligands else atom_array

    # Altloc detection and occupancy assignment (reused from density script)
    altloc_info = detect_altlocs(atom_array)  # ty: ignore[invalid-argument-type]
    if row.occ_values:
        if occ_mode != "custom":
            logger.warning(
                f"Custom occupancy values provided for {row.filename}, "
                f"but occ_mode is '{occ_mode}'. Using 'custom' mode."
            )
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, "custom", row.occ_values)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {row.filename}: {e}")
            raise
    elif occ_mode in {"uniform", "default"}:
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, occ_mode)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {row.filename}: {e}")
            raise
    else:
        logger.error(f"Invalid occupancy mode '{occ_mode}' for {row.filename}")
        raise ValueError(f"Invalid occupancy mode '{occ_mode}'")

    # Convert to gemmi and initialize SFcalculator
    try:
        unit_cell = row.unit_cell
        space_group = row.space_group
        if unit_cell is None or space_group is None:
            gemmi_meta = gemmi.read_structure(str(structure_path))
            if unit_cell is None:
                unit_cell = gemmi_meta.cell
            if space_group is None:
                space_group = gemmi_meta.spacegroup_hm
        gemmi_structure = atomarray_to_gemmi(atom_array, unit_cell, space_group)
    except Exception as e:
        logger.error(
            f"Failed to convert {row.filename} to gemmi ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    # Compute structure factors
    try:
        sfc = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=dmin,
            mode=mode,
            anomalous=anomalous,
            set_experiment=False,
            device=device,
        )
        sfc.calc_fprotein()
        if simulate_solvent_and_scale:
            sfc.inspect_data()
            sfc.calc_fsolvent()
            sfc.init_scales(requires_grad=False)
            sfc.calc_ftotal()
            F_attribute = "Ftotal_asu"
        else:
            F_attribute = "Fprotein_asu"
        logger.info(f"Computed {F_attribute} for {row.filename} on {device}")
    except Exception as e:
        logger.error(
            f"Failed to compute for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    # Output MTZ file
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{dmin:.2f}A.mtz")
    try:
        write_amplitudes_to_mtz(
            sfc, output_path, f_attr=F_attribute, test_fraction=test_fraction, seed=seed
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
