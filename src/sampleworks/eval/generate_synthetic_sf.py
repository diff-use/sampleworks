"""Generate synthetic structure factor amplitudes via SFcalculator-torch.

Produces an MTZ file of |F_protein| (no bulk solvent, no scaling) for each input
PDB/mmCIF structure. v1 scope: protein-only structure factors, with optional
per-row override of unit cell + space group and optional R-free flag column.
"""

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch
from loguru import logger
from sampleworks.utils.atom_array_utils import AltlocInfo, BLANK_ALTLOC_IDS
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser


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
        Optional Hermann-Mauguin space group string to override the one in the
        structure file (e.g., 'P 21 21 21')
    occ_values
        Custom list of occupancy values for altlocs, must be in range [0.0, 1.0]
    """

    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".pdb", ".cif", ".mmcif", ".ent"})

    filename: str
    mtzfile: str | None = None
    unit_cell: gemmi.UnitCell | None = None
    space_group: gemmi.SpaceGroup | None = None
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
        separated by ':' — a:b:c:alpha:beta:gamma), space_group.
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

        space_group: gemmi.SpaceGroup | None = None
        if row.get("space_group"):
            space_group = row["space_group"]
            if space_group.isdigit():
                space_group = int(space_group)
            space_group = gemmi.SpaceGroup(space_group)

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


def load_structure_with_cell_and_space_group(
    structure_path: Path,
    unit_cell: gemmi.UnitCell | None = None,
    space_group: gemmi.SpaceGroup | None = None,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands_and_waters: bool = False,
) -> gemmi.Structure:
    """Load a structure file and optionally override its unit cell / space group."""
    structure = gemmi.read_structure(str(structure_path))
    if unit_cell is not None:
        structure.cell = unit_cell
    if space_group is not None:
        structure.spacegroup_hm = (
            space_group.hm if isinstance(space_group, gemmi.SpaceGroup) else space_group
        )
    if strip_hydrogens:
        structure.remove_hydrogens()
    if strip_waters:
        structure.remove_waters()
    if strip_ligands_and_waters:
        structure.remove_ligands_and_waters()
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


def detect_altlocs_from_sfcpdbparser(sfc_pdbparser: PDBParser) -> AltlocInfo:
    """Detect alternate conformations from SFC's PDBParser instance.

    Mirror sampleworks.utils.atom_array_utils.detect_altlocs(). Excludes
    blank values defined in BLANK_ALTLOC_IDS.

    Parameters
    ----------
    sfc_pdbparser
        SFcalculator instance whose internal PDBParser holds per-atom altloc labels

    Returns
    -------
    AltlocInfo
        Sorted list of detected altloc IDs and per-altloc boolean atom masks
    """
    altloc_arr: np.ndarray[Any, np.dtype[np.str_]] = np.array(sfc_pdbparser.atom_altloc)
    altloc_ids = sorted(v for v in set(altloc_arr) if v not in BLANK_ALTLOC_IDS)
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {
        altloc: altloc_arr == altloc for altloc in altloc_ids
    }
    return AltlocInfo(altloc_ids=altloc_ids, atom_masks=atom_masks)


def assign_occupancies(
    sfc_pdbparser: PDBParser,
    altloc_info: AltlocInfo,
    mode: str,
    occ_values: list[float] | None = None,
) -> None:
    """Assign occupancy values to atoms based on their altloc membership.

    Mirror assign_occupancies in generate_synthetic_density.py.

    Parameters
    ----------
    sfc_pdbparser
        PDBParser (SFC's parser) instance whose atom_altloc array will be modified.
    altloc_info
        AltlocInfo object from detect_altlocs_from_sfcpdbparser()
    mode
        Assignment mode: 'default' (no change), 'uniform' (1/n_altlocs each),
        or 'custom' (user-specified values)
    occ_values
        For 'custom' mode: list of occupancy values [0.0-1.0] assigned to altlocs
        in sorted order (e.g., [0.3, 0.7] assigns 0.3 to altloc 'A', 0.7 to 'B').
        If fewer values than altlocs, remaining altlocs get occupancy 0.

    Raises
    ------
    ValueError
        If 'custom' mode is requested but no altlocs exist, or if occ_values
        is None in custom mode, or if any occupancy value is outside [0.0, 1.0]
    """
    if mode == "default":
        return

    if not altloc_info.altloc_ids:
        if mode == "custom":
            raise ValueError(
                "Custom occupancy mode was requested, but the structure has no altlocs."
            )
        logger.warning("No altlocs detected, using default occupancies")
        return

    if mode == "uniform":
        uniform_occ = 1.0 / len(altloc_info.altloc_ids)
        for altloc in altloc_info.altloc_ids:
            mask = altloc_info.atom_masks[altloc]
            sfc_pdbparser.atom_occ[mask] = uniform_occ

    elif mode == "custom":
        if occ_values is None:
            raise ValueError("occ_values required for custom mode")
        for i, v in enumerate(occ_values):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Occupancy value {v} at index {i} is out of range [0.0, 1.0]")

        if len(occ_values) > len(altloc_info.altloc_ids):
            raise ValueError(
                f"Too many occupancy values: got {len(occ_values)}, "
                f"but structure has {len(altloc_info.altloc_ids)} altloc(s) "
                f"({', '.join(altloc_info.altloc_ids)})."
            )
        if len(occ_values) < len(altloc_info.altloc_ids):
            logger.warning(
                f"Expected {len(altloc_info.altloc_ids)} occupancy values, got {len(occ_values)}. "
                "The missing values are automatically set to 0."
            )
            occ_values = occ_values + [0.0] * (len(altloc_info.altloc_ids) - len(occ_values))

        for altloc, occ in zip(sorted(altloc_info.altloc_ids), occ_values):
            mask = altloc_info.atom_masks[altloc]
            sfc_pdbparser.atom_occ[mask] = occ
    else:
        raise ValueError(f"Invalid occupancy mode '{mode}'")


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
    strip_ligands_and_waters: bool = False,
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
    occ_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform' (1/n_altlocs each),
        or 'custom' (use per-row occ_values from BatchRow).
    strip_ligands_and_waters
        If True, remove ligand molecules and water molecules before computing structure factors.
        Default is False. Note that this parameter is different from strip_ligands in
        generate_synthetic_density.py because gemmi does not support removing only non-water
        heteroatoms.
    """
    structure_path = base_dir / row.filename
    if not structure_path.exists():
        logger.error(f"Structure not found: {structure_path}")
        return

    try:
        # load structure with unit cell / space group
        structure = load_structure_with_cell_and_space_group(
            structure_path,
            row.unit_cell,
            row.space_group,
            strip_hydrogens,
            strip_waters,
            strip_ligands_and_waters,
        )
        sfc_pdbparser = PDBParser(structure)
        # detect altlocs and re-assign occupancies
        altloc_info = detect_altlocs_from_sfcpdbparser(sfc_pdbparser)
        if row.occ_values:
            if occ_mode != "custom":
                logger.warning(
                    f"Custom occupancy values provided for {row.filename}, "
                    f"but occ_mode is '{occ_mode}'. Using 'custom' mode."
                )
            assign_occupancies(sfc_pdbparser, altloc_info, "custom", row.occ_values)
        else:
            assign_occupancies(sfc_pdbparser, altloc_info, occ_mode)
        # compute structure factors
        sfc = SFcalculator(
            pdbmodel=sfc_pdbparser,
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
    # output MTZ file
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
