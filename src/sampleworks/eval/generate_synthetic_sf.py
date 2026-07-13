"""Generate synthetic structure factor amplitudes via SFcalculator-torch.

Produces an MTZ file of |Fmodel| (or |Fprotein| if not simulate solvent and
scale) for each input PDB/mmCIF structure. The MTZ file has dummy values for
SIGFP and optionally R-free flag column. Each structure can be optionally
overridden with unit cell, space group, atom selection, and occupancy.
"""

import argparse
import csv
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import gemmi
import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.utils
import torch
from biotite.structure import AtomArray
from loguru import logger
from sampleworks.eval.synthetic_utils import (
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
    validate_occupancy_values,
)
from sampleworks.utils.atom_array_utils import BLANK_ALTLOC_IDS
from sampleworks.utils.torch_utils import try_gpu
from SFC_Torch import SFcalculator
from SFC_Torch.io import array2hier, PDBParser


@dataclass
class BatchRowForMTZ:
    """A row from the batch processing CSV file. Each row represents one structure.

    Attributes
    ----------
    filename
        Path to the structure file (relative to base_dir, which is meant to be the
        parent directory of all structures.)
    mtzfile
        Optional custom output filename for the MTZ
    unit_cell
        Optional unit cell to override the one in the structure file
    space_group
        Optional space group to override the one in the structure file. Accepts any
        string (e.g. "P 21 21 2") or integer (space group number, e.g. 18) that gemmi
        can convert to a SpaceGroup object, which is then converted to the international
        notation (Hermann-Mauguin string). See gemmi documentation for more details
        (https://gemmi.readthedocs.io/en/latest/symmetry.html#spacegroup).
    selection
        Optional atom selection string in pyMOL-like syntax (e.g., 'chain A and resi 10-50')
    occupancy_values
        Custom list of occupancy values for altlocs, must be in range [0.0, 1.0]
    """

    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".cif", ".mmcif"})
    LEGACY_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".pdb", ".ent"})

    filename: Path | str
    mtzfile: str | None = None
    unit_cell: gemmi.UnitCell | None = None
    space_group: str | None = None
    selection: str | None = None
    occupancy_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        ext = Path(self.filename).suffix.lower()
        all_supported = self.VALID_EXTENSIONS | self.LEGACY_EXTENSIONS
        if ext not in all_supported:
            raise ValueError(
                f"Invalid file extension '{ext}' for '{self.filename}'. "
                f"Expected one of: {', '.join(sorted(all_supported))}"
            )
        if ext in self.LEGACY_EXTENSIONS:
            logger.warning(
                f"'{ext}' is a legacy PDB format and support may be removed in a future version. "
                "Prefer .cif or .mmcif (mmCIF format)."
            )
        validate_occupancy_values(self.occupancy_values)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BatchRowForMTZ":
        """Create a BatchRowForMTZ from a CSV row dictionary.

        CSV columns:
        - filename (required): e.g. '1abc.cif', relative to base_dir
        - mtzfile: e.g. 'output/1abc.mtz'
        - unit_cell (six floats separated by ':'): e.g. '1.0:1.0:1.0:90.0:90.0:90.0'
          in units of Angstroms and degrees
        - space_group (number or Hermann-Mauguin string): e.g. '18' or 'P 21 21 2'
        - occupancy_values (colon-separated floats summing to 1.0): e.g. '0.3:0.7'
        - selection (PyMOL-like syntax): e.g. 'chain A'
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

        occupancy_values: list[float] = []
        if row.get("occupancy_values"):
            occupancy_values = [float(v.strip()) for v in row["occupancy_values"].split(":")]

        return cls(
            filename=row["filename"],
            mtzfile=row.get("mtzfile") or None,
            unit_cell=unit_cell,
            space_group=space_group,
            selection=row.get("selection") or None,
            occupancy_values=occupancy_values,
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
        to (1.0, 1.0, 1.0, 90.0, 90.0, 90.0) in units of Angstroms and degrees.
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
    # gemmi uses '\x00' for blank altloc
    atom_altloc = ["\x00" if a in BLANK_ALTLOC_IDS else a for a in atom_array.altloc_id]
    structure: gemmi.Structure = array2hier(
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


def process_amplitudes_to_dataset(
    sfc: SFcalculator,
    test_fraction: float = 0.05,
    seed: int | None = None,
    miller_index_column: str = "Hasu_array",
    structure_factor_column: str = "Ftotal_asu",
    ccp4_convention: bool = False,
    sigma_f_scale: float = 0.2,
    output_path: Path | None = None,
) -> rs.DataSet:
    """Build an rs.DataSet from SFcalculator outputs, optionally writing it as MTZ.

    Parameters
    ----------
    sfc: SFcalculator
        SFcalculator instance
    test_fraction: float
        Fraction of reflections to mark as R-free test set (0 disables)
    seed: int | None
        Optional seed for reproducible R-free flag assignment
    miller_index_column: str
        Attribute name in SFcalculator for hkl indices
    structure_factor_column: str
        Attribute name in SFcalculator for structure factors
    ccp4_convention: bool
        If True, use CCP4 convention for R-free flag assignment. Default
        is False, which uses Phenix convention (1 = test, 0 = working).
    sigma_f_scale: float
        Scale factor to make a fake sigma column from the structure factor amplitude
        values so that SFcalculator can load the output MTZ file as synthetic reward
        without errors. The actual sigma values only matter when computing R-factor.
    output_path: Path | None
        If provided, write the dataset to this MTZ file path.

    Returns
    -------
    rs.DataSet
        Dataset with structure factor amplitudes, fake sigma column, and optionally
        R-free flags.
    """
    dataset: rs.DataSet = sfc.prepare_dataset(miller_index_column, structure_factor_column)
    # assumes the first detected column of dtype F is the structure factor amplitude column
    # avoids hardcoding unexposed column name "FMODEL" from sfc.prepare_dataset().
    structure_factor_amplitude_column = dataset.select_mtzdtype(
        rs.StructureFactorAmplitudeDtype()
    ).columns[0]
    sigma_f_column = f"SIG{structure_factor_amplitude_column}"
    dataset[sigma_f_column] = dataset[structure_factor_amplitude_column] * sigma_f_scale
    dataset[sigma_f_column] = dataset[sigma_f_column].astype(rs.StandardDeviationDtype())
    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset,
            ccp4_convention=ccp4_convention,
            fraction=test_fraction,
            seed=seed,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_mtz(str(output_path))
        logger.info(f"Saved structure factors to {output_path}")
    return dataset


def _process_single_row(
    row: BatchRowForMTZ,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    simulate_solvent_and_scale: bool = False,
    save_structure: bool = False,
) -> None:
    """Compute synthetic protein structure factors for a single structure.
    Assume no anomalous scattering.

    Parameters
    ----------
    row
        BatchRowForMTZ describing the input structure and optional per-row overrides
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output MTZ files will be written
    resolution
        High-resolution (dmin) limit in Angstroms
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem"
    occupancy_mode
        Occupancy assignment mode: 'default' (keep file values), 'uniform'
        (1/n_altlocs each), or 'custom' (use per-row occupancy_values from BatchRowForMTZ)
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
        If True, remove ligand molecules (non-water heteroatoms) before computing structure
        factors. Default is False.
    simulate_solvent_and_scale
        If True, compute bulk solvent and scale factors for Ftotal instead of Fprotein.
        Default is False.
    save_structure
        If True, save the processed structure (after selection and occupancy assignment)
        as mmCIF to output_dir. Unit cell and space group are preserved. Default is False.
    """
    structure_path = base_dir / row.filename
    atom_array = load_structure_for_synthetic_reward(
        structure_path,
        occupancy_mode=occupancy_mode,
        occupancy_values=row.occupancy_values,
        strip_hydrogens=strip_hydrogens,
        strip_waters=strip_waters,
        strip_ligands=strip_ligands,
        selection=row.selection,
    )
    if atom_array is None:
        return

    # Convert to gemmi for SFcalculator
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

    if save_structure:
        structure_output_path = output_dir / f"{structure_path.stem}_sf_input.cif"
        structure_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gemmi_structure.make_mmcif_document().write_file(str(structure_output_path))
            logger.info(f"Saved processed structure to {structure_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save structure for {row.filename} ({type(e).__name__}): {e}\n"
                f"{''.join(traceback.format_tb(e.__traceback__))}"
            )

    # Compute structure factors
    try:
        sfc = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=resolution,
            mode=scattering_factor_mode,
            anomalous=False,
            set_experiment=False,
            device=device,
        )
        logger.debug(
            f"SFC info for {row.filename}: cell: {sfc.unit_cell}, "
            f"space group: {sfc.space_group.hm}, "
            f"n_atoms: {len(sfc.atom_pos_orth)}"
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
        logger.debug(f"Computed {F_attribute} for {row.filename} on {device}")
    except Exception as e:
        logger.error(
            f"Failed to compute for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    # Output MTZ file
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{resolution:.2f}A.mtz")
    try:
        process_amplitudes_to_dataset(
            sfc,
            structure_factor_column=F_attribute,
            test_fraction=test_fraction,
            seed=seed,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )


def load_batch_csv(csv_path: Path) -> list[BatchRowForMTZ]:
    """Load and parse a CSV file for batch processing.

    Parameters
    ----------
    csv_path
        Path to CSV file with columns: filename (required), mtzfile, unit_cell,
        space_group, selection, occupancy_values (all optional)

    Returns
    -------
    list[BatchRowForMTZ]
        List of validated batch processing rows

    Raises
    ------
    KeyError
        If the CSV is missing the required 'filename' column
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise KeyError(f"CSV file '{csv_path}' is missing required 'filename' column")
        for row in reader:
            rows.append(BatchRowForMTZ.from_dict(row))
    return rows


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    occupancy_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    n_jobs: int = -1,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    simulate_solvent_and_scale: bool = False,
    save_structure: bool = False,
) -> None:
    """Process multiple structures from a CSV file in batch mode.

    Parameters
    ----------
    csv_path
        Path to CSV file listing structures to process
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output MTZ files will be written
    resolution
        High-resolution (dmin) limit in Angstroms
    scattering_factor_mode
        Scattering factor type: 'xray' or 'cryoem'
    occupancy_mode
        Occupancy assignment mode: 'default', 'uniform', or 'custom'
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables)
    seed
        Optional seed for reproducible R-free flag assignment
    device
        PyTorch device for computation
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    simulate_solvent_and_scale
        If True, compute bulk solvent and scale factors in addition to F_protein.
    save_structure
        If True, save each processed structure as mmCIF to output_dir.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {effective_n_jobs} jobs")

    Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            scattering_factor_mode=scattering_factor_mode,
            occupancy_mode=occupancy_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            simulate_solvent_and_scale=simulate_solvent_and_scale,
            save_structure=save_structure,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic structure factor amplitudes from atomic structures"
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--structure", "-s", type=Path, help="Path to input structure file (mmCIF or PDB)"
    )
    input_group.add_argument("--batch-csv", type=Path, help="Path to CSV file for batch processing")
    input_group.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths in CSV, not used in single-structure mode",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection (e.g., 'chain A and resi 10-50' or 'chain A and resi 10')",
    )

    occupancy_group = parser.add_argument_group("Occupancy Options")
    occupancy_group.add_argument(
        "--occupancy-mode",
        choices=["default", "uniform", "custom"],
        default="default",
        help="Occupancy assignment mode",
    )
    occupancy_group.add_argument(
        "--occupancy-values",
        type=str,
        help="Colon-separated occupancy values for custom mode (e.g., '0.3:0.7')",
    )

    sf_group = parser.add_argument_group("Structure Factor Options")
    sf_group.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=1.0,
        help="High-resolution (dmin) limit in Angstroms",
    )
    sf_group.add_argument(
        "--scattering-factor-mode",
        choices=["xray", "cryoem"],
        default="xray",
        help="Scattering factor type",
    )
    sf_group.add_argument(
        "--simulate-solvent-and-scale",
        action="store_true",
        help="Compute bulk solvent and overall scale factors (outputs Ftotal instead of Fprotein)",
    )
    sf_group.add_argument(
        "--remove-hydrogens",
        action="store_true",
        help="Remove hydrogen atoms before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-waters",
        action="store_true",
        help="Remove water molecules before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-ligands",
        action="store_true",
        help="Remove ligand molecules (non-water heteroatoms) before computing structure factors",
    )

    rfree_group = parser.add_argument_group("R-free Options")
    rfree_group.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Fraction of reflections flagged as R-free test set (0 disables)",
    )
    rfree_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible R-free flag assignment",
    )

    crystal_group = parser.add_argument_group("Crystal Options (single-structure mode only)")
    crystal_group.add_argument(
        "--unit-cell",
        type=str,
        help="Unit cell as 'a:b:c:alpha:beta:gamma' (overrides CRYST1 record)",
    )
    crystal_group.add_argument(
        "--space-group",
        type=str,
        help="Space group as Hermann-Mauguin string or number (overrides CRYST1 record)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the processed structure (after selection, occupancy assignment) to CIF",
    )
    output_group.add_argument("--output", "-o", type=Path, help="Output MTZ file path")
    output_group.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Output directory for batch mode"
    )

    parallel_group = parser.add_argument_group("Parallelization Options")
    parallel_group.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for batch processing (-1 uses all CPUs)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = try_gpu()

    if args.batch_csv:
        process_batch(
            csv_path=args.batch_csv,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            n_jobs=args.n_jobs,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            simulate_solvent_and_scale=args.simulate_solvent_and_scale,
            save_structure=args.save_structure,
        )
    elif args.structure:
        row = BatchRowForMTZ.from_dict(
            {
                "filename": args.structure.name,
                "mtzfile": args.output.name if args.output else None,
                "unit_cell": args.unit_cell,
                "space_group": args.space_group,
                "selection": args.selection,
                "occupancy_values": args.occupancy_values,
            }
        )
        _process_single_row(
            row=row,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else Path("."),
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            occupancy_mode=args.occupancy_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            simulate_solvent_and_scale=args.simulate_solvent_and_scale,
            save_structure=args.save_structure,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
