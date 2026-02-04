import argparse
import csv
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, ClassVar

import torch
from atomworks.io.transforms.atom_array import remove_waters
from biotite.structure import AtomArray, AtomArrayStack
from joblib import delayed, Parallel
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density import XMap_torch
from sampleworks.eval.structure_utils import apply_selection
from sampleworks.utils.atom_array_utils import (
    AltlocInfo,
    detect_altlocs,
    keep_amino_acids,
    keep_polymer,
    load_structure_with_altlocs,
    remove_hydrogens,
    save_structure_to_cif,
)
from sampleworks.utils.density_utils import compute_density_from_atomarray
from sampleworks.utils.torch_utils import try_gpu


@dataclass
class BatchRow:
    """A row from the batch processing CSV file.

    Attributes
    ----------
    filename
        Path to the structure file (relative to base_dir)
    selection
        Optional atom selection string in pyMOL-like syntax (e.g., 'chain A and resi 10-50')
    occ_values
        Custom occupancy values for altlocs, must be in range [0.0, 1.0]
    mapfile
        Optional custom output filename for the density map
    """

    VALID_EXTENSIONS: ClassVar[frozenset[str]] = frozenset({".pdb", ".cif", ".mmcif", ".ent"})

    filename: str
    selection: str | None = None
    occ_values: list[float] = field(default_factory=list)
    mapfile: str | None = None

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

        Parameters
        ----------
        row
            Dictionary with keys 'filename' (required), and optionally
            'selection', 'occ_values' (colon-separated), and 'mapfile'

        Returns
        -------
        BatchRow
            Validated batch row instance

        Raises
        ------
        KeyError
            If required 'filename' column is missing
        ValueError
            If occupancy values are invalid
        """
        if "filename" not in row:
            raise KeyError("CSV is missing required 'filename' column")

        occ_values: list[float] = []
        if row.get("occ_values"):
            occ_values = [float(v.strip()) for v in row["occ_values"].split(":")]

        return cls(
            filename=row["filename"],
            selection=row.get("selection") or None,
            occ_values=occ_values,
            mapfile=row.get("mapfile") or None,
        )


def assign_occupancies(
    atom_array: AtomArray | AtomArrayStack,
    altloc_info: AltlocInfo,
    mode: str,
    occ_values: list[float] | None = None,
) -> AtomArray | AtomArrayStack:
    """Assign occupancy values to atoms based on their altloc membership.

    Parameters
    ----------
    atom_array
        Structure to modify
    altloc_info
        Detected altloc information from detect_altlocs()
    mode
        Assignment mode: 'default' (no change), 'uniform' (1/n_altlocs each),
        or 'custom' (user-specified values)
    occ_values
        For 'custom' mode: list of occupancy values [0.0-1.0] assigned to altlocs
        in sorted order (e.g., [0.3, 0.7] assigns 0.3 to altloc 'A', 0.7 to 'B').
        If fewer values than altlocs, remaining altlocs get occupancy 0.

    Returns
    -------
    AtomArray | AtomArrayStack
        Modified structure with updated occupancies

    Raises
    ------
    ValueError
        If 'custom' mode is requested but no altlocs exist, or if occ_values
        is None in custom mode, or if any occupancy value is outside [0.0, 1.0]
    """
    if mode == "default":
        return atom_array

    if not altloc_info.altloc_ids:
        if mode == "custom":
            raise ValueError(
                "Custom occupancy mode was requested, but the structure has no altlocs."
            )
        logger.warning("No altlocs detected, using default occupancies")
        return atom_array

    result = atom_array.copy()
    occupancy = result.occupancy

    if mode == "uniform":
        n_altlocs = len(altloc_info.altloc_ids)
        uniform_occ = 1.0 / n_altlocs
        for altloc in altloc_info.altloc_ids:
            occupancy[altloc_info.atom_masks[altloc]] = uniform_occ

    elif mode == "custom":
        if occ_values is None:
            raise ValueError("occ_values required for custom mode")
        for i, v in enumerate(occ_values):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Occupancy value {v} at index {i} is out of range [0.0, 1.0]")

        if len(occ_values) != len(altloc_info.altloc_ids):
            logger.warning(
                f"Expected {len(altloc_info.altloc_ids)} occupancy values, got {len(occ_values)}. "
                "The missing values are automatically set to 0."
            )
            occ_values = occ_values + [0.0] * (len(altloc_info.altloc_ids) - len(occ_values))

        for altloc, occ in zip(sorted(altloc_info.altloc_ids), occ_values):
            occupancy[altloc_info.atom_masks[altloc]] = occ

    return cast(AtomArray, result)


def save_density(density: torch.Tensor, xmap_torch: XMap_torch, output_path: Path) -> None:
    """Save a density map to disk in CCP4 format.

    Parameters
    ----------
    density
        Computed density tensor
    xmap_torch
        XMap_torch object containing grid parameters
    output_path
        Path where the CCP4 map file will be written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xmap_torch.tofile(str(output_path), density)
    logger.info(f"Saved density map to {output_path}")


def load_batch_csv(csv_path: Path) -> list[BatchRow]:
    """Load and parse a CSV file for batch processing.

    Parameters
    ----------
    csv_path
        Path to CSV file with columns: filename (required), selection (optional),
        occ_values (optional), mapfile (optional)

    Returns
    -------
    list[BatchRow]
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
            rows.append(BatchRow.from_dict(row))
    return rows


def _process_single_row(
    row: BatchRow,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    em_mode: bool,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
) -> None:
    """Process a single structure row.

    Parameters
    ----------
    row
        BatchRow containing structure information
    base_dir
        Base directory for resolving relative structure file paths
    output_dir
        Directory where output density maps will be written
    resolution
        Map resolution in Angstroms
    em_mode
        If True, use electron scattering factors. If False, use X-ray factors.
    device
        PyTorch device for computation
    strip_hydrogens
        If True, remove hydrogen atoms before computing density.
    strip_waters
        If True, remove water molecules before computing density.
    strip_ligands
        If True, remove ligand molecules (non-water heteroatoms) before computing density. This is
        done by keeping only polymer atoms in the selection, which are typically not ligands.
        TODO: be more thorough with this? We could make this a transform
    save_structure
        If True, save the processed structure to a CIF file in the input directory.
    """
    structure_path = base_dir / row.filename
    if not structure_path.exists():
        logger.error(f"Structure not found: {structure_path}")
        return

    try:
        atom_array = load_structure_with_altlocs(structure_path)
    except Exception as e:
        logger.error(
            f"Failed to load {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    try:
        atom_array = apply_selection(atom_array, row.selection)
    except ValueError as e:
        logger.error(f"Selection error for {row.filename}: {e}")
        return

    if strip_hydrogens:
        atom_array = remove_hydrogens(atom_array)

    if strip_waters:
        atom_array = remove_waters(atom_array)

    if strip_ligands:
        atom_array = keep_polymer(keep_amino_acids(atom_array))

    altloc_info = detect_altlocs(atom_array)  # pyright: ignore[reportArgumentType]
    if row.occ_values:
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, "custom", row.occ_values)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {row.filename}: {e}")
            return

    if save_structure:
        structure_output_path = structure_path.parent / f"{structure_path.stem}_density_input.cif"
        try:
            save_structure_to_cif(atom_array, structure_output_path)
            logger.info(f"Saved processed structure to {structure_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save structure for {row.filename} ({type(e).__name__}): {e}\n"
                f"{''.join(traceback.format_tb(e.__traceback__))}"
            )

    try:
        density, xmap_torch = compute_density_from_atomarray(
            atom_array, resolution=resolution, em_mode=em_mode, device=device
        )
    except Exception as e:
        logger.error(
            f"Failed to compute density for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    if row.mapfile:
        output_path = output_dir / row.mapfile
    else:
        output_path = output_dir / f"{structure_path.stem}_{resolution:.2f}A.ccp4"

    try:
        save_density(density, xmap_torch, output_path)
    except Exception as e:
        logger.error(
            f"Failed to save density for {row.filename} to {output_path} "
            f"({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    em_mode: bool,
    device: torch.device,
    n_jobs: int = -1,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
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
        Directory where output density maps will be written
    resolution
        Map resolution in Angstroms
    em_mode
        If True, use electron scattering factors. If False, use X-ray factors.
    device
        PyTorch device for computation
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    strip_hydrogens
        If True, remove hydrogen atoms before computing density.
    strip_waters
        If True, remove water molecules before computing density.
    strip_ligands
        If True, remove ligand molecules (non-water heteroatoms) before computing density.
    save_structure
        If True, save the processed structure to a CIF file in the input directory.
    """
    rows = load_batch_csv(csv_path)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {n_jobs} jobs")

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row,
            base_dir,
            output_dir,
            resolution,
            em_mode,
            device,
            strip_hydrogens,
            strip_waters,
            strip_ligands,
            save_structure,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic electron density maps from atomic structures"
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
        help="Base directory for relative paths in CSV",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection (e.g., 'chain A and resi 10-50' or 'chain A and resi 10')",
    )

    occ_group = parser.add_argument_group("Occupancy Options")
    occ_group.add_argument(
        "--occ-mode",
        choices=["default", "uniform", "custom"],
        default="default",
        help="Occupancy assignment mode",
    )
    occ_group.add_argument(
        "--occ-values",
        type=str,
        help="Colon-separated occupancy values for custom mode (e.g., '0.3:0.7')",
    )

    density_group = parser.add_argument_group("Density Options")
    density_group.add_argument(
        "--resolution", "-r", type=float, default=2.0, help="Map resolution in Angstroms"
    )
    density_group.add_argument(
        "--em-mode", action="store_true", help="Use electron scattering factors (EM mode)"
    )
    density_group.add_argument(
        "--remove-hydrogens",
        action="store_true",
        help="Remove hydrogen atoms before computing density",
    )
    density_group.add_argument(
        "--remove-waters",
        action="store_true",
        help="Remove water molecules before computing density",
    )
    density_group.add_argument(
        "--remove-ligands",
        action="store_true",
        help="Remove ligand molecules (non-water heteroatoms) before computing density",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the processed structure (after selection, occupancy assignment) to CIF",
    )
    output_group.add_argument("--output", "-o", type=Path, help="Output CCP4 map file path")
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
            args.batch_csv,
            args.base_dir,
            args.output_dir,
            args.resolution,
            args.em_mode,
            device,
            args.n_jobs,
            args.remove_hydrogens,
            args.remove_waters,
            args.remove_ligands,
            args.save_structure,
        )
    elif args.structure:
        atom_array = load_structure_with_altlocs(args.structure)
        atom_array = apply_selection(atom_array, args.selection)

        if args.remove_hydrogens:
            atom_array = remove_hydrogens(atom_array)

        if args.remove_waters:
            atom_array = remove_waters(atom_array)

        if args.remove_ligands:
            atom_array = keep_polymer(atom_array)

        altloc_info = detect_altlocs(atom_array)  # pyright: ignore[reportArgumentType]
        occ_values = (
            [float(v.strip()) for v in args.occ_values.split(":")] if args.occ_values else None
        )
        atom_array = assign_occupancies(atom_array, altloc_info, args.occ_mode, occ_values)

        if args.save_structure:
            structure_output_path = (
                args.structure.parent / f"{args.structure.stem}_density_input.cif"
            )
            save_structure_to_cif(atom_array, structure_output_path)
            logger.info(f"Saved processed structure to {structure_output_path}")

        density, xmap_torch = compute_density_from_atomarray(
            atom_array, resolution=args.resolution, em_mode=args.em_mode, device=device
        )

        output_path = (
            args.output or args.output_dir / f"{args.structure.stem}_{args.resolution:.2f}A.ccp4"
        )
        save_density(density, xmap_torch, output_path)
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
