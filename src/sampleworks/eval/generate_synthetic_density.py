import argparse
import csv
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, ClassVar

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack
from joblib import delayed, Parallel
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    GridParameters,
    Resolution,
    XMap,
)
from sampleworks.core.rewards.real_space_density import (
    extract_density_inputs_from_atomarray,
    setup_scattering_params,
)
from sampleworks.eval.structure_utils import parse_selection_string
from sampleworks.utils.atom_array_utils import load_structure_with_altlocs
from sampleworks.utils.torch_utils import try_gpu


@dataclass
class AltlocInfo:
    """Information about alternate conformations (altlocs) in a structure.

    Attributes
    ----------
    altloc_ids
        Sorted list of altloc identifiers (e.g., ['A', 'B'])
    atom_masks
        Dictionary mapping each altloc ID to a boolean mask indicating which atoms
        belong to that altloc
    """

    altloc_ids: list[str]
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]


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


def apply_selection(atom_array: AtomArray, selection: str | None) -> AtomArray:
    """Apply an atom selection string to filter a structure.

    Parameters
    ----------
    atom_array
        Structure to filter
    selection
        Selection string (e.g., 'chain A and resi 10-50'). If None, returns
        the entire structure unchanged.

    Returns
    -------
    AtomArray
        Filtered structure containing only atoms matching the selection

    Raises
    ------
    ValueError
        If the selection string matches no atoms
    """
    if selection is None:
        return atom_array

    chain_id, resi_start, resi_end = parse_selection_string(selection)
    mask = np.ones(len(atom_array), dtype=bool)

    if chain_id is not None:
        mask &= atom_array.chain_id == chain_id

    if resi_start is not None:
        res_ids = cast(np.ndarray[Any, np.dtype[np.int64]], atom_array.res_id)
        if resi_end is not None:
            mask &= (res_ids >= resi_start) & (res_ids <= resi_end)
        else:
            mask &= res_ids == resi_start

    if mask.sum() == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms")

    return cast(AtomArray, atom_array[mask])


ALTLOC_DEFAULT_VALUES = {"", ".", " ", "?"}


def detect_altlocs(atom_array: AtomArray) -> AltlocInfo:
    """Detect alternate conformations in a structure.

    Identifies all non-default altloc IDs and creates boolean masks for each.
    Default values ("", ".", " ") are excluded from the detected altlocs.
    """
    if not hasattr(atom_array, "altloc_id"):
        return AltlocInfo(altloc_ids=[], atom_masks={})

    altloc_arr = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.altloc_id)
    altloc_ids = sorted(set(altloc_arr) - ALTLOC_DEFAULT_VALUES)
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]] = {}
    for altloc in altloc_ids:
        atom_masks[altloc] = altloc_arr == altloc

    return AltlocInfo(altloc_ids=altloc_ids, atom_masks=atom_masks)


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
            occupancy[altloc_info.atom_masks[altloc]] = uniform_occ  # pyright:ignore[reportOptionalSubscript]

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
            occupancy[altloc_info.atom_masks[altloc]] = occ  # pyright:ignore[reportOptionalSubscript]

    return cast(AtomArray, result)


def create_synthetic_grid(
    atom_array: AtomArray | AtomArrayStack, resolution: float, padding: float = 5.0
) -> XMap:
    """Create an empty density map grid sized to fit the structure.

    Parameters
    ----------
    atom_array
        Structure to create a grid for
    resolution
        Map resolution in Angstroms
    padding
        Extra space to add around the structure in each dimension (Angstroms)

    Returns
    -------
    XMap
        Empty density map with appropriate grid parameters and unit cell
        dimensions to contain the structure
    """
    coords = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.coord)
    if coords.ndim == 3:
        coords = coords.reshape(-1, 3)

    valid_mask = np.isfinite(coords).all(axis=1)
    coords = coords[valid_mask]

    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding
    extent = max_coords - min_coords

    # standard voxel spacing from Phenix, etc.
    voxel_spacing = resolution / 4.0
    grid_shape = np.ceil(extent / voxel_spacing).astype(int)

    unit_cell = UnitCell(
        a=float(extent[0]),
        b=float(extent[1]),
        c=float(extent[2]),
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        space_group="P1",
    )

    # (nz, ny, nx) ordering for array, since most fft routines will expect
    # z as the "fastest" axis, but CCP4 format uses X as the fastest axis
    empty_array = np.zeros((grid_shape[2], grid_shape[1], grid_shape[0]), dtype=np.float32)

    empty_xmap = XMap(
        empty_array,
        grid_parameters=GridParameters(voxelspacing=voxel_spacing),
        unit_cell=unit_cell,
        resolution=Resolution(high=resolution, low=1000.0),
        origin=min_coords,
    )

    return empty_xmap


def compute_density(
    atom_array: AtomArray | AtomArrayStack,
    resolution: float,
    em_mode: bool,
    device: torch.device,
) -> tuple[torch.Tensor, XMap_torch]:
    """Compute synthetic electron density from atomic coordinates.

    Parameters
    ----------
    atom_array
        Structure to compute density for
    resolution
        Map resolution in Angstroms
    em_mode
        If True, use electron scattering factors. If False, use X-ray factors.
    device
        PyTorch device for computation

    Returns
    -------
    tuple[torch.Tensor, XMap_torch]
        Tuple of (density tensor, XMap_torch object). The density tensor contains
        the computed electron density values on the grid.
    """
    xmap = create_synthetic_grid(atom_array, resolution)
    scattering_params = setup_scattering_params(atom_array, em_mode, device)

    xmap_torch = XMap_torch(xmap, device=device)
    transformer = DifferentiableTransformer(
        xmap=xmap_torch,
        scattering_params=scattering_params,
        em=em_mode,
        device=device,
        use_cuda_kernels=torch.cuda.is_available(),
    )

    coords, elements, b_factors, occupancies = extract_density_inputs_from_atomarray(
        atom_array, device
    )

    with torch.no_grad():
        density = transformer(
            coordinates=coords,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    return density.sum(dim=0), xmap_torch


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

    altloc_info = detect_altlocs(atom_array)
    if row.occ_values:
        try:
            atom_array = assign_occupancies(atom_array, altloc_info, "custom", row.occ_values)
        except ValueError as e:
            logger.error(f"Occupancy assignment error for {row.filename}: {e}")
            return

    try:
        density, xmap_torch = compute_density(atom_array, resolution, em_mode, device)
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
    """
    rows = load_batch_csv(csv_path)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {n_jobs} jobs")

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_row)(row, base_dir, output_dir, resolution, em_mode, device)
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

    output_group = parser.add_argument_group("Output Options")
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
        )
    elif args.structure:
        atom_array = load_structure_with_altlocs(args.structure)
        atom_array = apply_selection(atom_array, args.selection)

        altloc_info = detect_altlocs(atom_array)
        occ_values = (
            [float(v.strip()) for v in args.occ_values.split(":")] if args.occ_values else None
        )
        atom_array = assign_occupancies(atom_array, altloc_info, args.occ_mode, occ_values)

        density, xmap_torch = compute_density(atom_array, args.resolution, args.em_mode, device)

        output_path = (
            args.output or args.output_dir / f"{args.structure.stem}_{args.resolution:.2f}A.ccp4"
        )
        save_density(density, xmap_torch, output_path)
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
