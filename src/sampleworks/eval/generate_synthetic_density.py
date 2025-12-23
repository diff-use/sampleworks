import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
    ELECTRON_SCATTERING_FACTORS,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.unitcell import (
    UnitCell,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    GridParameters,
    Resolution,
    XMap,
)
from sampleworks.eval.structure_utils import parse_selection_string
from sampleworks.utils.torch_utils import try_gpu


@dataclass
class AltlocInfo:
    altloc_ids: list[str]
    atom_masks: dict[str, np.ndarray[Any, np.dtype[np.bool_]]]


@dataclass
class BatchRow:
    filename: str
    selection: str | None = None
    occ_values: list[float] = field(default_factory=list)
    mapfile: str | None = None


def load_structure(path: Path) -> AtomArray:
    # Currently, we need to specify extra_fields=["occupancy"] to load altlocs properly
    atom_array = load_any(path, altloc="all", extra_fields=["occupancy", "b_factor"])
    if isinstance(atom_array, AtomArrayStack):
        atom_array = cast(AtomArray, atom_array[0])
    return cast(AtomArray, atom_array)


def apply_selection(atom_array: AtomArray, selection: str | None) -> AtomArray:
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


def detect_altlocs(atom_array: AtomArray) -> AltlocInfo:
    if not hasattr(atom_array, "altloc_id"):
        return AltlocInfo(altloc_ids=[], atom_masks={})

    altloc_arr = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.altloc_id)
    altloc_ids = sorted(set(altloc_arr) - {""})
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
    if mode == "default":
        return atom_array

    if not altloc_info.altloc_ids:
        logger.warning("No altlocs detected, using default occupancies")
        return atom_array

    result = atom_array.copy()
    occupancy = cast(np.ndarray[Any, np.dtype[np.float64]], result.occupancy)

    if mode == "uniform":
        n_altlocs = len(altloc_info.altloc_ids)
        uniform_occ = 1.0 / n_altlocs
        for altloc in altloc_info.altloc_ids:
            occupancy[altloc_info.atom_masks[altloc]] = uniform_occ

    elif mode == "custom":
        if occ_values is None:
            raise ValueError("occ_values required for custom mode")
        if len(occ_values) != len(altloc_info.altloc_ids):
            logger.warning(
                f"Expected {len(altloc_info.altloc_ids)} occupancy values, got {len(occ_values)}. "
                "The missing values are automatically set to 0."
            )
            occ_values = occ_values + [0.0] * (len(altloc_info.altloc_ids) - len(occ_values))
        for altloc, occ in zip(sorted(altloc_info.altloc_ids), occ_values):
            occupancy[altloc_info.atom_masks[altloc]] = occ

    return cast(AtomArray, result)


def normalize_element(elem: str) -> str:
    if len(elem) == 1:
        return elem.upper()
    return elem[0].upper() + elem[1:].lower()


def setup_scattering_params(
    atom_array: AtomArray | AtomArrayStack, em_mode: bool, device: torch.device
) -> torch.Tensor:
    elements = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.element)
    unique_elements = sorted(set(normalize_element(e) for e in elements))
    atomic_num_dict = {elem: ATOMIC_NUM_TO_ELEMENT.index(elem) for elem in unique_elements}

    structure_factors = ELECTRON_SCATTERING_FACTORS if em_mode else ATOM_STRUCTURE_FACTORS

    max_atomic_num = max(atomic_num_dict.values())
    n_coeffs = len(structure_factors["C"][0])
    dense_size = torch.Size([max_atomic_num + 1, n_coeffs, 2])
    scattering_tensor = torch.zeros(dense_size, dtype=torch.float32, device=device)

    for elem in unique_elements:
        atomic_num = atomic_num_dict[elem]
        if elem in structure_factors:
            factor = structure_factors[elem]
        else:
            logger.warning(f"Scattering factors for {elem} not found, using C")
            factor = structure_factors["C"]
        factor_tensor = torch.tensor(factor, dtype=torch.float32, device=device).T
        scattering_tensor[atomic_num, :, :] = factor_tensor

    return scattering_tensor


def create_synthetic_grid(
    atom_array: AtomArray | AtomArrayStack, resolution: float, padding: float = 5.0
) -> XMap:
    coords = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.coord)
    if coords.ndim == 3:
        coords = coords.reshape(-1, 3)

    valid_mask = np.isfinite(coords).all(axis=1)
    coords = coords[valid_mask]

    min_coords = coords.min(axis=0) - padding
    max_coords = coords.max(axis=0) + padding
    extent = max_coords - min_coords

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

    # (nz, ny, nx) ordering for array
    empty_array = np.zeros((grid_shape[2], grid_shape[1], grid_shape[0]), dtype=np.float32)

    xmap = XMap(
        empty_array,
        grid_parameters=GridParameters(voxelspacing=voxel_spacing),
        unit_cell=unit_cell,
        resolution=Resolution(high=resolution, low=1000.0),
        origin=min_coords,
    )

    return xmap


def extract_density_inputs(
    atom_array: AtomArray | AtomArrayStack, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.coord)
    occupancy = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.occupancy)
    b_factor = cast(np.ndarray[Any, np.dtype[np.float64]], atom_array.b_factor)
    elements = cast(np.ndarray[Any, np.dtype[np.str_]], atom_array.element)

    valid_mask = np.isfinite(coords).all(axis=1)
    valid_mask &= occupancy > 0

    coords_tensor = torch.from_numpy(coords[valid_mask]).to(device, dtype=torch.float32)
    elements_tensor = torch.tensor(
        [ATOMIC_NUM_TO_ELEMENT.index(normalize_element(e)) for e in elements[valid_mask]],
        device=device,
        dtype=torch.long,
    )
    b_factors_tensor = torch.from_numpy(b_factor[valid_mask]).to(device, dtype=torch.float32)
    b_factors_tensor = torch.where(
        torch.isnan(b_factors_tensor),
        torch.tensor(20.0, device=device),
        b_factors_tensor,
    )
    occupancies_tensor = torch.from_numpy(occupancy[valid_mask]).to(device, dtype=torch.float32)

    # batch dimension: (1, n_atoms, ...)
    return (
        coords_tensor.unsqueeze(0) if coords_tensor.ndim == 2 else coords_tensor,
        elements_tensor.unsqueeze(0),
        b_factors_tensor.unsqueeze(0),
        occupancies_tensor.unsqueeze(0),
    )


def compute_density(
    atom_array: AtomArray | AtomArrayStack,
    resolution: float,
    em_mode: bool,
    device: torch.device,
) -> tuple[torch.Tensor, XMap_torch]:
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

    coords, elements, b_factors, occupancies = extract_density_inputs(atom_array, device)

    with torch.no_grad():
        density = transformer(
            coordinates=coords,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    return density.sum(dim=0), xmap_torch


def save_density(density: torch.Tensor, xmap_torch: XMap_torch, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xmap_torch.tofile(str(output_path), density)
    logger.info(f"Saved density map to {output_path}")


def load_batch_csv(csv_path: Path) -> list[BatchRow]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            occ_values: list[float] = []
            if row.get("occ_values"):
                occ_values = [float(v.strip()) for v in row["occ_values"].split(":")]

            rows.append(
                BatchRow(
                    filename=row["filename"],
                    selection=row.get("selection") or None,
                    occ_values=occ_values,
                    mapfile=row.get("mapfile") or None,
                )
            )
    return rows


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    em_mode: bool,
    device: torch.device,
) -> None:
    rows = load_batch_csv(csv_path)
    logger.info(f"Processing {len(rows)} structures from {csv_path}")

    for row in rows:
        structure_path = base_dir / row.filename
        if not structure_path.exists():
            logger.error(f"Structure not found: {structure_path}")
            continue

        try:
            atom_array = load_structure(structure_path)
            atom_array = apply_selection(atom_array, row.selection)

            altloc_info = detect_altlocs(atom_array)
            if row.occ_values:
                atom_array = assign_occupancies(atom_array, altloc_info, "custom", row.occ_values)

            density, xmap_torch = compute_density(atom_array, resolution, em_mode, device)

            if row.mapfile:
                output_path = output_dir / row.mapfile
            else:
                output_path = output_dir / f"{structure_path.stem}_{resolution:.2f}A.ccp4"

            save_density(density, xmap_torch, output_path)

        except Exception as e:
            logger.error(f"Failed to process {row.filename}: {e}")


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
        "--selection", type=str, help="Atom selection (e.g., 'chain A and resi 10-50')"
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
        )
    elif args.structure:
        atom_array = load_structure(args.structure)
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


if __name__ == "__main__":
    main()
