"""
# RSCC Analysis for Grid Search Results
# ported to a Python script by Marcus Collins marcus.collins@astera.org, from a notebook file
# provided by karson.chrispens@ucsf.edu

This script calculates the Real Space Correlation Coefficient (RSCC) between computed maps
from refined structures and reference (ground truth) maps for all experiments in the grid search results.

## Workflow:
1. Scan the `grid_search_results` directory for completed experiments
2. For each experiment with a `refined.cif`, compute the electron density map
3. Compare against the corresponding base map and calculate RSCC
4. Aggregate and visualize results by ensemble size, guidance weight, and scaler type
"""

import copy
import re
import traceback
import warnings

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Import local modules for density calculation
from atomworks.io.parser import parse
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import (
    RewardFunction,
    setup_scattering_params,
)
from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import ATOMIC_NUM_TO_ELEMENT


def rscc(array1, array2):
    """
    Calculate the Real Space Correlation Coefficient between two arrays.

    Returns NaN if correlation cannot be computed.
    """
    if array1.shape != array2.shape:
        warnings.warn(f"Shape mismatch: {array1.shape} vs {array2.shape}")
        return np.nan

    if array1.size == 0 or array2.size == 0:
        warnings.warn("Empty array provided to rscc")
        return np.nan

    # Flatten arrays
    arr1_flat = array1.flatten()
    arr2_flat = array2.flatten()

    # Check for NaN/Inf
    if not (np.isfinite(arr1_flat).all() and np.isfinite(arr2_flat).all()):
        warnings.warn("NaN or Inf values in input arrays")
        return np.nan

    # Check for zero variance (constant arrays)
    if np.std(arr1_flat) < 1e-10 or np.std(arr2_flat) < 1e-10:
        warnings.warn("Zero or near-zero variance in input arrays")
        return np.nan

    try:
        corr = np.corrcoef(arr1_flat, arr2_flat)[0, 1]
        return corr
    except Exception as e:
        warnings.warn(f"Correlation calculation failed: {e}")
        return np.nan


def extract_protein_and_occupancy(dir_name):
    """Extract protein name and occupancy from directory name.

    Examples:
    - '1vme_0.5occA_0.5occB' -> ('1vme', 0.5)
    - '6b8x_1.0occA' -> ('6b8x', 1.0)
    - '5sop_1.0occB' -> ('5sop', 0.0)
    """
    # Extract protein name (first part before underscore with occupancy)
    parts = dir_name.lower().split("_")
    protein = parts[0]

    # Parse occupancy
    if "1.0occa" in dir_name.lower() or "1occa" in dir_name.lower():
        # Check it's not a mixed case like 0.1occA
        if not any(f"0.{i}occa" in dir_name.lower() for i in range(1, 10)):
            occ_a = 1.0
        else:
            match = re.search(r"(\d+\.?\d*)occA", dir_name, re.IGNORECASE)
            occ_a = float(match.group(1)) if match else None
    elif "1.0occb" in dir_name.lower() or "1occb" in dir_name.lower():
        if not any(f"0.{i}occb" in dir_name.lower() for i in range(1, 10)):
            occ_a = 0.0
        else:
            match = re.search(r"(\d+\.?\d*)occA", dir_name, re.IGNORECASE)
            occ_a = float(match.group(1)) if match else None
    else:
        match = re.search(r"(\d+\.?\d*)occA", dir_name, re.IGNORECASE)
        occ_a = float(match.group(1)) if match else None

    return protein, occ_a


def occupancy_to_str(occ_a):
    """Convert occupancy float to string format used in filenames.

    Examples:
    - 1.0 -> '1.0occA'
    - 0.0 -> '1.0occB'
    - 0.5 -> '0.5occA_0.5occB'
    - 0.25 -> '0.25occA_0.75occB'
    """
    if abs(occ_a - 1.0) < 1e-6:
        return "1.0occA"
    elif abs(occ_a) < 1e-6:
        return "1.0occB"
    else:
        occ_b = round(1.0 - occ_a, 2)
        return f"{occ_a}occA_{occ_b}occB"


def occupancy_to_str_6b8x(occ_a):
    """Convert occupancy float to 6b8x-style string format.

    Examples:
    - 1.0 -> '1.0occAconf'
    - 0.0 -> '1.0occBconf'
    - 0.5 -> '0.5occAconf_0.5occBconf'
    """
    if abs(occ_a - 1.0) < 1e-6:
        return "1.0occAconf"
    elif abs(occ_a) < 1e-6:
        return "1.0occBconf"
    else:
        occ_b = round(1.0 - occ_a, 2)
        return f"{occ_a}occAconf_{occ_b}occBconf"


def parse_experiment_dir(exp_dir):
    """Parse experiment directory name to extract parameters.

    Handles both:
    - fk_steering format: ens{N}_gw{W}_gd{D}
    - pure_guidance format: ens{N}_gw{W}
    """
    dir_name = exp_dir.name

    # Extract ensemble size
    ens_match = re.search(r"ens(\d+)", dir_name)
    ensemble_size = int(ens_match.group(1)) if ens_match else None

    # Extract guidance weight
    gw_match = re.search(r"gw([\d.]+)", dir_name)
    guidance_weight = float(gw_match.group(1)) if gw_match else None

    # Extract gradient descent steps (for fk_steering)
    gd_match = re.search(r"gd(\d+)", dir_name)
    gd_steps = int(gd_match.group(1)) if gd_match else None

    return {
        "ensemble_size": ensemble_size,
        "guidance_weight": guidance_weight,
        "gd_steps": gd_steps,
    }


def scan_grid_search_results(grid_search_dir):
    """Scan the grid_search_results directory for all experiments with refined.cif files."""
    experiments = []

    if not grid_search_dir.exists():
        print(f"Grid search directory not found: {grid_search_dir}")
        return experiments

    # Iterate through protein directories
    for protein_dir in grid_search_dir.iterdir():
        if not protein_dir.is_dir() or protein_dir.name.endswith(".json"):
            continue

        protein, occ_a = extract_protein_and_occupancy(protein_dir.name)

        # Iterate through model directories (boltz2_MD, boltz2_X-RAY_DIFFRACTION, protenix)
        for model_dir in protein_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            # Determine method from model directory name
            if "MD" in model_name:
                method = "MD"
                model = model_name.replace("_MD", "")
            elif "X-RAY" in model_name:
                method = "X-RAY"
                model = model_name.replace("_X-RAY_DIFFRACTION", "")
            else:
                method = None
                model = model_name

            # Iterate through scaler directories (pure_guidance, fk_steering)
            for scaler_dir in model_dir.iterdir():
                if not scaler_dir.is_dir():
                    continue

                scaler = scaler_dir.name

                # Iterate through experiment parameter directories
                for exp_dir in scaler_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue

                    refined_cif = exp_dir / "refined.cif"
                    if not refined_cif.exists():
                        continue

                    params = parse_experiment_dir(exp_dir)

                    experiments.append(
                        {
                            "protein": protein,
                            "occ_a": occ_a,
                            "model": model,
                            "method": method,
                            "scaler": scaler,
                            "ensemble_size": params["ensemble_size"],
                            "guidance_weight": params["guidance_weight"],
                            "gd_steps": params["gd_steps"],
                            "exp_dir": exp_dir,
                            "refined_cif_path": refined_cif,
                            "protein_dir_name": protein_dir.name,
                        }
                    )

    return experiments


def get_base_map_path(protein, occ_a, protein_configs):
    """Get the path to the base/reference map for a given protein and occupancy."""
    if protein not in protein_configs:
        print(f"Warning: Unknown protein {protein}")
        return None

    config = protein_configs[protein]
    base_map_dir = config["base_map_dir"]

    # Handle 6b8x special naming
    if protein == "6b8x":
        occ_str = occupancy_to_str_6b8x(occ_a)
    else:
        occ_str = occupancy_to_str(occ_a)

    map_pattern = config["map_pattern"]
    map_filename = map_pattern.format(occ_str=occ_str)
    map_path = base_map_dir / map_filename

    if not map_path.exists():
        # Try alternate naming patterns
        alt_patterns = []
        if protein == "6b8x":
            # Try without "conf" suffix for some files
            alt_occ_str = occupancy_to_str(occ_a)
            alt_patterns.append(f"6b8x_{alt_occ_str}_1.74A.ccp4")

        for alt in alt_patterns:
            alt_path = base_map_dir / alt
            if alt_path.exists():
                return alt_path

        print(f"Warning: Base map not found: {map_path}")
        return None

    return map_path


def get_reference_structure_path(protein, occ_a, protein_configs):
    """Get the path to the reference structure CIF file for a given protein and occupancy.

    Parameters
    ----------
    protein : str
        Protein name (e.g., '1vme', '6b8x')
    occ_a : float
        Occupancy of conformer A (0.0 to 1.0)
    protein_configs : dict
        Configuration dictionary

    Returns
    -------
    Path or None
        Path to the reference structure CIF file, or None if not found
    """
    if protein not in protein_configs:
        return None

    config = protein_configs[protein]
    base_dir = config["base_map_dir"]
    occ_str = occupancy_to_str(occ_a)
    structure_pattern = config.get("structure_pattern", "")

    if not structure_pattern:
        return None

    # Handle 6b8x special naming with "conf" suffix
    if protein == "6b8x":
        occ_str = occ_str.replace("occA", "occAconf").replace("occB", "occBconf")

    structure_path = base_dir / structure_pattern.format(occ_str=occ_str)

    if structure_path.exists():
        return structure_path

    # Try shifted version for 6b8x
    if protein == "6b8x":
        shifted_path = base_dir / structure_pattern.format(occ_str=occ_str).replace(
            ".cif", "_shifted.cif"
        )
        if shifted_path.exists():
            return shifted_path

    print(f"Warning: Reference structure not found: {structure_path}")
    return None


def parse_selection_string(selection):
    """Parse a selection string like 'chain A and resi 326-339'.

    Parameters
    ----------
    selection : str
        Selection string

    Returns
    -------
    tuple
        (chain_id, resi_start, resi_end)
    """
    # Parse "chain X and resi N-M" format
    parts = selection.lower().replace("and", "").split()
    chain_id = None
    resi_start = None
    resi_end = None

    for i, part in enumerate(parts):
        if part == "chain" and i + 1 < len(parts):
            chain_id = parts[i + 1].upper()
        elif part == "resi" and i + 1 < len(parts):
            resi_range = parts[i + 1]
            if "-" in resi_range:
                resi_start, resi_end = map(int, resi_range.split("-"))
            else:
                resi_start = resi_end = int(resi_range)

    return chain_id, resi_start, resi_end


def extract_selection_coordinates(structure, selection):
    """Extract coordinates for atoms matching a selection from an atomworks structure.

    Parameters
    ----------
    structure : dict
        Atomworks parsed structure dictionary
    selection : str
        Selection string like 'chain A and resi 326-339'

    Returns
    -------
    np.ndarray
        Coordinates of selected atoms, shape (n_atoms, 3)

    Raises
    ------
    ValueError
        If no atoms match the selection or coordinates are invalid
    """
    atom_array = structure["asym_unit"]
    if hasattr(atom_array, "__len__") and not isinstance(atom_array, np.ndarray):
        # AtomArrayStack - take first frame
        if len(atom_array) > 0:
            atom_array = atom_array[0]

    chain_id, resi_start, resi_end = parse_selection_string(selection)

    # Create selection mask
    mask = np.ones(len(atom_array), dtype=bool)

    if chain_id is not None:
        mask &= atom_array.chain_id == chain_id

    if resi_start is not None and resi_end is not None:
        mask &= (atom_array.res_id >= resi_start) & (atom_array.res_id <= resi_end)

    selected_coords = atom_array.coord[mask]

    # VALIDATION
    if len(selected_coords) == 0:
        raise ValueError(
            f"No atoms matched selection: '{selection}'. "
            f"Chain ID: {chain_id}, Residue range: {resi_start}-{resi_end}. "
            f"Total atoms in structure: {len(atom_array)}"
        )

    # Filter out atoms with NaN or Inf coordinates (common in alt conf structures)
    finite_mask = np.isfinite(selected_coords).all(axis=1)
    if not finite_mask.all():
        n_invalid = (~finite_mask).sum()
        n_total = len(selected_coords)
        warnings.warn(
            f"Filtered {n_invalid} atoms with NaN/Inf coordinates from "
            f"selection '{selection}' ({n_total - n_invalid} valid atoms remaining)"
        )
        selected_coords = selected_coords[finite_mask]

    # Check if we have any valid coordinates left
    if len(selected_coords) == 0:
        raise ValueError(
            f"No valid (finite) coordinates after filtering NaN/Inf from "
            f"selection: '{selection}'"
        )

    return selected_coords


def compute_density_from_structure(structure, xmap, device=None):
    """Compute electron density from a structure dictionary.

    Parameters
    ----------
    structure : dict
        Atomworks parsed structure dictionary
    xmap : XMap
        Reference XMap for grid parameters
    device : torch.device, optional
        Device to use for computation

    Returns
    -------
    np.ndarray
        Computed electron density array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get atom array from structure
    atom_array = structure["asym_unit"]
    if hasattr(atom_array, "__len__") and not isinstance(atom_array, np.ndarray):
        # AtomArrayStack - take first frame
        if len(atom_array) > 0:
            atom_array = atom_array[0]

    # Filter atoms with occupancy > 0
    mask = atom_array.occupancy > 0
    atom_array = atom_array[mask]

    # Set up scattering parameters
    scattering_params = setup_scattering_params(structure, em=False)

    # Create differentiable transformer
    xmap_torch = XMap_torch(xmap, device=device)
    transformer = DifferentiableTransformer(
        xmap=xmap_torch,
        scattering_params=scattering_params.to(device),
        em=False,
        device=device,
        use_cuda_kernels=torch.cuda.is_available(),
    )

    # Prepare input tensors
    elements = [
        ATOMIC_NUM_TO_ELEMENT.index(
            elem.upper() if len(elem) == 1 else elem[0].upper() + elem[1:].lower()
        )
        for elem in atom_array.element
    ]
    elements = torch.tensor(elements, device=device).unsqueeze(0)
    coordinates = torch.from_numpy(atom_array.coord).float().to(device).unsqueeze(0)
    b_factors = (
        torch.from_numpy(atom_array.b_factor).float().to(device).unsqueeze(0)
    )
    occupancies = (
        torch.from_numpy(atom_array.occupancy).float().to(device).unsqueeze(0)
    )

    # Compute density
    with torch.no_grad():
        density = transformer(
            coordinates=coordinates,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
        )

    return density.cpu().numpy().squeeze()


def main():
    # Configuration: paths and protein configs
    # Try to determine workspace root relative to this script
    try:
        WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
    except NameError:
        # Fallback if __file__ is not defined (e.g. interactive mode) or path structure is different
        WORKSPACE_ROOT = Path("/home/kchrispens/sampleworks")

    # If the relative path doesn't seem right (e.g. no grid_search_results), use the hardcoded one as fallback/override
    if not (WORKSPACE_ROOT / "grid_search_results").exists():
        WORKSPACE_ROOT = Path("/home/kchrispens/sampleworks")

    GRID_SEARCH_DIR = WORKSPACE_ROOT / "grid_search_results"

    # Protein configurations: base map paths, structure selections, and resolutions
    PROTEIN_CONFIGS = {
        "1vme": {
            "base_map_dir": WORKSPACE_ROOT / "1vme_final_carved_edited",
            "selection": "chain A and resi 326-339",
            "resolution": 1.8,
            "map_pattern": "1vme_final_carved_edited_{occ_str}_1.80A.ccp4",
            "structure_pattern": "1vme_final_carved_edited_{occ_str}.cif",
        },
        "4ole": {
            "base_map_dir": WORKSPACE_ROOT / "4ole_final_carved",
            "selection": "chain B and resi 60-67",
            "resolution": 2.52,
            "map_pattern": "4ole_final_carved_{occ_str}_2.52A.ccp4",
            "structure_pattern": "4ole_final_carved_{occ_str}.cif",
        },
        "5sop": {
            "base_map_dir": WORKSPACE_ROOT / "5sop",
            "selection": "chain A and resi 129-135",
            "resolution": 1.05,
            "map_pattern": "5sop_{occ_str}_1.05A.ccp4",
            "structure_pattern": "5sop_{occ_str}.cif",
        },
        "6b8x": {
            "base_map_dir": WORKSPACE_ROOT / "6b8x",
            "selection": "chain A and resi 180-184",
            "resolution": 1.74,
            "map_pattern": "6b8x_{occ_str}_1.74A.ccp4",
            "structure_pattern": "6b8x_synthetic_{occ_str}.cif",
        },
    }

    print(f"Grid search directory: {GRID_SEARCH_DIR}")
    print(f"Proteins configured: {list(PROTEIN_CONFIGS.keys())}")

    # Test base map path resolution
    print("Testing base map path resolution:")
    for _protein in PROTEIN_CONFIGS.keys():
        for _occ in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _path = get_base_map_path(_protein, _occ, PROTEIN_CONFIGS)
            if _path:
                print(f"  {_protein} occ={_occ}: {_path.name}")
            else:
                print(f"  {_protein} occ={_occ}: NOT FOUND")

    # Scan for experiments
    all_experiments = scan_grid_search_results(GRID_SEARCH_DIR)
    print(f"Found {len(all_experiments)} experiments with refined.cif files")

    # Show summary
    if all_experiments:
        proteins_found = set(e["protein"] for e in all_experiments)
        models_found = set(e["model"] for e in all_experiments)
        scalers_found = set(e["scaler"] for e in all_experiments)
        print(f"Proteins: {proteins_found}")
        print(f"Models: {models_found}")
        print(f"Scalers: {scalers_found}")

    # Test reference structure resolution
    print("\nTesting reference structure path resolution:")
    for _protein in PROTEIN_CONFIGS.keys():
        _path = get_reference_structure_path(_protein, 0.5, PROTEIN_CONFIGS)
        if _path:
            print(f"  {_protein} occ=0.5: {_path.name}")
            # Also test coordinate extraction
            try:
                _struct = parse(str(_path), ccd_mirror_path=None)
                _selection = PROTEIN_CONFIGS[_protein]["selection"]
                _coords = extract_selection_coordinates(_struct, _selection)
                print(f"    Selection '{_selection}': {len(_coords)} atoms")
            except Exception as _e:
                print(f"    Error extracting coordinates: {_e}")
        else:
            print(f"  {_protein} occ=0.5: NOT FOUND")

    # Calculate RSCC for all experiments
    print("Calculating RSCC values for all experiments...")
    print(
        "Note: RSCC is computed on the region around altloc residues (defined by selection)"
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    # Pre-load reference structures for each protein (at 0.5 occupancy for coordinate extraction)
    _ref_structures = {}
    _ref_coords = {}
    for _protein_key, _config in PROTEIN_CONFIGS.items():
        _ref_path = get_reference_structure_path(_protein_key, 0.5, PROTEIN_CONFIGS)
        if _ref_path and _ref_path.exists():
            try:
                _ref_struct = parse(str(_ref_path), ccd_mirror_path=None)
                _ref_structures[_protein_key] = _ref_struct
                _selection = _config["selection"]
                _coords = extract_selection_coordinates(_ref_struct, _selection)
                _ref_coords[_protein_key] = _coords
                print(
                    f"  Loaded reference structure for {_protein_key}: {len(_coords)} atoms in selection '{_selection}'"
                )
            except Exception as _e:
                print(f"  ERROR: Failed to load reference for {_protein_key}: {_e}")
                print(f"    Path: {_ref_path}")
                print(f"    Selection: {_config.get('selection', 'N/A')}")
                print(f"    Traceback: {traceback.format_exc()}")
        else:
            print(
                f"  WARNING: Reference structure not found for {_protein_key} "
                f"(occ=0.5): {_ref_path}"
            )

    results = []

    for _i, _exp in enumerate(all_experiments):
        _protein = _exp["protein"]
        _occ_a = _exp["occ_a"]

        if _protein not in PROTEIN_CONFIGS:
            print(f"Skipping unknown protein: {_protein}")
            continue

        _config = PROTEIN_CONFIGS[_protein]
        _resolution = _config["resolution"]

        # Check if we have reference coordinates for region extraction
        if _protein not in _ref_coords:
            print(
                f"Skipping {_exp['protein_dir_name']}: no reference structure available"
            )
            continue

        _selection_coords = _ref_coords[_protein]

        # Get base map path
        _base_map_path = get_base_map_path(_protein, _occ_a, PROTEIN_CONFIGS)
        if _base_map_path is None:
            print(f"Skipping {_exp['protein_dir_name']}: base map not found")
            continue

        try:
            # VALIDATE coordinates before use
            if len(_selection_coords) == 0:
                raise ValueError("Empty selection coordinates")

            if not np.isfinite(_selection_coords).all():
                raise ValueError("Invalid coordinates contain NaN/Inf")

            # Load base map
            _base_xmap = XMap.fromfile(str(_base_map_path), resolution=_resolution)
            _base_xmap = _base_xmap.canonical_unit_cell()

            # Extract region around altloc residues from base map
            _extracted_base = _base_xmap.extract(_selection_coords, padding=2.0)

            # Validate extraction
            if _extracted_base.array.size == 0:
                raise ValueError("Extracted base map is empty")

            # Load refined structure
            _structure = parse(str(_exp["refined_cif_path"]), ccd_mirror_path=None)

            # Compute density from refined structure
            _computed_density = compute_density_from_structure(
                _structure, _base_xmap, _device
            )

            # Create an XMap from the computed density by copying the base xmap
            # and replacing its array with the computed density
            _computed_xmap = copy.deepcopy(_base_xmap)
            _computed_xmap.array = _computed_density
            _extracted_computed = _computed_xmap.extract(_selection_coords, padding=2.0)

            # Validate extraction
            if _extracted_computed.array.size == 0:
                raise ValueError("Extracted computed map is empty")

            # Calculate RSCC on extracted regions
            _rscc_value = rscc(_extracted_base.array, _extracted_computed.array)

            results.append(
                {
                    **_exp,
                    "rscc": _rscc_value,
                    "base_map_path": str(_base_map_path),
                }
            )

            if (_i + 1) % 10 == 0 or _i == 0:
                print(
                    f"  [{_i + 1}/{len(all_experiments)}] {_exp['protein_dir_name']} / "
                    f"{_exp['model']} / {_exp['scaler']} / ens{_exp['ensemble_size']}_"
                    f"gw{_exp['guidance_weight']}: RSCC = {_rscc_value:.4f}"
                )

        except Exception as _e:
            print(f"ERROR processing {_exp['exp_dir']}: {_e}")
            print(f"  Traceback: {traceback.format_exc()}")
            results.append(
                {
                    **_exp,
                    "rscc": np.nan,
                    "base_map_path": str(_base_map_path) if _base_map_path else None,
                    "error": str(_e),
                }
            )

    print(f"\nCompleted RSCC calculation for {len(results)} experiments")

    # Create DataFrame from results
    df = pd.DataFrame(results)

    if not df.empty:
        # Remove error column for display if present
        display_cols = [
            c
            for c in df.columns
            if c
               not in [
                   "exp_dir",
                   "refined_cif_path",
                   "base_map_path",
                   "error",
                   "protein_dir_name",
               ]
        ]

        print("Results Summary:")
        print(df[display_cols].head(20).to_string())

        print("\n\nSummary Statistics by Protein and Scaler:")
        summary = (
            df.groupby(["protein", "scaler"])["rscc"]
            .agg(["count", "mean", "std", "min", "max"])
            .round(4)
        )
        print(summary)

    # Calculate correlation between base maps and pure conformer maps
    print("Calculating correlations between base maps and pure conformer maps...")
    print("This shows how well single conformers explain occupancy-mixed data")

    _ref_structures_for_corr = {}
    _ref_coords_for_corr = {}
    for _protein_key, _config in PROTEIN_CONFIGS.items():
        try:
            # Use the same structure pattern from config
            occ_str = "0.5occA_0.5occB"
            if _protein_key == "6b8x":
                occ_str = "0.5occAconf_0.5occBconf"

            structure_pattern = _config.get("structure_pattern", "")
            if structure_pattern:
                structure_path = _config["base_map_dir"] / structure_pattern.format(
                    occ_str=occ_str
                )
                if not structure_path.exists() and _protein_key == "6b8x":
                    # Try shifted version
                    structure_path = _config["base_map_dir"] / (
                        structure_pattern.format(occ_str=occ_str).replace(
                            ".cif", "_shifted.cif"
                        )
                    )

                if structure_path.exists():
                    _ref_struct = parse(str(structure_path), ccd_mirror_path=None)
                    _ref_structures_for_corr[_protein_key] = _ref_struct

                    # Extract coordinates for selection using the shared function
                    # which properly filters out NaN/Inf coordinates
                    _selection = _config["selection"]
                    _coords = extract_selection_coordinates(_ref_struct, _selection)
                    _ref_coords_for_corr[_protein_key] = _coords
                    print(
                        f"  Loaded reference structure for {_protein_key}: "
                        f"{len(_ref_coords_for_corr[_protein_key])} atoms"
                    )
        except Exception as _e:
            print(
                f"  Warning: Failed to load reference structure for "
                f"{_protein_key}: {_e}"
            )

    base_pure_correlations = []

    for _protein_key, _config in PROTEIN_CONFIGS.items():
        if _protein_key not in _ref_coords_for_corr:
            print(f"Skipping {_protein_key}: no reference coordinates available")
            continue

        _resolution = _config["resolution"]
        _selection_coords = _ref_coords_for_corr[_protein_key]

        # Get pure conformer maps (1.0occA and 1.0occB)
        _base_map_1occA = get_base_map_path(_protein_key, 1.0, PROTEIN_CONFIGS)
        _base_map_1occB = get_base_map_path(_protein_key, 0.0, PROTEIN_CONFIGS)

        if _base_map_1occA is None or _base_map_1occB is None:
            print(f"Skipping {_protein_key}: pure conformer maps not found")
            continue

        if not _base_map_1occA.exists() or not _base_map_1occB.exists():
            print(f"Skipping {_protein_key}: pure conformer map files don't exist")
            continue

        print(f"\nProcessing {_protein_key} single conformer explanatory power:")
        print(f"  Pure A reference: {_base_map_1occA.name}")
        print(f"  Pure B reference: {_base_map_1occB.name}")

        try:
            # Load pure conformer maps
            _pure_xmap_A = XMap.fromfile(str(_base_map_1occA), resolution=_resolution)
            _pure_xmap_A = _pure_xmap_A.canonical_unit_cell()
            _pure_xmap_B = XMap.fromfile(str(_base_map_1occB), resolution=_resolution)
            _pure_xmap_B = _pure_xmap_B.canonical_unit_cell()

            # Extract regions using reference coordinates
            _extracted_pure_A = _pure_xmap_A.extract(_selection_coords, padding=0.0)
            _extracted_pure_B = _pure_xmap_B.extract(_selection_coords, padding=0.0)

            # Calculate correlations for each occupancy
            _occupancies = [0.0, 0.25, 0.5, 0.75, 1.0]
            for _occ_a in _occupancies:
                try:
                    _base_map_path = get_base_map_path(_protein_key, _occ_a, PROTEIN_CONFIGS)
                    if _base_map_path is None or not _base_map_path.exists():
                        print(f"  Warning: base map not found for occ_A={_occ_a}")
                        continue

                    print(f"  Processing occ_A={_occ_a}: {_base_map_path.name}")

                    # Load base map for this occupancy
                    _base_xmap = XMap.fromfile(
                        str(_base_map_path), resolution=_resolution
                    )
                    _base_xmap = _base_xmap.canonical_unit_cell()
                    _extracted_base = _base_xmap.extract(_selection_coords, padding=0.0)

                    # Calculate correlations
                    _corr_base_vs_pureA = rscc(
                        _extracted_base.array, _extracted_pure_A.array
                    )
                    _corr_base_vs_pureB = rscc(
                        _extracted_base.array, _extracted_pure_B.array
                    )

                    base_pure_correlations.append(
                        {
                            "protein": _protein_key,
                            "occ_a": _occ_a,
                            "base_vs_1occA": _corr_base_vs_pureA,
                            "base_vs_1occB": _corr_base_vs_pureB,
                        }
                    )

                    print(f"    Base map vs pure A: {_corr_base_vs_pureA:.4f}")
                    print(f"    Base map vs pure B: {_corr_base_vs_pureB:.4f}")

                except Exception as _e:
                    print(f"  Error processing occ_A={_occ_a}: {_e}")

        except Exception as _e:
            print(f"Error calculating correlations for {_protein_key}: {_e}")

    df_base_vs_pure = pd.DataFrame(base_pure_correlations)
    print(
        f"\nCalculated single conformer explanatory power for "
        f"{len(df_base_vs_pure)} occupancy points"
    )

    if not df_base_vs_pure.empty:
        print("\nSummary by protein:")
        for _protein in df_base_vs_pure["protein"].unique():
            _protein_data = df_base_vs_pure[
                df_base_vs_pure["protein"] == _protein
                ].sort_values("occ_a")
            print(f"\n{_protein}:")
            for _, _row in _protein_data.iterrows():
                print(
                    f"  occ_A={_row['occ_a']:.2f}: "
                    f"vs_pureA={_row['base_vs_1occA']:.4f}, "
                    f"vs_pureB={_row['base_vs_1occB']:.4f}"
                )

    # Visualization: RSCC by ensemble size and guidance weight
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc", "ensemble_size", "guidance_weight"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            # Set up the plotting style
            sns.set_theme(context="poster", style="whitegrid")

            # Plot 1: RSCC vs ensemble size, faceted by scaler
            _fig1, _axes1 = plt.subplots(1, 2, figsize=(14, 5))

            for _idx, _scaler in enumerate(["pure_guidance", "fk_steering"]):
                _ax = _axes1[_idx]
                _scaler_df = _plot_df[_plot_df["scaler"] == _scaler]

                if not _scaler_df.empty:
                    for _gw in sorted(_scaler_df["guidance_weight"].unique()):
                        _gw_df = _scaler_df[_scaler_df["guidance_weight"] == _gw]
                        _agg = (
                            _gw_df.groupby("ensemble_size")["rscc"]
                            .agg(["mean", "std"])
                            .reset_index()
                        )
                        _ax.errorbar(
                            _agg["ensemble_size"],
                            _agg["mean"],
                            yerr=_agg["std"],
                            marker="o",
                            label=f"gw={_gw}",
                            capsize=3,
                        )

                _ax.set_xlabel("Ensemble Size", fontsize=12)
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title(f"{_scaler.replace('_', ' ').title()}", fontsize=14)
                _ax.legend()
                _ax.set_xticks([1, 2, 4, 8])

            plt.tight_layout()
            plt.show()

            # Plot 2: Heatmap of RSCC by ensemble size and guidance weight for each scaler
            _fig2, _axes2 = plt.subplots(1, 2, figsize=(14, 5))

            for _idx, _scaler in enumerate(["pure_guidance", "fk_steering"]):
                _ax = _axes2[_idx]
                _scaler_df = _plot_df[_plot_df["scaler"] == _scaler]

                if not _scaler_df.empty:
                    _pivot = _scaler_df.pivot_table(
                        values="rscc",
                        index="ensemble_size",
                        columns="guidance_weight",
                        aggfunc="mean",
                    )

                    sns.heatmap(
                        _pivot,
                        annot=True,
                        fmt=".3f",
                        cmap="RdYlGn",
                        vmin=0,
                        vmax=1,
                        ax=_ax,
                    )
                    _ax.set_title(f"{_scaler.replace('_', ' ').title()}", fontsize=14)
                    _ax.set_xlabel("Guidance Weight", fontsize=12)
                    _ax.set_ylabel("Ensemble Size", fontsize=12)

            plt.tight_layout()
            plt.show()

    # Visualization: RSCC by protein and occupancy
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc", "occ_a"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            # Get unique proteins
            _proteins = sorted(_plot_df["protein"].unique())
            _n_proteins = len(_proteins)

            _fig, _axes = plt.subplots(
                1, _n_proteins, figsize=(5 * _n_proteins, 5), squeeze=False
            )
            _axes = _axes.flatten()

            for _idx, _protein in enumerate(_proteins):
                _ax = _axes[_idx]
                _protein_df = _plot_df[_plot_df["protein"] == _protein]

                for _scaler in _protein_df["scaler"].unique():
                    _scaler_df = _protein_df[_protein_df["scaler"] == _scaler]
                    _agg = (
                        _scaler_df.groupby("occ_a")["rscc"]
                        .agg(["mean", "std"])
                        .reset_index()
                    )

                    _ax.errorbar(
                        _agg["occ_a"],
                        _agg["mean"],
                        yerr=_agg["std"],
                        marker="o",
                        label=_scaler.replace("_", " ").title(),
                        capsize=3,
                    )

                _ax.set_xlabel("Conformer A Occupancy", fontsize=12)
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title(f"{_protein.upper()}", fontsize=14)
                _ax.set_xlim(-0.05, 1.05)
                _ax.set_ylim(0, 1.05)
                _ax.legend()
                _ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

            plt.tight_layout()
            plt.show()

    # Visualization: Compare models (boltz2 vs protenix)
    if df.empty or df["rscc"].isna().all():
        print("No valid RSCC values to plot")
    else:
        _plot_df = df.dropna(subset=["rscc"])

        if _plot_df.empty:
            print("No valid data for plotting")
        else:
            _models = sorted(_plot_df["model"].unique())

            if len(_models) > 1:
                _fig, _ax = plt.subplots(figsize=(10, 6))

                _agg = (
                    _plot_df.groupby(["model", "scaler"])["rscc"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )

                _x_pos = np.arange(len(_agg))
                _labels = [
                    f"{_row['model']}\n{_row['scaler']}" for _, _row in _agg.iterrows()
                ]

                _colors = sns.color_palette("husl", len(_agg))
                _bars = _ax.bar(
                    _x_pos, _agg["mean"], yerr=_agg["std"], capsize=5, color=_colors
                )

                _ax.set_xticks(_x_pos)
                _ax.set_xticklabels(_labels, rotation=45, ha="right")
                _ax.set_ylabel("RSCC", fontsize=12)
                _ax.set_title("RSCC by Model and Scaler", fontsize=14)
                _ax.set_ylim(0, 1.05)

                # Add count labels
                for _bar, _count in zip(_bars, _agg["count"]):
                    _ax.text(
                        _bar.get_x() + _bar.get_width() / 2,
                        _bar.get_height() + 0.02,
                        f"n={_count}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.tight_layout()
                plt.show()
            else:
                print(f"Only one model found: {_models}")

    # Visualization: Guidance RSCC vs Single Conformer
    print("\nPlotting guidance effectiveness vs single conformer explanatory power...")

    if df.empty or df["rscc"].isna().all():
        print("No valid guidance RSCC data to plot")
    elif df_base_vs_pure.empty:
        print("No pure conformer correlation data to plot")
    else:
        # Aggregate guidance RSCC by protein and occupancy
        _plot_df = df.dropna(subset=["rscc", "occ_a"])
        _agg_guidance = (
            _plot_df.groupby(["protein", "occ_a"], as_index=False)
            .agg(
                rscc_mean=("rscc", "mean"),
                rscc_std=("rscc", "std"),
                n=("rscc", "size"),
            )
            .sort_values(["protein", "occ_a"])
        )

        # Get unique proteins that have both guidance and pure correlation data
        _proteins_guidance = set(_agg_guidance["protein"].unique())
        _proteins_pure = set(df_base_vs_pure["protein"].unique())
        _proteins = sorted(_proteins_guidance & _proteins_pure)

        if not _proteins:
            print("No proteins with both guidance and pure correlation data")
        else:
            # Set plotting style
            sns.set_theme(context="paper", style="whitegrid")

            # Define colors
            _colors = {
                "guidance": "#1f77b4",
                "pure_A": "#ff7f0e",
                "pure_B": "#2ca02c",
            }

            # Create one plot per protein
            for _protein in _proteins:
                _fig, _ax = plt.subplots(figsize=(10, 6))

                # Plot guidance RSCC
                _protein_guidance = _agg_guidance[
                    _agg_guidance["protein"] == _protein
                    ].sort_values("occ_a")

                if len(_protein_guidance) > 0:
                    _ax.plot(
                        _protein_guidance["occ_a"],
                        _protein_guidance["rscc_mean"],
                        color=_colors["guidance"],
                        marker="o",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Guided Ensemble Map",
                    )

                    # Add error bars if available
                    _has_error = (_protein_guidance["n"] > 1) & ~_protein_guidance[
                        "rscc_std"
                    ].isna()
                    if _has_error.any():
                        _error_sub = _protein_guidance[_has_error]
                        _ax.errorbar(
                            _error_sub["occ_a"],
                            _error_sub["rscc_mean"],
                            yerr=_error_sub["rscc_std"],
                            fmt="none",
                            color=_colors["guidance"],
                            alpha=0.5,
                            capsize=3,
                        )

                # Plot pure conformer correlations
                _protein_pure = df_base_vs_pure[
                    df_base_vs_pure["protein"] == _protein
                    ].sort_values("occ_a")

                if len(_protein_pure) > 0:
                    _ax.plot(
                        _protein_pure["occ_a"],
                        _protein_pure["base_vs_1occA"],
                        color=_colors["pure_A"],
                        marker="s",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Conformer A Map",
                    )

                    _ax.plot(
                        _protein_pure["occ_a"],
                        _protein_pure["base_vs_1occB"],
                        color=_colors["pure_B"],
                        marker="^",
                        linestyle="-",
                        markersize=8,
                        linewidth=2,
                        label="Conformer B Map",
                    )

                # Formatting
                _ax.set_xlabel("Conformer A Occupancy", fontsize=12, fontweight="bold")
                _ax.set_ylabel("RSCC", fontsize=12, fontweight="bold")
                _ax.set_title(
                    f"{_protein.upper()} - Guidance vs Single Conformer RSCC",
                    fontsize=14,
                    fontweight="bold",
                )
                _ax.set_xlim(-0.05, 1.05)
                _ax.set_ylim(0.0, 1.05)
                _ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                _ax.set_xticklabels(
                    [
                        "0.0\n(pure B)",
                        "0.25",
                        "0.5\n(equal mix)",
                        "0.75",
                        "1.0\n(pure A)",
                    ]
                )
                _ax.legend(
                    fontsize=10,
                    loc="best",
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                )
                _ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            print(
                f"Plotted guidance vs pure conformer comparisons for "
                f"{len(_proteins)} proteins"
            )

    # Export results to CSV with optional base vs pure correlations
    _output_path = WORKSPACE_ROOT / "grid_search_results" / "rscc_results.csv"

    # Select columns to export
    _export_cols = [
        "protein",
        "occ_a",
        "model",
        "method",
        "scaler",
        "ensemble_size",
        "guidance_weight",
        "gd_steps",
        "rscc",
    ]
    _export_df = df[[c for c in _export_cols if c in df.columns]]

    # Merge with base vs pure correlations if available
    if not df_base_vs_pure.empty:
        # Merge on protein and occ_a to add correlation columns
        _export_df = pd.merge(
            _export_df,
            df_base_vs_pure[["protein", "occ_a", "base_vs_1occA", "base_vs_1occB"]],
            on=["protein", "occ_a"],
            how="left",
        )
        print("Added base vs pure conformer correlation columns to export")

    # Ensure output directory exists
    _output_path.parent.mkdir(parents=True, exist_ok=True)

    _export_df.to_csv(_output_path, index=False)
    print(f"Results exported to: {_output_path}")
    print(f"Exported {len(_export_df)} rows with {len(_export_df.columns)} columns")


if __name__ == "__main__":
    main()
```
