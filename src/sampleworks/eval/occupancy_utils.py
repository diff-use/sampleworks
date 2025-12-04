import re


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


def occupancy_to_str(occ_a, use_6b8x_format=False):
    """Convert occupancy float to string format used in filenames."""
    if use_6b8x_format:
        return _occupancy_to_str_6b8x(occ_a)
    else:
        return _occupancy_to_str(occ_a)


def _occupancy_to_str(occ_a):
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


# TODO: @karson.chrispens can you fix your file paths so this isn't needed? Or generalize if this is a common case?
def _occupancy_to_str_6b8x(occ_a):
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
