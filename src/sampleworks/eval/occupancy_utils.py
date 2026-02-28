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
    if "native" in dir_name.lower():
        # this is a hack, it would be better to properly name the directory
        occ_a = 0.5
    elif "1.0occa" in dir_name.lower() or "1occa" in dir_name.lower():
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


def occupancy_to_str(**altloc_occupancies: float) -> str:
    """Convert altloc occupancies to the string format used in filenames.

    Zero-occupancy altlocs are omitted. Values are rounded to two decimal
    places to avoid floating-point artifacts in filenames.

    Parameters
    ----------
    **altloc_occupancies : float
        Keyword arguments mapping altloc labels to their occupancies.

    Returns
    -------
    str
        Underscore-joined occupancy string, e.g. ``"0.5occA_0.5occB"``.

    Raises
    ------
    ValueError
        If no altlocs have non-zero occupancy.

    Examples
    -------
    >>> occupancy_to_str(A=1.0, B=0.0)
    '1.0occA'
    >>> occupancy_to_str(A=0.0, B=1.0)
    '1.0occB'
    >>> occupancy_to_str(A=0.5, B=0.5)
    '0.5occA_0.5occB'
    >>> occupancy_to_str(A=0.25, B=0.75)
    '0.25occA_0.75occB'
    >>> occupancy_to_str(A=0.5, B=0.3, C=0.2)
    '0.5occA_0.3occB_0.2occC'
    """
    parts = []
    for label, occ in altloc_occupancies.items():
        if occ < 0:
            raise ValueError(f"Negative occupancy for altloc {label}: {occ}")
        occ = round(occ, 2)
        if abs(occ) > 1e-6:
            parts.append(f"{occ}occ{label}")
    if not parts:
        raise ValueError("At least one altloc must have non-zero occupancy")
    return "_".join(parts)
