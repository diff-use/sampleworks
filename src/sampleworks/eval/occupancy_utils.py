import re


def extract_protein_and_occupancy(dir_name: str) -> tuple[str, dict[str, float]]:
    """Extract protein name and altloc occupancies from a directory name.

    Parses all ``{value}occ{label}`` tokens found in *dir_name*.  The protein
    name is taken as the first underscore-delimited token that does not match
    the occupancy pattern.

    Parameters
    ----------
    dir_name : str
        Directory name to parse, e.g. ``"1vme_0.5occA_0.5occB"``.

    Returns
    -------
    tuple[str, dict[str, float]]
        ``(protein, altloc_occupancies)`` where *altloc_occupancies* maps
        uppercase altloc labels to their occupancy values.  The dict is empty
        when no occupancy tokens are found.

    Examples
    -------
    >>> extract_protein_and_occupancy('1vme_0.5occA_0.5occB')
    ('1vme', {'A': 0.5, 'B': 0.5})
    >>> extract_protein_and_occupancy('6b8x_1.0occA')
    ('6b8x', {'A': 1.0})
    >>> extract_protein_and_occupancy('5sop_1.0occB')
    ('5sop', {'B': 1.0})
    >>> extract_protein_and_occupancy('1abc_0.5occA_0.3occB_0.2occC')
    ('1abc', {'A': 0.5, 'B': 0.3, 'C': 0.2})
    """
    protein = dir_name.split("_")[0].lower()

    altloc_occupancies: dict[str, float] = {}
    for match in re.finditer(r"(\d+\.?\d*)occ([A-Za-z])", dir_name, re.IGNORECASE):
        label = match.group(2).upper()
        altloc_occupancies[label] = float(match.group(1))

    return protein, altloc_occupancies


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
    for label in sorted(altloc_occupancies, key=lambda l: str(l).upper()):
        occ = altloc_occupancies[label]
        if occ < 0 or occ > 1:
            raise ValueError(f"Invalid (<0 or >1) occupancy for altloc {label}: {occ}")
        occ = round(occ, 2)
        if abs(occ) > 1e-6:
            label_str = str(label).upper()
            parts.append(f"{occ}occ{label_str}")
    if not parts:
        raise ValueError("At least one altloc must have non-zero occupancy")
    return "_".join(parts)
