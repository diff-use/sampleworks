from collections.abc import Iterable

from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
    ELEMENT_TO_SCATTERING_INDEX,
)


VALID_ELEMENTS = set(ATOMIC_NUM_TO_ELEMENT) | set(ATOM_STRUCTURE_FACTORS.keys())


def normalize_element(elem: str) -> str:
    """Normalize element symbol to title case (e.g., 'CA' -> 'Ca', 'c' -> 'C')."""
    normalized = elem.strip().title()
    if normalized not in VALID_ELEMENTS:
        logger.warning(f"Unrecognized element symbol: '{elem}' (normalized: '{normalized}')")
    return normalized


def element_to_scattering_idx(raw: str) -> int:
    """Normalize an element symbol and return its scattering tensor index.

    Unknown elements are logged and mapped to index 0 (the ``'?'``
    placeholder row, which carries zero scattering factors).

    Parameters
    ----------
    raw
        Raw element symbol as found on an ``AtomArray`` (e.g. ``'CA'``,
        ``'Fe2+'``, ``'SE'``).

    Returns
    -------
    int
        Index into the scattering parameter tensor built by
        ``setup_scattering_params``.
    """
    normalized = normalize_element(raw)
    idx = ELEMENT_TO_SCATTERING_INDEX.get(normalized)
    if idx is None:
        logger.warning(
            f"Element '{raw}' (normalized: '{normalized}') is not in the scattering "
            "index table and will contribute zero density."
        )
        return 0
    return idx


def elements_to_scattering_indices(elements: Iterable[str]) -> list[int]:
    """Map a sequence of raw element symbols to scattering tensor indices.

    Convenience wrapper around :func:`element_to_scattering_idx` for the
    common case of converting an entire ``AtomArray.element`` array.

    Parameters
    ----------
    elements
        Iterable of raw element symbols.

    Returns
    -------
    list[int]
        Scattering tensor indices, one per input element.
    """
    return [element_to_scattering_idx(e) for e in elements]
