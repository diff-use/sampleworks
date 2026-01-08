from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
)


VALID_ELEMENTS = set(ATOMIC_NUM_TO_ELEMENT) | set(ATOM_STRUCTURE_FACTORS.keys())


def normalize_element(elem: str) -> str:
    """Normalize element symbol to title case (e.g., 'CA' -> 'Ca', 'c' -> 'C')."""
    normalized = elem.strip().title()
    if normalized not in VALID_ELEMENTS:
        logger.warning(f"Unrecognized element symbol: '{elem}' (normalized: '{normalized}')")
    return normalized
