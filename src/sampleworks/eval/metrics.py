import numpy as np
from loguru import logger


def rscc(array1, array2):
    """
    Calculate the Real Space Correlation Coefficient between two arrays.

    Returns NaN if correlation cannot be computed.
    """
    if array1.shape != array2.shape:
        # FIXME? should this raise an error @karson
        logger.warning(f"Shape mismatch: {array1.shape} vs {array2.shape}")
        return np.nan

    if array1.size == 0 or array2.size == 0:
        logger.warning("Empty array provided to rscc")
        return np.nan

    # Flatten arrays
    arr1_flat = array1.flatten()
    arr2_flat = array2.flatten()

    # Check for NaN/Inf
    # TODO: q for Karson: do you want to ignore these values instead?
    if not (np.isfinite(arr1_flat).all() and np.isfinite(arr2_flat).all()):
        logger.warning("NaN or Inf values in input arrays")
        return np.nan

    # Check for zero variance (constant arrays)
    if np.std(arr1_flat) < 1e-10 or np.std(arr2_flat) < 1e-10:
        logger.warning("Zero or near-zero variance in input arrays")
        return np.nan

    try:
        corr = np.corrcoef(arr1_flat, arr2_flat)[0, 1]
        return corr
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {e}")
        return np.nan
