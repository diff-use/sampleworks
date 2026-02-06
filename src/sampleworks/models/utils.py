"""Shared utilities for model implementations."""

from typing import Any

from biotite.structure import AtomArray, AtomArrayStack


def get_atom_array_from_model_input(
    features: Any, structure: dict, model_class_name: str
) -> AtomArray | AtomArrayStack:
    """Extract atom array handling model-specific differences.

    Parameters
    ----------
    features
        Model features as returned by featurize()
    structure
        Atomworks structure dictionary
    model_class_name
        Name of the model wrapper class (e.g., "ProtenixWrapper", "Boltz1Wrapper", "Boltz2Wrapper")

    Returns
    -------
    AtomArray | AtomArrayStack
        The atom array to use for reward computation
    """
    if model_class_name in ("ProtenixWrapper", "Boltz1Wrapper", "Boltz2Wrapper"):
        if hasattr(features, "conditioning") and hasattr(features.conditioning, "true_atom_array"):
            if features.conditioning.true_atom_array is not None:
                return features.conditioning.true_atom_array
        if hasattr(features, "conditioning") and hasattr(features.conditioning, "feats"):
            atom_array = features.conditioning.feats.get("true_atom_array")
            if atom_array is not None:
                return atom_array
    return structure["asym_unit"][0]
