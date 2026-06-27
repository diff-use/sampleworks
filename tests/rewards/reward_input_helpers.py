"""Shared input builders for reward tests."""

import torch
from sampleworks.utils.elements import elements_to_scattering_indices


def build_scattering_indices(atom_array, device: torch.device) -> torch.Tensor:
    """Map biotite element symbols to scattering-tensor indices (production path).

    Mirrors ``RewardInputs.from_atom_array``: uses ``elements_to_scattering_indices``
    (so ionic forms resolve correctly) and ``dtype=torch.long``. Shared by the reward
    test files (contract + structure-factor) so they build ``elements`` identically.
    """
    return torch.tensor(
        elements_to_scattering_indices(atom_array.element),
        device=device,
        dtype=torch.long,
    )
