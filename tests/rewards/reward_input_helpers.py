"""Shared input builders for reward tests."""

import torch
from sampleworks.utils.elements import elements_to_scattering_indices


def build_scattering_indices(atom_array, device: torch.device) -> torch.Tensor:
    """Map biotite element symbols to scattering-tensor indices (production path).

    Mirrors ``RewardInputs.from_atom_array``: uses ``elements_to_scattering_indices``
    (so ionic forms resolve correctly) and ``dtype=torch.long``. Shared by the reward
    test files (contract + structure-factor) so they build ``elements`` identically.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Atoms whose ``element`` annotation is mapped to scattering-tensor indices.
    device : torch.device
        Device the returned tensor is placed on.

    Returns
    -------
    torch.Tensor
        1-D ``torch.long`` tensor of scattering-tensor indices, one per atom, on ``device``.
    """
    return torch.tensor(
        elements_to_scattering_indices(atom_array.element),
        device=device,
        dtype=torch.long,
    )
