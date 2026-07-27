"""Reward-specific pytest fixtures.

Fixtures here are scoped to the reward test suite (`tests/rewards/`) so they don't
pollute the global fixture namespace. They depend on cross-cutting fixtures defined in
the parent `tests/conftest.py` (`device`, `resources_dir`, `density_map_1vme`), which
pytest resolves up the conftest hierarchy automatically.

`density_map_1vme` deliberately stays in the top-level conftest because
`tests/utils/test_density_utils.py` also consumes it; a fixture lives at the lowest
common ancestor of everything that uses it.
"""

from pathlib import Path

import pytest
import torch
from atomworks.io.parser import parse


@pytest.fixture(scope="session")
def structure_1vme_density(resources_dir: Path):
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Structure not found at {cif_path}")
    return parse(cif_path, ccd_mirror_path=None)


@pytest.fixture(scope="session")
def reward_function_1vme(density_map_1vme, structure_1vme_density, device: torch.device):
    from sampleworks.core.rewards.real_space_density import (
        RealSpaceRewardFunction,
        setup_scattering_params,
    )

    params = setup_scattering_params(em_mode=False, device=device)
    rf = RealSpaceRewardFunction(density_map_1vme, params, torch.tensor([1], device=device))
    return rf


@pytest.fixture(scope="session")
def test_coordinates_1vme(structure_1vme_density, device: torch.device):
    atom_array = structure_1vme_density["asym_unit"]

    # Handle both AtomArray and AtomArrayStack
    if hasattr(atom_array, "stack_depth"):
        # AtomArrayStack - take first model
        atom_array = atom_array[0]

    mask = atom_array.occupancy > 0
    atom_array = atom_array[mask]
    coords = torch.from_numpy(atom_array.coord).to(device=device, dtype=torch.float32)
    return coords, atom_array
