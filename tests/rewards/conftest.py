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


# Committed (cif, MTZ) pair for the SF reward — produced from generate_synthetic_sf.py
# using chain-A 1vme model (altloc occ 0.5/0.5, H + waters stripped) at 1.8 A with
# --simulate-solvent-and-scale --save-structure. The cif is kept in the P2_1 crystal
# frame rather than recentered like the carved density cif. All three fixtures below are
# consumed only within tests/rewards/, so (unlike density_map_1vme) they live here.
_SF_1VME_STEM = "1vme_final_crystalframe_0.5occA_0.5occB_1.80A"


@pytest.fixture(scope="session")
def mtz_path_1vme(resources_dir: Path) -> Path:
    # Synthetic target: Fprotein/SIGFprotein/PHIFprotein + Ftotal/SIGFtotal/PHIFtotal.
    # The SFC reward (v1) fits |Fprotein|.
    mtz_path = resources_dir / "1vme" / f"{_SF_1VME_STEM}.mtz"
    if not mtz_path.exists():
        pytest.skip(f"MTZ not found at {mtz_path}")
    return mtz_path


@pytest.fixture(scope="session")
def structure_1vme_sf(resources_dir: Path):
    from sampleworks.utils.atom_array_utils import load_structure_with_altlocs

    cif_path = resources_dir / "1vme" / f"{_SF_1VME_STEM}.cif"
    if not cif_path.exists():
        pytest.skip(f"SF model structure not found at {cif_path}")
    return load_structure_with_altlocs(cif_path)


@pytest.fixture(scope="session")
def test_coordinates_1vme_sf(structure_1vme_sf, device: torch.device):
    # No occupancy>0 filter here (unlike the density fixture) as SFC topology is fixed.
    atom_array = structure_1vme_sf
    coords = torch.from_numpy(atom_array.coord).to(device=device, dtype=torch.float32)
    return coords, atom_array


@pytest.fixture(scope="session")
def reward_function_1vme_sf(mtz_path_1vme, test_coordinates_1vme_sf, device: torch.device):
    from sampleworks.core.rewards.structure_factor import StructureFactorRewardFunction

    _, atom_array = test_coordinates_1vme_sf

    # normalize_amplitude=True scores normalized E-values (|Ec| vs sfc.Eo), which are
    # unit-variance per resolution shell, so the MSE can be tested on an absolute scale.
    reward_function = StructureFactorRewardFunction(
        mtz_path_1vme,
        expcolumns=["Fprotein", "SIGFprotein"],
        normalize_amplitude=True,
        device=device,
    )
    reward_function.prepare(atom_array)
    return reward_function
