"""Shared pytest fixtures for sampleworks tests."""

from collections.abc import Generator
from pathlib import Path
from site import getsitepackages
from typing import TYPE_CHECKING

import pytest
import torch
from atomworks.io.parser import parse
from sampleworks.utils.imports import BOLTZ_AVAILABLE, PROTENIX_AVAILABLE, RF3_AVAILABLE
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray
from biotite.structure.celllist import AtomArrayStack

from sampleworks.utils.torch_utils import try_gpu


if TYPE_CHECKING:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
        PredictArgs,
    )
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
    from sampleworks.models.rf3.wrapper import RF3Wrapper

if BOLTZ_AVAILABLE:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
        PredictArgs,
    )

if PROTENIX_AVAILABLE:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper

if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import RF3Wrapper


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def structure_1vme(resources_dir: Path) -> dict:
    return parse(resources_dir / "1vme" / "1vme_final.cif", ccd_mirror_path=None)


@pytest.fixture(scope="session")
def structure_6b8x(resources_dir: Path) -> dict:
    return parse(resources_dir / "6b8x" / "6b8x_final.pdb", ccd_mirror_path=None)


@pytest.fixture(scope="session")
def structure_6b8x_with_altlocs(resources_dir: Path) -> AtomArrayStack:
    return load_any(
        resources_dir / "6b8x" / "6b8x_final.pdb", altloc='all', extra_fields=["occupancy"]
    )


@pytest.fixture(
    scope="session", params=["1vme_final.cif", "6b8x_final.pdb"], ids=["cif", "pdb"]
)
def test_structure(request, resources_dir: Path) -> dict:
    # this requires the 1st 4 characters of the filename to match the folder name,
    # so PDB IDs need to be matching case
    return parse(resources_dir / request.param[:4] / request.param, ccd_mirror_path=None)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return try_gpu()


@pytest.fixture(scope="session")
def boltz1_checkpoint_path() -> Path:
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    path = Path("~/.boltz/boltz1_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz1 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def boltz2_checkpoint_path() -> Path:
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    path = Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz2 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def boltz1_wrapper(boltz1_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    predict_args = PredictArgs(recycling_steps=1, sampling_steps=10, diffusion_samples=1)
    return Boltz1Wrapper(
        checkpoint_path=boltz1_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture(scope="session")
def boltz2_wrapper(boltz2_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    predict_args = PredictArgs(recycling_steps=1, sampling_steps=10, diffusion_samples=1)
    return Boltz2Wrapper(
        checkpoint_path=boltz2_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture(scope="session")
def protenix_checkpoint_path() -> Path:
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    path = Path(getsitepackages()[0]) / "release_data/checkpoint/protenix_base_default_v0.5.0.pt"
    if not path.exists():
        pytest.skip(f"Protenix checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def protenix_wrapper(protenix_checkpoint_path: Path, device: torch.device):
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    return ProtenixWrapper(
        checkpoint_path=protenix_checkpoint_path,
        device=device,
    )


@pytest.fixture(scope="session")
def rf3_checkpoint_path() -> Path:
    if not RF3_AVAILABLE:
        pytest.skip("RF3 dependencies not installed in this environment")
    path = Path("~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"RF3 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def rf3_wrapper(rf3_checkpoint_path: Path):  # will run on Fabric device
    if not RF3_AVAILABLE:
        pytest.skip("RF3 dependencies not installed in this environment")
    return RF3Wrapper(
        checkpoint_path=rf3_checkpoint_path,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    output_dir = tmp_path / "boltz_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir


@pytest.fixture(scope="session")
def density_map_1vme(resources_dir: Path):
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (  # noqa: E501
        XMap,
    )

    map_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    if not map_path.exists():
        pytest.skip(f"Density map not found at {map_path}")
    return XMap.fromfile(str(map_path), resolution=1.8)


@pytest.fixture(scope="session")
def structure_1vme_density(resources_dir: Path):
    cif_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    if not cif_path.exists():
        pytest.skip(f"Structure not found at {cif_path}")
    return parse(cif_path, ccd_mirror_path=None)


@pytest.fixture(scope="session")
def reward_function_1vme(density_map_1vme, structure_1vme_density, device: torch.device):
    from sampleworks.core.rewards.real_space_density import (
        RewardFunction,
        setup_scattering_params,
    )

    params = setup_scattering_params(structure_1vme_density)
    rf = RewardFunction(density_map_1vme, params, torch.tensor([1], device=device))
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
