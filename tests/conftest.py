"""Shared pytest fixtures for sampleworks tests."""

import sys
from collections.abc import Generator
from pathlib import Path
from site import getsitepackages
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch


# Add the project root to sys.path so that 'tests' can be imported as a package
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, stack
from sampleworks.utils.imports import BOLTZ_AVAILABLE, PROTENIX_AVAILABLE, RF3_AVAILABLE
from sampleworks.utils.torch_utils import try_gpu


if TYPE_CHECKING:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
    )
    from sampleworks.models.protenix.wrapper import ProtenixWrapper
    from sampleworks.models.rf3.wrapper import RF3Wrapper

if BOLTZ_AVAILABLE:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
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
def structure_6b8x_with_altlocs(resources_dir: Path) -> AtomArray | AtomArrayStack:
    return load_any(
        resources_dir / "6b8x" / "6b8x_final.pdb", altloc="all", extra_fields=["occupancy"]
    )


@pytest.fixture(scope="session", params=["1vme_final.cif", "6b8x_final.pdb"], ids=["cif", "pdb"])
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
    return Boltz1Wrapper(
        checkpoint_path=boltz1_checkpoint_path,
        use_msa_manager=True,
        device=device,
    )


@pytest.fixture(scope="session")
def boltz2_wrapper(boltz2_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    return Boltz2Wrapper(
        checkpoint_path=boltz2_checkpoint_path,
        use_msa_manager=True,
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

    params = setup_scattering_params(
        structure_1vme_density["asym_unit"], em_mode=False, device=device
    )
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


@pytest.fixture(scope="module")
def simple_atom_array():
    """Small AtomArray with valid coords, elements, occupancy, b_factor."""

    atom_array = AtomArray(5)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_nan_coords():
    """AtomArray with some NaN coordinates."""

    atom_array = AtomArray(5)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, np.inf, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture(scope="module")
def simple_atom_array_stack():
    """AtomArrayStack with 2 models."""

    arrays = []
    for i in range(2):
        atom_array = AtomArray(3)
        base_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_array.coord = base_coords + i * 0.1
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0]))
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)

    return atom_array_stack


@pytest.fixture(scope="module")
def basic_atom_array_altloc():
    """AtomArray with mixed altloc_ids."""

    atom_array = AtomArray(5)
    coord = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A", "A", "A", "A", "A"]))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.4, 0.5]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_full_occupancy():
    """AtomArray with some atoms having full occupancy."""

    atom_array = AtomArray(7)
    atom_array.coord = np.random.rand(7, 3)
    atom_array.set_annotation("chain_id", np.array(["A"] * 7))
    atom_array.set_annotation("res_id", np.arange(1, 8))
    atom_array.set_annotation("res_name", np.array(["ALA"] * 7))
    atom_array.set_annotation("atom_name", np.array(["CA"] * 7))
    atom_array.set_annotation("element", np.array(["C"] * 7))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B", "D", "E"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 1.0, 0.4, 1.0, 0.7]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack_altloc():
    """AtomArrayStack with 3 models."""

    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.arange(1, 6))
    atom_array_stack.set_annotation("res_name", np.array(["ALA"] * 5))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))
    atom_array_stack.set_annotation("altloc_id", np.array(["A", "B", "A", "C", "B"]))
    atom_array_stack.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 1.0, 0.4]))

    return atom_array_stack


@pytest.fixture(scope="module")
def atom_array_missing_altloc_id():
    """AtomArray without altloc_id annotation."""

    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("occupancy", np.ones(5))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_missing_occupancy():
    """AtomArray without occupancy annotation."""

    atom_array = AtomArray(5)
    atom_array.coord = np.random.rand(5, 3)
    atom_array.set_annotation("altloc_id", np.array(["A"] * 5))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_partial_overlap():
    """Two AtomArrays with partial overlap."""

    array1 = AtomArray(5)
    array1.coord = np.random.rand(5, 3)
    array1.set_annotation("chain_id", np.array(["A"] * 5))
    array1.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    array1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    array1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array1.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    array2 = AtomArray(5)
    array2.coord = np.random.rand(5, 3)
    array2.set_annotation("chain_id", np.array(["A"] * 5))
    array2.set_annotation("res_id", np.array([3, 4, 5, 6, 7]))
    array2.set_annotation("res_name", np.array(["VAL", "LEU", "SER", "THR", "TYR"]))
    array2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    array2.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))

    return array1, array2


@pytest.fixture(scope="module")
def atom_array_stacks_partial_overlap():
    """Two AtomArrayStacks with partial overlap."""

    arrays1 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays1.append(array)
    stack1 = stack(arrays1)
    stack1.set_annotation("chain_id", np.array(["A"] * 4))
    stack1.set_annotation("res_id", np.array([1, 2, 3, 4]))
    stack1.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU"]))
    stack1.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack1.set_annotation("element", np.array(["C", "C", "C", "C"]))

    arrays2 = []
    for i in range(2):
        array = AtomArray(4)
        array.coord = np.random.rand(4, 3) + i * 0.1
        arrays2.append(array)
    stack2 = stack(arrays2)
    stack2.set_annotation("chain_id", np.array(["A"] * 4))
    stack2.set_annotation("res_id", np.array([2, 3, 4, 5]))
    stack2.set_annotation("res_name", np.array(["GLY", "VAL", "LEU", "SER"]))
    stack2.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA"]))
    stack2.set_annotation("element", np.array(["C", "C", "C", "C"]))

    return stack1, stack2


@pytest.fixture(scope="module")
def basic_atom_array_multichain():
    """AtomArray with chain_id, res_id, atom_name annotations."""

    atom_array = AtomArray(10)
    coord = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation(
        "chain_id", np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    )
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))
    atom_array.set_annotation(
        "res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER", "ALA", "GLY", "VAL", "LEU", "SER"])
    )
    atom_array.set_annotation(
        "atom_name", np.array(["CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA", "CA"])
    )
    atom_array.set_annotation(
        "element", np.array(["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"])
    )
    return atom_array


@pytest.fixture(scope="module")
def atom_array_stack_simple():
    """AtomArrayStack with 3 models."""

    arrays = []
    for i in range(3):
        atom_array = AtomArray(5)
        atom_array.coord = np.random.rand(5, 3) + i * 0.1
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)
    atom_array_stack.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array_stack.set_annotation("res_id", np.arange(1, 6))
    atom_array_stack.set_annotation("res_name", np.array(["ALA"] * 5))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))

    return atom_array_stack
