"""Shared pytest fixtures for sampleworks tests."""

import importlib
import inspect
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from site import getsitepackages
from typing import Any, TYPE_CHECKING

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
from sampleworks.core.samplers.edm import AF3EDMSampler
from sampleworks.core.samplers.protocol import StepContext
from sampleworks.utils.imports import (
    BOLTZ_AVAILABLE,
    PROTENIX_AVAILABLE,
    RF3_AVAILABLE,
)
from sampleworks.utils.torch_utils import try_gpu

from tests.mocks import MockFlowModelWrapper


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


# ============================================================================
# Unified wrapper configuration
# ============================================================================


@dataclass(frozen=True)
class WrapperInfo:
    """Unified configuration for wrapper tests.

    Parameters
    ----------
    name
        Human-readable name for test IDs (e.g., "boltz1", "mock").
    fixture_name
        Name of the pytest fixture that provides the wrapper instance.
    annotate_fn_path
        Fully qualified path to the annotate function (empty string for mock).
    conditioning_type_path
        Fully qualified path to the conditioning type (empty string for mock).
    requires_out_dir
        Whether the annotate function requires an out_dir parameter.
    requires_slow
        Whether this wrapper requires @pytest.mark.slow (needs checkpoints).
    """

    name: str
    fixture_name: str
    annotate_fn_path: str
    conditioning_type_path: str
    requires_out_dir: bool = True
    requires_slow: bool = True


WRAPPERS: list[WrapperInfo] = [
    WrapperInfo(
        "boltz1",
        "boltz1_wrapper",
        "sampleworks.models.boltz.wrapper.annotate_structure_for_boltz",
        "sampleworks.models.boltz.wrapper.BoltzConditioning",
    ),
    WrapperInfo(
        "boltz2",
        "boltz2_wrapper",
        "sampleworks.models.boltz.wrapper.annotate_structure_for_boltz",
        "sampleworks.models.boltz.wrapper.BoltzConditioning",
    ),
    WrapperInfo(
        "protenix",
        "protenix_wrapper",
        "sampleworks.models.protenix.wrapper.annotate_structure_for_protenix",
        "sampleworks.models.protenix.wrapper.ProtenixConditioning",
    ),
    WrapperInfo(
        "rf3",
        "rf3_wrapper",
        "sampleworks.models.rf3.wrapper.annotate_structure_for_rf3",
        "sampleworks.models.rf3.wrapper.RF3Conditioning",
        requires_out_dir=False,
    ),
    WrapperInfo(
        "mock",
        "mock_wrapper",
        "",
        "",
        requires_out_dir=False,
        requires_slow=False,
    ),
]

STRUCTURES: list[str] = ["structure_1vme", "structure_6b8x"]


def get_wrapper_by_name(name: str) -> WrapperInfo:
    """Get wrapper info by name."""
    for wrapper in WRAPPERS:
        if wrapper.name == name:
            return wrapper
    raise ValueError(f"Unknown wrapper: {name}")


def get_wrapper_by_fixture(fixture_name: str) -> WrapperInfo:
    """Get wrapper info by fixture name."""
    for wrapper in WRAPPERS:
        if wrapper.fixture_name == fixture_name:
            return wrapper
    raise ValueError(f"Unknown wrapper fixture: {fixture_name}")


def get_wrapper_ids() -> list[str]:
    """Get wrapper IDs for parametrization."""
    return [info.name for info in WRAPPERS]


def get_fast_wrappers() -> list[WrapperInfo]:
    """Get wrappers that don't require slow tests (mock wrappers)."""
    return [info for info in WRAPPERS if not info.requires_slow]


def get_slow_wrappers() -> list[WrapperInfo]:
    """Get wrappers that require slow tests (real model checkpoints)."""
    return [info for info in WRAPPERS if info.requires_slow]


# ============================================================================
# Sampler test configuration
# ============================================================================


@dataclass(frozen=True)
class SamplerTestConfig:
    """Configuration for sampler tests.

    Parameters
    ----------
    sampler_factory
        Fully qualified path to the sampler class.
    is_trajectory_sampler
        Whether this sampler implements TrajectorySampler protocol.
    default_num_steps
        Default number of steps for testing.
    sampler_kwargs
        Keyword arguments for sampler construction (as tuple of tuples for hashability).
    """

    sampler_factory: str
    is_trajectory_sampler: bool = True
    default_num_steps: int = 10
    sampler_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class StepScalerTestConfig:
    """Configuration for step scaler tests.

    Parameters
    ----------
    scaler_factory
        Fully qualified path to the scaler class, or None for no scaler.
    requires_reward
        Whether this scaler requires a reward function.
    scaler_kwargs
        Keyword arguments for scaler construction (as tuple of tuples for hashability).
    """

    scaler_factory: str | None
    requires_reward: bool = False
    scaler_kwargs: tuple[tuple[str, Any], ...] = ()


SAMPLER_CONFIGS: dict[str, SamplerTestConfig] = {
    "edm": SamplerTestConfig(
        sampler_factory="sampleworks.core.samplers.edm.AF3EDMSampler",
        sampler_kwargs=(("augmentation", False), ("align_to_input", False)),
    ),
    "mock_trajectory": SamplerTestConfig(
        sampler_factory="tests.mocks.samplers.MockTrajectorySampler",
    ),
}

STEP_SCALER_CONFIGS: dict[str, StepScalerTestConfig] = {
    "none": StepScalerTestConfig(scaler_factory=None),
    "no_scaling": StepScalerTestConfig(
        scaler_factory="sampleworks.core.scalers.score_scalers.NoScalingScaler",
    ),
    "mock_step": StepScalerTestConfig(
        scaler_factory="tests.mocks.scalers.MockStepScaler",
        scaler_kwargs=(("step_size", 0.1),),
    ),
}

ALL_SAMPLER_IDS = list(SAMPLER_CONFIGS.keys())
FAST_SCALER_IDS = ["none", "no_scaling", "mock_step"]


def create_sampler(config: SamplerTestConfig, device: torch.device) -> Any:
    """Create sampler instance from config.

    Parameters
    ----------
    config
        The sampler test configuration.
    device
        The device to use for the sampler.

    Returns
    -------
    Any
        The instantiated sampler.
    """
    cls = _import_from_path(config.sampler_factory)
    kwargs = dict(config.sampler_kwargs)
    sig = inspect.signature(cls)
    if "device" in sig.parameters:
        kwargs["device"] = device
    return cls(**kwargs)


def create_scaler(config: StepScalerTestConfig) -> Any | None:
    """Create step scaler instance from config.

    Parameters
    ----------
    config
        The step scaler test configuration.

    Returns
    -------
    Any | None
        The instantiated scaler, or None if config.scaler_factory is None.
    """
    if config.scaler_factory is None:
        return None
    cls = _import_from_path(config.scaler_factory)
    return cls(**dict(config.scaler_kwargs))


def _import_from_path(path: str) -> Any:
    """Dynamically import an object from a fully qualified path."""
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def get_annotate_fn(wrapper: WrapperInfo) -> Callable[..., dict]:
    """Get the annotate function for a wrapper configuration."""
    if not wrapper.annotate_fn_path:
        raise ValueError(f"Wrapper {wrapper.name} does not have an annotate function")
    return _import_from_path(wrapper.annotate_fn_path)


def get_conditioning_type(wrapper: WrapperInfo) -> type:
    """Get the conditioning type for a wrapper configuration."""
    if not wrapper.conditioning_type_path:
        raise ValueError(f"Wrapper {wrapper.name} does not have a conditioning type")
    return _import_from_path(wrapper.conditioning_type_path)


def annotate_structure_for_wrapper(
    wrapper: WrapperInfo,
    structure: dict,
    temp_output_dir: Path | None = None,
    **kwargs: Any,
) -> dict:
    """Call the appropriate annotate function for a wrapper configuration.

    Parameters
    ----------
    wrapper
        The wrapper info configuration.
    structure
        The structure dictionary to annotate.
    temp_output_dir
        Temporary output directory (required if wrapper.requires_out_dir is True).
    **kwargs
        Additional keyword arguments passed to the annotate function.

    Returns
    -------
    dict
        The annotated structure.
    """
    annotate_fn = get_annotate_fn(wrapper)
    if wrapper.requires_out_dir:
        return annotate_fn(structure, out_dir=temp_output_dir, **kwargs)
    return annotate_fn(structure, **kwargs)


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
    from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
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
        RealSpaceRewardFunction,
        setup_scattering_params,
    )

    params = setup_scattering_params(
        structure_1vme_density["asym_unit"], em_mode=False, device=device
    )
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
    atom_array.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_full_occupancy():
    """AtomArray with some atoms having full occupancy."""

    atom_array = AtomArray(8)
    atom_array.coord = np.random.rand(8, 3)
    atom_array.set_annotation("chain_id", np.array(["A"] * 8))
    atom_array.set_annotation("res_id", [1, 1, 2, 3, 4, 5, 5, 5])
    atom_array.set_annotation("res_name", np.array(["ALA"] * 8))
    atom_array.set_annotation("atom_name", np.array(["CA"] * 8))
    atom_array.set_annotation("element", np.array(["C"] * 8))
    atom_array.set_annotation("altloc_id", np.array(["A", "B", "A", "A", "A", "A", "B", "C"]))
    atom_array.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.4, 1.0, 0.7, 0.2, 0.1]))
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
    atom_array_stack.set_annotation("res_id", np.array([1, 1, 2, 2, 2]))
    atom_array_stack.set_annotation("res_name", np.array(["ALA", "ALA", "VAL", "VAL", "VAL"]))
    atom_array_stack.set_annotation("atom_name", np.array(["CA"] * 5))
    atom_array_stack.set_annotation("element", np.array(["C"] * 5))
    atom_array_stack.set_annotation("altloc_id", np.array(["A", "B", "A", "B", "C"]))
    atom_array_stack.set_annotation("occupancy", np.array([0.5, 0.5, 0.6, 0.3, 0.1]))

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


# ============================================================================
# Mock wrapper and sampler fixtures for fast testing
# ============================================================================


@pytest.fixture
def mock_wrapper() -> MockFlowModelWrapper:
    """MockFlowModelWrapper with default settings."""
    return MockFlowModelWrapper(num_atoms=50)


@pytest.fixture
def mock_structure() -> dict:
    """Mock structure dict for testing without real PDB files."""
    atom_array = type(
        "MockAtomArray",
        (),
        {
            "occupancy": np.ones(50),
            "coord": np.random.randn(50, 3).astype(np.float32),
            "element": np.array(["C"] * 50),
            "b_factor": np.ones(50) * 20.0,
        },
    )()
    return {"asym_unit": [atom_array], "metadata": {"id": "mock"}}


@pytest.fixture
def converging_mock_wrapper() -> MockFlowModelWrapper:
    """MockFlowModelWrapper configured for convergence testing."""
    target = torch.randn(1, 50, 3)
    return MockFlowModelWrapper(num_atoms=50, target=target, convergence_rate=0.2)


@pytest.fixture
def edm_sampler(device: torch.device) -> AF3EDMSampler:
    """AF3EDMSampler configured for testing."""
    return AF3EDMSampler(
        device=device,
        augmentation=False,
        align_to_input=False,
    )


@pytest.fixture
def mock_trajectory_context() -> StepContext:
    """Valid StepContext for trajectory-based sampling."""
    return StepContext(
        step_index=0,
        total_steps=10,
        t=torch.tensor([1.0]),
        dt=torch.tensor([-0.1]),
        noise_scale=torch.tensor([1.003]),
    )
