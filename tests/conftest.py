"""Shared pytest fixtures for sampleworks tests."""

from collections.abc import Generator
from pathlib import Path
from site import getsitepackages

import pytest
import torch
from atomworks.io.parser import parse
from sampleworks.utils.torch_utils import try_gpu


try:
    from sampleworks.models.boltz.wrapper import (
        Boltz1Wrapper,
        Boltz2Wrapper,
        PredictArgs,
    )

    BOLTZ_AVAILABLE = True
except ImportError:
    BOLTZ_AVAILABLE = False

try:
    from sampleworks.models.protenix.wrapper import ProtenixWrapper

    PROTENIX_AVAILABLE = True
except ImportError:
    PROTENIX_AVAILABLE = False


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def structure_1vme(resources_dir: Path) -> dict:
    return parse(resources_dir / "1vme_final.cif")


@pytest.fixture(scope="session")
def structure_6b8x(resources_dir: Path) -> dict:
    return parse(resources_dir / "6b8x_final.pdb")


@pytest.fixture(
    scope="session", params=["1vme_final.cif", "6b8x_final.pdb"], ids=["cif", "pdb"]
)
def test_structure(request, resources_dir: Path) -> dict:
    return parse(resources_dir / request.param)


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
    predict_args = PredictArgs(  # type: ignore[possibly-unbound]
        recycling_steps=1, sampling_steps=10, diffusion_samples=1
    )
    return Boltz1Wrapper(  # type: ignore[possibly-unbound]
        checkpoint_path=boltz1_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture(scope="session")
def boltz2_wrapper(boltz2_checkpoint_path: Path, device: torch.device):
    if not BOLTZ_AVAILABLE:
        pytest.skip("Boltz dependencies not installed in this environment")
    predict_args = PredictArgs(  # type: ignore[possibly-unbound]
        recycling_steps=1, sampling_steps=10, diffusion_samples=1
    )
    return Boltz2Wrapper(  # type: ignore[possibly-unbound]
        checkpoint_path=boltz2_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture(scope="session")
def protenix_checkpoint_path() -> Path:
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    path = (
        Path(getsitepackages()[0])
        / "release_data/checkpoint//protenix_base_default_v0.5.0.pt"
    )
    if not path.exists():
        pytest.skip(f"Protenix checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def protenix_wrapper(protenix_checkpoint_path: Path, device: torch.device):
    if not PROTENIX_AVAILABLE:
        pytest.skip("Protenix dependencies not installed in this environment")
    return ProtenixWrapper(  # type: ignore[possibly-unbound]
        checkpoint_path=protenix_checkpoint_path,
        device=device,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    output_dir = tmp_path / "boltz_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
