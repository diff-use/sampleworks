"""Shared pytest fixtures for sampleworks tests."""

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from atomworks.io import parse
from sampleworks.models.boltz.wrapper import Boltz1Wrapper, Boltz2Wrapper, PredictArgs
from sampleworks.utils.setup import try_gpu


@pytest.fixture(scope="session")
def resources_dir() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def structure_1vme(resources_dir: Path) -> dict:
    return parse(str(resources_dir / "1vme_final.cif"))


@pytest.fixture(scope="session")
def structure_6b8x(resources_dir: Path) -> dict:
    return parse(str(resources_dir / "6b8x_final.pdb"))


@pytest.fixture(
    scope="session", params=["1vme_final.cif", "6b8x_final.pdb"], ids=["cif", "pdb"]
)
def test_structure(request, resources_dir: Path) -> dict:
    return parse(str(resources_dir / request.param))


@pytest.fixture(scope="session")
def boltz1_checkpoint_path() -> Path:
    path = Path("~/.boltz/boltz1_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz1 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def boltz2_checkpoint_path() -> Path:
    path = Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    if not path.exists():
        pytest.skip(f"Boltz2 checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def device() -> torch.device:
    return try_gpu()


@pytest.fixture(scope="session")
def boltz1_wrapper(boltz1_checkpoint_path: Path, device: torch.device) -> Boltz1Wrapper:
    predict_args = PredictArgs(
        recycling_steps=1, sampling_steps=10, diffusion_samples=1
    )
    return Boltz1Wrapper(
        checkpoint_path=boltz1_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture(scope="session")
def boltz2_wrapper(boltz2_checkpoint_path: Path, device: torch.device) -> Boltz2Wrapper:
    predict_args = PredictArgs(
        recycling_steps=1, sampling_steps=10, diffusion_samples=1
    )
    return Boltz2Wrapper(
        checkpoint_path=boltz2_checkpoint_path,
        use_msa_server=True,
        predict_args=predict_args,
        device=device,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Generator[Path, None, None]:
    output_dir = tmp_path / "boltz_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
