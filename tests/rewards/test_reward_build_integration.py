"""End-to-end checks that a CLI invocation produces a usable reward function.

These cover the whole path a run takes: command line -> RewardConfig -> the built
reward -> prepare() -> a value. They are the tests that would have caught the
structure-factor reward being unreachable from the CLI.
"""

from pathlib import Path

import pytest
import torch
from sampleworks.core.rewards.config import build_reward
from sampleworks.core.rewards.protocol import (
    PreparableRewardFunctionProtocol,
    prepare_reward_if_needed,
    RewardFunctionProtocol,
    RewardInputs,
)
from sampleworks.core.rewards.real_space_density import RealSpaceRewardFunction
from sampleworks.core.rewards.registry import RewardBuildContext
from sampleworks.core.rewards.structure_factor import StructureFactorRewardFunction
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import load_guidance_structure


# Building either reward loads real experimental data through the qFit / SFcalculator
# stacks, the same code the gpu-marked reward tests exercise.
pytestmark = pytest.mark.gpu


BASE_ARGV = [
    "--model",
    "boltz2",
    "--guidance-type",
    "pure_guidance",
    "--protein",
    "1VME",
]


def build_from_argv(argv: list[str], structure_path: Path, device: torch.device):
    """Run the CLI path, then build the reward the resulting configuration describes."""
    config = GuidanceConfig.from_cli(argv)
    structure = load_guidance_structure(structure_path)
    return build_reward(
        config.resolved_reward_config(),
        RewardBuildContext(structure=structure, device=device),
    )


def test_density_reward_is_reachable_from_the_command_line(resources_dir: Path, device):
    structure_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
    density_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"

    reward = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--density",
            str(density_path),
            "--resolution",
            "1.8",
            "--loss-order",
            "1",
        ],
        structure_path,
        device,
    )

    assert isinstance(reward, RealSpaceRewardFunction)
    assert isinstance(reward.loss, torch.nn.L1Loss)


def test_structure_factor_reward_scores_a_structure_end_to_end(
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path], device: torch.device
):
    """--reward-type structure_factor: parse, build, prepare, and score."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths

    reward = build_from_argv(
        BASE_ARGV
        + [
            "--structure",
            str(structure_path),
            "--reward-type",
            "structure_factor",
            "--mtzfile",
            str(mtz_path),
            "--expcolumns",
            "Fprotein",
            "SIGFprotein",
        ],
        structure_path,
        device,
    )

    assert isinstance(reward, StructureFactorRewardFunction)
    assert isinstance(reward, PreparableRewardFunctionProtocol)

    structure = load_guidance_structure(structure_path)
    atom_array = structure["asym_unit"]
    prepare_reward_if_needed(reward, atom_array, device=device)
    reward_inputs = RewardInputs.from_atom_array(atom_array, ensemble_size=1, device=device)

    value = reward(
        reward_inputs.input_coords,
        reward_inputs.elements,
        reward_inputs.b_factors,
        reward_inputs.occupancies,
    )

    # The MTZ was generated from this structure, so the amplitudes agree: the value
    # is finite, non-negative, and near zero.
    assert torch.isfinite(value)
    assert 0.0 <= value.item() < 1e-2


def test_a_configuration_file_composes_two_rewards(
    tmp_path: Path,
    resources_dir: Path,
    sf_1vme_cif_and_mtz_paths: tuple[Path, Path],
    device: torch.device,
):
    """The two rewards score different data about the same structure, and combine."""
    structure_path, mtz_path = sf_1vme_cif_and_mtz_paths
    density_path = resources_dir / "1vme" / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
    config_file = tmp_path / "rewards.yaml"
    config_file.write_text(
        "real_space_density:\n"
        "  weight: 0.4\n"
        f"  reward_options: {{density: {density_path}, resolution: 1.8}}\n"
        "structure_factor:\n"
        "  weight: 0.6\n"
        f"  reward_options: {{mtzfile: {mtz_path}, expcolumns: [Fprotein, SIGFprotein]}}\n"
    )

    reward = build_from_argv(
        BASE_ARGV + ["--structure", str(structure_path), "--reward-config", str(config_file)],
        structure_path,
        device,
    )

    assert isinstance(reward, RewardFunctionProtocol)
    assert [type(term).__name__ for term in reward.rewards] == [
        "RealSpaceRewardFunction",
        "StructureFactorRewardFunction",
    ]
    assert reward.weights == [0.4, 0.6]
