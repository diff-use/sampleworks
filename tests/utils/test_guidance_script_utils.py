"""Tests for guidance_script_utils saving helpers."""

from pathlib import Path

import torch
from sampleworks.utils.guidance_script_utils import save_everything

from tests.utils.atom_array_builders import build_test_atom_array


def test_save_everything_uses_model_atom_array_for_mismatch(tmp_path: Path):
    """Mismatch final_state should save with model template when provided."""
    refined_structure = {"asym_unit": build_test_atom_array(n_atoms=3, with_occupancy=True)}
    model_atom_array = build_test_atom_array(n_atoms=5, with_occupancy=False)

    final_state = torch.zeros((1, 5, 3), dtype=torch.float32)

    save_everything(
        output_dir=tmp_path,
        losses=[],
        refined_structure=refined_structure,
        traj_denoised=[],
        traj_next_step=[],
        scaler_type="pure_guidance",
        final_state=final_state,
        model_atom_array=model_atom_array,
    )

    assert (tmp_path / "refined.cif").exists()
