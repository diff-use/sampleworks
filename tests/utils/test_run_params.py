"""Tests for flexible Sampleworks params.json handling."""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from sampleworks.utils.run_params import (
    apply_run_params,
    infer_model,
    infer_pixi_env,
    load_run_params,
    normalize_space_list,
    write_params_artifacts,
)


def _namespace(**overrides) -> Namespace:
    defaults = dict(
        params=None,
        proteins=None,
        model="boltz2",
        method="X-RAY DIFFRACTION",
        scalers="pure_guidance fk_steering",
        ensemble_sizes="1 2 4 8",
        gradient_weights="0.01 0.1 0.2",
        partial_diffusion_step=0,
        num_gd_steps="20",
        num_particles=3,
        fk_lambda=0.5,
        fk_resampling_interval=1,
        step_scaler_type="noisespace",
        gradient_normalization=False,
        augmentation=False,
        align_to_input=False,
        disable_chiral_features=False,
        track_chiral_features=False,
        loss_order=2,
        em=False,
        guidance_start=-1,
        output_dir="./grid_search_results",
        max_parallel="auto",
        dry_run=False,
        force_all=False,
        only_failed=False,
        only_missing=False,
        model_checkpoint="",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_run_params_from_file(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"model": "boltz2"}))

    params, source = load_run_params(str(params_path))

    assert params == {"model": "boltz2"}
    assert source == f"file:{params_path}"


def test_load_run_params_unwraps_diffuse_params_file(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "params_json": {"model": "rf3", "scalers": ["pure_guidance"]},
                "output_dir": "/data/results/run-001",
            }
        )
    )

    params, source = load_run_params(str(params_path))

    assert source == f"file:{params_path}"
    assert params == {
        "model": "rf3",
        "scalers": ["pure_guidance"],
        "output_dir": "/data/results/run-001",
    }


def test_infer_model_and_pixi_env():
    assert infer_model({"model": "boltz2"}) == "boltz2"
    assert infer_pixi_env("boltz2") == "boltz"
    assert infer_pixi_env("protenix") == "protenix"
    assert infer_pixi_env("rf3") == "rf3"


def test_infer_model_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown Sampleworks model"):
        infer_model({"model": "madeup"})


def test_normalize_space_list():
    assert normalize_space_list([8], item_type=int, param_name="ensemble_sizes") == "8"
    assert (
        normalize_space_list([0.1, 0.2], item_type=float, param_name="gradient_weights")
        == "0.1 0.2"
    )
    assert (
        normalize_space_list("0.1 0.2", item_type=float, param_name="gradient_weights")
        == "0.1 0.2"
    )


def test_normalize_space_list_rejects_object():
    with pytest.raises(ValueError, match="gradient_weights must be a string or list"):
        normalize_space_list({"bad": True}, item_type=float, param_name="gradient_weights")


def test_apply_run_params_maps_nested_guidance_and_output_dir(tmp_path):
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "out"
    params_path.write_text(
        json.dumps(
            {
                "model": "boltz2",
                "inputs": {"proteins": "/data/input/proteins.csv"},
                "guidance": {
                    "scalers": ["pure_guidance"],
                    "ensemble_sizes": [8],
                    "gradient_weights": [0.1, 0.2],
                    "partial_diffusion_step": 120,
                    "gradient_normalization": True,
                    "augmentation": True,
                    "align_to_input": True,
                },
                "paths": {"output_dir": str(output_dir)},
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path), output_dir=None))

    assert args.model == "boltz2"
    assert args.proteins == "/data/input/proteins.csv"
    assert args.output_dir == str(output_dir)
    assert args.scalers == "pure_guidance"
    assert args.ensemble_sizes == "8"
    assert args.gradient_weights == "0.1 0.2"
    assert args.partial_diffusion_step == 120
    assert args.gradient_normalization is True
    assert args.augmentation is True
    assert args.align_to_input is True


def test_apply_run_params_maps_nested_model_method(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": {"name": "boltz2", "method": "ELECTRON MICROSCOPY"},
                "inputs": {"proteins": "/data/input/proteins.csv"},
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path)))

    assert args.model == "boltz2"
    assert args.method == "ELECTRON MICROSCOPY"


def test_apply_run_params_maps_model_config_method_and_checkpoint(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model_config": {
                    "name": "boltz2",
                    "method": "NEUTRON DIFFRACTION",
                    "checkpoint": "/data/checkpoints/boltz2.ckpt",
                },
                "inputs": {"proteins": "/data/input/proteins.csv"},
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path)))

    assert args.model == "boltz2"
    assert args.method == "NEUTRON DIFFRACTION"
    assert args.model_checkpoint == "/data/checkpoints/boltz2.ckpt"


def test_apply_run_params_clears_method_for_non_boltz2(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": "protenix",
                "inputs": {"proteins": "/data/input/proteins.csv"},
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path)))

    assert args.method is None


def test_apply_run_params_rejects_method_for_non_boltz2(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": "rf3",
                "method": "MD",
                "inputs": {"proteins": "/data/input/proteins.csv"},
            }
        )
    )

    with pytest.raises(ValueError, match="method is only valid"):
        apply_run_params(_namespace(params=str(params_path)))


def test_apply_run_params_rejects_multiple_models(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"models": ["boltz2", "rf3"]}))

    with pytest.raises(ValueError, match="exactly one model"):
        apply_run_params(_namespace(params=str(params_path)))


def test_apply_run_params_rejects_conflicting_model_fields(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"model": "boltz2", "models": ["rf3"]}))

    with pytest.raises(ValueError, match="conflicting model"):
        apply_run_params(_namespace(params=str(params_path)))


def test_apply_run_params_rejects_missing_inputs(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"model": "boltz2"}))

    with pytest.raises(ValueError, match="inputs.proteins or proteins_path"):
        apply_run_params(_namespace(params=str(params_path), proteins=None))


def test_apply_run_params_rejects_bad_list_type(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": "boltz2",
                "inputs": {"proteins": "/data/input/proteins.csv"},
                "guidance": {"gradient_weights": {"bad": True}},
            }
        )
    )

    with pytest.raises(ValueError, match="gradient_weights must be a string or list"):
        apply_run_params(_namespace(params=str(params_path)))


def test_apply_run_params_preserves_unknown_params(tmp_path):
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "out"
    params_path.write_text(
        json.dumps(
            {
                "model": "boltz2",
                "inputs": {"proteins": "/data/input/proteins.csv"},
                "scientist_note": {"arbitrary": ["kept"]},
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path), output_dir=str(output_dir)))
    write_params_artifacts(args, output_dir)

    original = json.loads((output_dir / "params.original.json").read_text())
    assert original["scientist_note"] == {"arbitrary": ["kept"]}


def test_apply_run_params_cli_output_dir_wins(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "model": "rf3",
                "inputs": {"proteins": "/data/input/proteins.csv"},
                "paths": {"output_dir": "/ignored"},
            }
        )
    )

    args = apply_run_params(
        _namespace(params=str(params_path), output_dir="/data/results/from-cli")
    )

    assert args.output_dir == "/data/results/from-cli"


def test_apply_run_params_materializes_inline_proteins(tmp_path):
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "out"
    params_path.write_text(
        json.dumps(
            {
                "model": "protenix",
                "inputs": {
                    "proteins": [
                        {
                            "name": "1abc",
                            "structure": "/data/1abc.cif",
                            "density": "/data/1abc.ccp4",
                            "resolution": 1.8,
                        }
                    ]
                },
            }
        )
    )

    args = apply_run_params(_namespace(params=str(params_path), output_dir=str(output_dir)))

    assert args.proteins == str(output_dir / "inputs" / "proteins.csv")
    csv_text = (output_dir / "inputs" / "proteins.csv").read_text()
    assert "1abc,/data/1abc.cif,/data/1abc.ccp4,1.8" in csv_text


def test_write_params_artifacts(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"model": "boltz2", "inputs": {"proteins": "p.csv"}}))
    output_dir = tmp_path / "out"
    args = apply_run_params(_namespace(params=str(params_path), output_dir=str(output_dir)))

    write_params_artifacts(args, output_dir)

    assert (output_dir / "params.original.json").exists()
    assert (output_dir / "params.resolved.json").exists()
