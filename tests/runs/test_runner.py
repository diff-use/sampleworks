"""Unit tests for sampleworks.runs.runner argv builder."""

from __future__ import annotations

from pathlib import Path

import pytest
from sampleworks.runs import loader, runner


def test_argv_for_rf3_partial_matches_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Faithful translation: argv should match the canonical rf3_partial bash invocation."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    preset = loader.load_preset("rf3_partial")
    invocations = runner.build_invocations(preset, results_dir=Path("/results"))

    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.job.name == "rf3"
    assert inv.env["CUDA_VISIBLE_DEVICES"] == "4"
    assert inv.log_path == Path("/results/rf3_run.log")

    argv = inv.argv
    assert argv[:6] == ["pixi", "run", "-e", "rf3", "python", "/app/run_grid_search.py"]
    pairs = _argv_to_dict(argv[6:])
    assert pairs["--proteins"] == (
        "/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps/proteins.csv"
    )
    assert pairs["--model"] == "rf3"
    assert pairs["--scalers"] == "pure_guidance"
    assert pairs["--partial-diffusion-step"] == "120"
    assert pairs["--ensemble-sizes"] == "8"
    assert pairs["--gradient-weights"] == "0.0 0.005 0.01 0.02 0.035 0.05 0.1"
    assert pairs["--model-checkpoint"] == "/checkpoints/rf3_foundry_01_24_latest.ckpt"
    assert pairs["--output-dir"] == "/results/rf3"
    # store_true flags appear as bare keys (value=True in our dict)
    assert pairs["--gradient-normalization"] is True
    assert pairs["--augmentation"] is True
    assert pairs["--align-to-input"] is True


def test_argv_omits_false_bool_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial", overrides=["shared_args.gradient-normalization=false"]
    )
    inv = runner.build_invocations(preset, results_dir=Path("/results"))[0]
    assert "--gradient-normalization" not in inv.argv


def test_explicit_output_dir_in_args_wins_over_subdir_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[shared_args]\n"
        '[[jobs]]\nname = "j"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "sub"\n'
        'args = { "output-dir" = "/explicit/path" }\n'
    )
    preset = loader.load_preset(str(custom))
    inv = runner.build_invocations(preset, results_dir=Path("/results"))[0]
    pairs = _argv_to_dict(inv.argv[6:])
    assert pairs["--output-dir"] == "/explicit/path"


def test_all_models_has_four_jobs_with_distinct_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("all_models")
    invocations = runner.build_invocations(preset, results_dir=Path("/r"))
    assert [i.job.name for i in invocations] == ["boltz2_xrd", "boltz2_md", "rf3", "protenix"]
    gpu_assignments = [i.env["CUDA_VISIBLE_DEVICES"] for i in invocations]
    assert gpu_assignments == ["0,1", "2,3", "4,5", "6,7"]


def test_protenix_dual_uses_different_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("protenix_dual")
    invocations = runner.build_invocations(preset, results_dir=Path("/r"))
    pairs = [_argv_to_dict(i.argv[6:]) for i in invocations]
    assert pairs[0]["--model-checkpoint"] == "/extra_checkpoints/protenix_tiny_default_v0.5.0.pt"
    assert pairs[1]["--model-checkpoint"] == "/extra_checkpoints/protenix_mini_default_v0.5.0.pt"


def test_rf3_partial_chiral_off_flag_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial_chiral_off")
    inv = runner.build_invocations(preset, results_dir=Path("/r"))[0]
    assert "--disable-chiral-features" in inv.argv
    assert "--force-all" in inv.argv


def _argv_to_dict(tail: list[str]) -> dict[str, object]:
    """Turn ``[--a, 1, --b, --c, 2]`` into ``{'--a': '1', '--b': True, '--c': '2'}``."""
    out: dict[str, object] = {}
    i = 0
    while i < len(tail):
        flag = tail[i]
        assert flag.startswith("--"), f"unexpected positional: {flag}"
        if i + 1 < len(tail) and not tail[i + 1].startswith("--"):
            out[flag] = tail[i + 1]
            i += 2
        else:
            out[flag] = True
            i += 1
    return out
