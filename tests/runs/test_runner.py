"""Unit tests for sampleworks.runs.runner argv builder."""

from __future__ import annotations

from pathlib import Path

import pytest
from sampleworks.runs import loader, runner


def test_argv_for_rf3_partial_matches_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    """RF3 partial builds the canonical argv and auto-assigns all GPUs."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: [str(i) for i in range(8)])
    preset = loader.load_preset("rf3_partial")
    invocations = runner.build_invocations(preset, results_dir=Path("/results"))

    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.job.name == "rf3"
    assert inv.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert inv.log_path == Path("/results/rf3_run.log")

    argv = inv.argv
    assert argv[:6] == ["pixi", "run", "-e", "rf3", "python", "/app/run_grid_search.py"]
    pairs = _argv_to_dict(argv[6:])
    assert pairs["--proteins"] == "/data/inputs/proteins.csv"
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
    """False boolean args are omitted rather than emitted as bare CLI flags."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial", overrides=["shared_args.gradient-normalization=false"]
    )
    inv = runner.build_invocations(preset, results_dir=Path("/results"))[0]
    assert "--gradient-normalization" not in inv.argv


def test_explicit_output_dir_in_args_wins_over_subdir_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit per-job output-dir beats the output_subdir-derived default."""
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


def test_full_8gpu_has_four_jobs_with_distinct_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full_8gpu preset maps its four jobs onto distinct GPU pairs."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: [str(i) for i in range(8)])
    preset = loader.load_preset("full_8gpu")
    invocations = runner.build_invocations(preset, results_dir=Path("/r"))
    assert [i.job.name for i in invocations] == ["boltz2_xrd", "boltz2_md", "rf3", "protenix"]
    gpu_assignments = [i.env["CUDA_VISIBLE_DEVICES"] for i in invocations]
    assert gpu_assignments == ["0,1", "2,3", "4,5", "6,7"]


def test_single_job_presets_use_all_eight_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone single-job presets default to all eight visible GPUs."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: [str(i) for i in range(8)])
    all_gpus = "0,1,2,3,4,5,6,7"
    for name in ("boltz1", "boltz2_xrd", "boltz2_md", "rf3", "protenix"):
        preset = loader.load_preset(name)
        invocations = runner.build_invocations(preset, results_dir=Path("/r"))
        assert len(invocations) == 1
        assert invocations[0].env["CUDA_VISIBLE_DEVICES"] == all_gpus


def test_gpu_count_uses_visible_gpus_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto GPU allocation consumes visible GPU IDs in preset order."""
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: ["4", "5", "6"])
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[shared_args]\n"
        '[[jobs]]\nname = "a"\nenv = "rf3"\ngpu_count = 2\noutput_subdir = "a"\n'
        '[[jobs]]\nname = "b"\nenv = "rf3"\ngpu_count = 1\noutput_subdir = "b"\n'
    )
    preset = loader.load_preset(str(custom))
    invocations = runner.build_invocations(preset, results_dir=tmp_path / "results")

    assert [inv.gpus for inv in invocations] == ["4,5", "6"]
    assert [inv.env["CUDA_VISIBLE_DEVICES"] for inv in invocations] == ["4,5", "6"]


def test_gpu_count_respects_explicit_gpu_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto GPU allocation skips GPUs already claimed explicitly."""
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: ["0", "1", "2", "3"])
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[shared_args]\n"
        '[[jobs]]\nname = "manual"\nenv = "rf3"\ngpus = "2"\noutput_subdir = "manual"\n'
        '[[jobs]]\nname = "auto"\nenv = "rf3"\ngpu_count = 2\noutput_subdir = "auto"\n'
    )
    preset = loader.load_preset(str(custom))
    invocations = runner.build_invocations(preset, results_dir=tmp_path / "results")

    assert [inv.gpus for inv in invocations] == ["2", "0,1"]


def test_gpu_count_rejects_insufficient_visible_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto GPU allocation fails clearly when visible GPUs are exhausted."""
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: ["0"])
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '[shared_args]\n[[jobs]]\nname = "a"\nenv = "rf3"\ngpu_count = 2\noutput_subdir = "a"\n'
    )
    preset = loader.load_preset(str(custom))

    with pytest.raises(RuntimeError, match="Not enough visible GPUs"):
        runner.build_invocations(preset, results_dir=tmp_path / "results")


def test_protenix_dual_uses_different_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Protenix dual preset uses separate tiny and mini checkpoints."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("protenix_dual")
    invocations = runner.build_invocations(preset, results_dir=Path("/r"))
    pairs = [_argv_to_dict(i.argv[6:]) for i in invocations]
    assert pairs[0]["--model-checkpoint"] == "/extra_checkpoints/protenix_tiny_default_v0.5.0.pt"
    assert pairs[1]["--model-checkpoint"] == "/extra_checkpoints/protenix_mini_default_v0.5.0.pt"


def test_rf3_partial_chiral_off_flag_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RF3 chiral-off preset passes the disable and force rerun flags."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial_chiral_off")
    inv = runner.build_invocations(preset, results_dir=Path("/r"))[0]
    assert "--disable-chiral-features" in inv.argv
    assert "--force-all" in inv.argv


def test_build_invocations_records_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_grid_search.py` assumes its --output-dir exists; the runner must mkdir it."""
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial")
    inv = runner.build_invocations(preset, results_dir=Path("/r"))[0]
    assert inv.output_dir == Path("/r/rf3")


def test_grid_search_script_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACTL wrappers can run the synced checkout instead of the baked /app copy."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("SAMPLEWORKS_GRID_SEARCH_SCRIPT", "/home/dev/workspace/run_grid_search.py")
    preset = loader.load_preset("rf3_partial")
    inv = runner.build_invocations(preset, results_dir=Path("/r"))[0]
    assert inv.argv[:6] == [
        "pixi",
        "run",
        "-e",
        "rf3",
        "python",
        "/home/dev/workspace/run_grid_search.py",
    ]


def test_gpu_validation_rejects_unavailable_gpu_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset for 8 GPUs should fail clearly on a smaller pod."""
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: ["0", "1", "2", "3"])
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[shared_args]\n"
        '[[jobs]]\nname = "ok"\nenv = "rf3"\ngpus = "0,1"\noutput_subdir = "ok"\n'
        '[[jobs]]\nname = "bad"\nenv = "rf3"\ngpus = "4,5"\noutput_subdir = "bad"\n'
    )
    preset = loader.load_preset(str(custom))
    invocations = runner.build_invocations(preset, results_dir=tmp_path / "results")

    with pytest.raises(RuntimeError, match="not visible"):
        runner._validate_gpu_assignments(invocations)


def test_gpu_validation_rejects_duplicate_gpu_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accidental GPU oversubscription is caught before jobs launch."""
    monkeypatch.setattr(runner, "_detect_available_gpus", lambda: ["0", "1"])
    monkeypatch.delenv("SAMPLEWORKS_ALLOW_GPU_OVERSUBSCRIPTION", raising=False)
    custom = tmp_path / "custom.toml"
    custom.write_text(
        "[shared_args]\n"
        '[[jobs]]\nname = "a"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "a"\n'
        '[[jobs]]\nname = "b"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "b"\n'
    )
    preset = loader.load_preset(str(custom))
    invocations = runner.build_invocations(preset, results_dir=tmp_path / "results")

    with pytest.raises(RuntimeError, match="same GPU"):
        runner._validate_gpu_assignments(invocations)


def test_uses_baked_env_python_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACTL image runs bypass pixi cache refreshes by calling env Python directly."""
    monkeypatch.delenv("SAMPLEWORKS_FORCE_PIXI", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    pixi_project = tmp_path / "app"
    python_bin = pixi_project / ".pixi" / "envs" / "rf3" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n")
    python_bin.chmod(0o755)
    monkeypatch.setenv("SAMPLEWORKS_PIXI_PROJECT_DIR", str(pixi_project))

    preset = loader.load_preset("rf3_partial")
    inv = runner.build_invocations(preset, results_dir=Path("/r"))[0]
    assert inv.argv[:2] == [str(python_bin), "/app/run_grid_search.py"]


def test_prebuilt_env_required_rejects_runtime_pixi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACTL runs fail clearly instead of installing missing pixi envs at runtime."""
    monkeypatch.delenv("SAMPLEWORKS_FORCE_PIXI", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_REQUIRE_PREBUILT_PIXI", "1")
    monkeypatch.delenv("SAMPLEWORKS_ALLOW_RUNTIME_PIXI", raising=False)
    monkeypatch.setenv("SAMPLEWORKS_PIXI_PROJECT_DIR", str(tmp_path / "app"))

    preset = loader.load_preset("rf3_partial")
    with pytest.raises(RuntimeError, match="Refusing to fall back to 'pixi run'"):
        runner.build_invocations(preset, results_dir=Path("/r"))


def test_dry_run_does_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--dry-run prints commands but never touches the filesystem."""
    monkeypatch.setenv("HOME", str(tmp_path))
    results_dir = tmp_path / "results"
    preset = loader.load_preset("rf3_partial")
    runner.run(preset, results_dir=results_dir, dry_run=True)
    # results_dir gets created by run() (for log file location) but per-job
    # output subdirs must NOT exist after dry-run.
    assert not (results_dir / "rf3").exists()


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
