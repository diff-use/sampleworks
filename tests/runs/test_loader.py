"""Unit tests for sampleworks.runs.loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from sampleworks.runs import loader


BUNDLED = ["all_models", "rf3_partial", "rf3_partial_chiral_off", "protenix_dual", "rf3_protenix"]


def test_list_bundled_presets_returns_the_five() -> None:
    names = loader.list_bundled_presets()
    assert set(names) == set(BUNDLED), f"unexpected bundled presets: {names}"


@pytest.mark.parametrize("name", BUNDLED)
def test_each_bundled_preset_loads(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(name)
    assert preset.name == name
    assert preset.jobs, f"{name} has no jobs"
    for job in preset.jobs:
        assert job.env in ("boltz", "protenix", "rf3")


def test_env_var_wins_over_defaults_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("DATA_DIR", "/from/env")
    preset = loader.load_preset("rf3_partial")
    assert preset.defaults["DATA_DIR"] == "/from/env"
    # PROTEINS_CSV expands to ${DATA_DIR}/proteins.csv; DATA_DIR overridden by env
    proteins = preset.shared_args["proteins"]
    assert proteins == "/from/env/proteins.csv"


def test_defaults_used_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial")
    expected = "/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps"
    assert preset.defaults["DATA_DIR"] == expected


def test_set_override_at_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("rf3_partial", overrides=["defaults.DATA_DIR=/custom"])
    assert preset.defaults["DATA_DIR"] == "/custom"
    assert preset.shared_args["proteins"] == "/custom/proteins.csv"


def test_set_override_at_job_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("all_models", overrides=["jobs.rf3.gpus=7"])
    assert preset.job("rf3").gpus == "7"


def test_set_override_at_job_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset("all_models", overrides=["jobs.0.gpus=9"])
    assert preset.jobs[0].gpus == "9"


def test_set_override_at_args_inside_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial", overrides=["jobs.rf3.args.gradient-weights=0.0 0.01"]
    )
    assert preset.job("rf3").args["gradient-weights"] == "0.0 0.01"


def test_set_coerces_bool_and_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    preset = loader.load_preset(
        "rf3_partial",
        overrides=[
            "shared_args.gradient-normalization=false",
            "jobs.rf3.args.partial-diffusion-step=200",
        ],
    )
    assert preset.shared_args["gradient-normalization"] is False
    # job.args["partial-diffusion-step"] doesn't exist by default in rf3_partial,
    # but --set should still create or override it
    assert preset.job("rf3").args["partial-diffusion-step"] == 200


def test_load_preset_from_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    custom = tmp_path / "mycustom.toml"
    custom.write_text(
        'description = "custom"\n'
        "[defaults]\n"
        'DATA_DIR = "/x"\n'
        "[shared_args]\n"
        'model = "rf3"\n'
        "[[jobs]]\n"
        'name = "j1"\n'
        'env = "rf3"\n'
        'gpus = "0"\n'
        'output_subdir = "j1"\n'
        "args = {}\n"
    )
    preset = loader.load_preset(str(custom))
    assert preset.name == "mycustom"
    assert preset.defaults["DATA_DIR"] == "/x"


def test_unknown_preset_raises() -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_preset("does_not_exist")


def test_undefined_variable_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[shared_args]\nproteins = "${NEVER_DEFINED_VAR}/x"\n'
        '[[jobs]]\nname = "j"\nenv = "rf3"\ngpus = "0"\noutput_subdir = "j"\nargs = {}\n'
    )
    monkeypatch.delenv("NEVER_DEFINED_VAR", raising=False)
    with pytest.raises(KeyError, match="NEVER_DEFINED_VAR"):
        loader.load_preset(str(bad))


def test_set_without_equals_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(ValueError, match="KEY=VALUE"):
        loader.load_preset("rf3_partial", overrides=["bogus_no_equals"])


def test_bad_env_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[[jobs]]\nname = "j"\nenv = "not_a_real_env"\ngpus = "0"\noutput_subdir = "j"\nargs = {}\n'
    )
    with pytest.raises(ValueError, match="env must be one of"):
        loader.load_preset(str(bad))
