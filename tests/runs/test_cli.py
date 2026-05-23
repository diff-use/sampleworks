"""End-to-end CLI tests (--list, --show, --dry-run, job shortcuts)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sampleworks.runs import cli


def test_list_prints_all_experiment_presets(capsys: pytest.CaptureFixture[str]) -> None:
    """``--list`` prints every bundled experiment preset exactly once."""
    exit_code = cli.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out.splitlines()
    assert set(out) == {
        "boltz",
        "boltz1",
        "boltz2",
        "boltz2_md",
        "boltz2_xrd",
        "full_8gpu",
        "protenix",
        "protenix_dual",
        "rf3",
        "rf3_partial",
        "rf3_partial_chiral_off",
        "rf3_protenix",
    }


def test_show_prints_resolved_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--show`` renders the resolved preset without launching jobs."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["--preset", "rf3_partial", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: rf3_partial" in out
    assert "gradient-weights" in out


def test_dry_run_does_not_invoke_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` prints commands and CUDA assignment instead of executing."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exit_code = cli.main(
        [
            "--preset",
            "rf3_partial",
            "--dry-run",
            "--results-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pixi run -e rf3 python /app/run_grid_search.py" in out
    assert "CUDA_VISIBLE_DEVICES=4" in out


def test_job_shortcut_filters_default_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A positional job shortcut filters the default full_8gpu preset."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["rf3,protenix", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: full_8gpu:rf3,protenix" in out
    assert "name: rf3" in out
    assert "name: protenix" in out
    assert "boltz2_xrd" not in out
    assert "boltz2_md" not in out


def test_model_target_uses_named_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single model target resolves to the matching standalone preset."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["boltz", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: boltz" in out
    assert "name: boltz2_xrd" in out
    assert "name: boltz2_md" in out


def test_boltz1_target_uses_named_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Boltz1 model has its own standalone preset target."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["boltz1", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: boltz1" in out
    assert "output_subdir: boltz1" in out


def test_jobs_filters_explicit_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--jobs`` filters an explicitly selected preset by job name."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["--preset", "full_8gpu", "--jobs", "rf3", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: full_8gpu:rf3" in out
    assert "name: rf3" in out
    assert "protenix" not in out


def test_job_shortcut_with_unknown_job_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown positional job shortcuts fail with a clear selector error."""
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(SystemExit, match="unknown jobs"):
        cli.main(["nonexistent", "--show"])


def test_set_override_propagates_through_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--set`` overrides are applied before the preset is displayed."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(
        [
            "--preset",
            "rf3_partial",
            "--set",
            "jobs.rf3.args.gradient-weights=0.0 0.01",
            "--show",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0.0 0.01" in out


def test_no_target_defaults_to_full_8gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running without a target resolves to the flagship full_8gpu preset."""
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: full_8gpu" in out
