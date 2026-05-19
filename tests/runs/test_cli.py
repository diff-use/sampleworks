"""End-to-end CLI tests (--list, --show, --dry-run, --only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sampleworks.runs import cli


def test_list_prints_all_bundled_presets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out.splitlines()
    assert set(out) == {
        "all_models",
        "rf3_partial",
        "rf3_partial_chiral_off",
        "protenix_dual",
        "rf3_protenix",
    }


def test_show_prints_resolved_preset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["rf3_partial", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: rf3_partial" in out
    assert "gradient-weights" in out


def test_dry_run_does_not_invoke_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    exit_code = cli.main(
        ["rf3_partial", "--dry-run", "--results-dir", str(tmp_path)]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pixi run -e rf3 python /app/run_grid_search.py" in out
    assert "CUDA_VISIBLE_DEVICES=4" in out


def test_only_filters_to_subset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(["all_models", "--only", "rf3,protenix", "--show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "name: rf3" in out
    assert "name: protenix" in out
    assert "boltz2_xrd" not in out
    assert "boltz2_md" not in out


def test_only_with_unknown_job_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(SystemExit, match="unknown jobs"):
        cli.main(["all_models", "--only", "nonexistent", "--show"])


def test_set_override_propagates_through_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    exit_code = cli.main(
        [
            "rf3_partial",
            "--set",
            "jobs.rf3.args.gradient-weights=0.0 0.01",
            "--show",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0.0 0.01" in out


def test_no_preset_and_no_list_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/test")
    with pytest.raises(SystemExit):
        cli.main([])
