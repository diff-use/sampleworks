"""
Integration tests for ``scripts/eval/rscc_grid_search_script.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "eval" / "rscc_grid_search_script.py"
)


def _load_script():
    """Import the script module by path so tests don't require it to be
    installed on ``sys.path``"""
    spec = importlib.util.spec_from_file_location("rscc_grid_search_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rscc_script():
    return _load_script()


def _make_args(fx) -> argparse.Namespace:
    return argparse.Namespace(
        grid_search_results_path=fx.grid_search_results_path,
        grid_search_inputs_path=fx.grid_search_inputs_path,
        protein_configs_csv=fx.protein_configs_csv,
        occupancies=None,
        target_filename="refined.cif",
        n_jobs=1,
    )


def _read_results(fx) -> pd.DataFrame:
    return pd.read_csv(fx.grid_search_results_path / "rscc_results.csv")


@pytest.mark.slow
def test_main_end_to_end_produces_csv(rscc_fixture_factory, rscc_script):
    """2 groups by 2 trials by 2 selections results in 8 success rows with non-NaN RSCC.

    This is a smoke test.
    """
    fx = rscc_fixture_factory(n_groups=2, trials_per_group=2)

    rscc_script.main(_make_args(fx))

    df = _read_results(fx)
    assert len(df) == 2 * 2 * 2

    expected_cols = {
        "protein",
        "altloc_occupancies",
        "model",
        "scaler",
        "ensemble_size",
        "trial_dir",
        "refined_cif_path",
        "protein_dir_name",
        "selection",
        "error",
        "rscc",
        "base_map_path",
    }
    assert expected_cols.issubset(df.columns)

    assert df["rscc"].notna().all(), "all RSCC values should be non-NaN on happy path"
    assert df["error"].isna().all(), "no error rows expected on happy path"
    assert set(df["selection"]) == set(fx.selections)


@pytest.mark.slow
def test_trial_parse_failure_emits_error_rows(rscc_fixture_factory, rscc_script, monkeypatch):
    """
    Trial level ``parse`` failure results in one NaN-RSCC row per valid selection.
    """
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)

    real_parse = rscc_script.parse

    def flaky_parse(path, ccd_mirror_path=None):
        if "refined" in str(path):
            raise RuntimeError("simulated trial parse failure")
        return real_parse(path, ccd_mirror_path=ccd_mirror_path)

    monkeypatch.setattr(rscc_script, "parse", flaky_parse)

    rscc_script.main(_make_args(fx))

    df = _read_results(fx)
    assert len(df) == len(fx.selections)
    assert df["rscc"].isna().all()
    assert df["error"].notna().all()
    assert set(df["selection"]) == set(fx.selections)


@pytest.mark.slow
def test_per_selection_failure_isolated(rscc_fixture_factory, rscc_script, monkeypatch):
    """One selection's RSCC raise must not abort the other selection's row."""
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1)
    bad_selection = fx.selections[0]

    real_rscc = rscc_script.rscc

    def flaky_rscc(a, b, _sel=bad_selection):
        if flaky_rscc.target_is_next:  # type: ignore[attr-defined]
            flaky_rscc.target_is_next = False  # type: ignore[attr-defined]
            raise RuntimeError("simulated rscc failure")
        return real_rscc(a, b)

    # The selection loop calls rscc once per selection, in the order configured
    # in the CSV. Fail on the first call (i.e. the first selection).
    flaky_rscc.target_is_next = True  # type: ignore[attr-defined]

    monkeypatch.setattr(rscc_script, "rscc", flaky_rscc)

    rscc_script.main(_make_args(fx))

    df = _read_results(fx).set_index("selection")
    assert len(df) == len(fx.selections)
    assert np.isnan(df.loc[bad_selection, "rscc"])
    assert pd.notna(df.loc[bad_selection, "error"])

    good_selection = fx.selections[1]
    assert not np.isnan(df.loc[good_selection, "rscc"])
    assert pd.isna(df.loc[good_selection, "error"])


@pytest.mark.slow
def test_selections_missing_from_ref_coords_produce_no_row(rscc_fixture_factory, rscc_script):
    """Selections absent from the reference structure produce no output row.

    The valid selections still emit rows — the missing one is silently
    skipped. Exercises the ``(protein, selection) not in ref_coords`` branch
    at the per selection loop.
    """
    selections = (
        "chain A and resi 326-339",
        "chain A and resi 326-332",
        "chain Z and resi 9999-10000",  # no atoms in 1vme
    )
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=1, selections=selections)

    rscc_script.main(_make_args(fx))

    df = _read_results(fx)
    assert len(df) == 2
    assert set(df["selection"]) == {selections[0], selections[1]}
    assert df["rscc"].notna().all()


@pytest.mark.slow
def test_caches_evict_on_group_transition(rscc_fixture_factory, rscc_script, monkeypatch):
    """``build_density_transformer`` is called once per ``(protein, occ_key)``
    group, not once per trial. Cache should hit inside a group and evict on
    transition."""
    fx = rscc_fixture_factory(n_groups=2, trials_per_group=2)

    real_build = rscc_script.build_density_transformer
    calls: list[object] = []

    def counting_build(base_xmap, *args, **kwargs):
        calls.append(base_xmap)
        return real_build(base_xmap, *args, **kwargs)

    monkeypatch.setattr(rscc_script, "build_density_transformer", counting_build)

    rscc_script.main(_make_args(fx))

    assert len(calls) == 2, f"expected 2 builds (one per group), got {len(calls)}"


@pytest.mark.slow
def test_trials_grouped_by_sort_order(rscc_fixture_factory, rscc_script):
    """Output rows for the same ``(protein, occ_key)`` are contiguous and
    sorted, so trials aren't interleaved by filesystem scan order."""
    fx = rscc_fixture_factory(
        n_groups=2, trials_per_group=2, selections=("chain A and resi 326-339",)
    )

    rscc_script.main(_make_args(fx))

    df = _read_results(fx)
    assert len(df) == 4

    dir_names = df["protein_dir_name"].tolist()
    assert dir_names == sorted(dir_names), (
        f"expected output sorted by (protein, occ_key); got {dir_names}"
    )

    # Same-group rows are contiguous (no interleaving).
    group_blocks = [
        dir_names[i] for i in range(len(dir_names)) if i == 0 or dir_names[i] != dir_names[i - 1]
    ]
    assert len(group_blocks) == len(set(group_blocks)), (
        f"group boundaries repeat — rows are interleaved: {dir_names}"
    )


@pytest.mark.slow
def test_cached_base_map_array_unchanged_across_trials(
    rscc_fixture_factory, rscc_script, monkeypatch
):
    """``base_xmap.array`` in the cache must not be mutated across trials.

    The script relies on ``copy.copy(base_xmap)`` producing a wrapper whose
    ``.array`` rebind doesn't touch the cached original. The cache eviction
    counter does not catch mutation of a retained cached map. This test
    does, by snapshotting ``.array`` after the first build and comparing
    after ``main()`` finishes.
    """
    fx = rscc_fixture_factory(n_groups=1, trials_per_group=2)

    real_build = rscc_script.build_density_transformer
    snapshot: dict[str, object] = {}

    def capturing_build(base_xmap, *args, **kwargs):
        if "xmap" not in snapshot:
            snapshot["xmap"] = base_xmap
            snapshot["initial_array"] = base_xmap.array.copy()
        return real_build(base_xmap, *args, **kwargs)

    monkeypatch.setattr(rscc_script, "build_density_transformer", capturing_build)

    rscc_script.main(_make_args(fx))

    xmap = snapshot["xmap"]
    initial = snapshot["initial_array"]
    assert np.array_equal(xmap.array, initial), (  # type: ignore[attr-defined]
        "cached base_xmap.array was mutated across trials"
    )
