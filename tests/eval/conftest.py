"""
Shared fixtures for `tests/eval/`
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest


_RESOURCES = Path(__file__).resolve().parents[1] / "resources" / "1vme"
_REAL_CCP4 = _RESOURCES / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
_REAL_CIF = _RESOURCES / "1vme_final_carved_edited_0.5occA_0.5occB.cif"

# Occupancy pairs used to produce distinct `(protein, occ_key)` groups.
# Each group reuses the same real CCP4/CIF via symlinks under different
# filenames to exercise the plumbing without multiple maps
_GROUP_OCCUPANCIES: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.4, 0.6),
)

DEFAULT_SELECTIONS: tuple[str, ...] = (
    "chain A and resi 326-339",
    "chain A and resi 326-332",
)


@dataclass(frozen=True)
class RsccFixture:
    """Paths produced by :func:`rscc_fixture_factory`.

    Attributes
    ----------
    root
        Top-level directory holding ``inputs/`` and ``grid_search_results/``.
    grid_search_results_path
        Directory scanned by the script for ``refined.cif`` files. Also
        receives the output ``rscc_results.csv``.
    grid_search_inputs_path
        Directory holding ``protein_configs.csv`` and the shared
        ``base_map_dir``.
    protein_configs_csv
        Path to the one-row ``protein_configs.csv``.
    selections
        Selection strings configured in the CSV, in order.
    groups
        Occupancy pairs configured (one per group), in configuration order.
    trials_per_group
        Number of trial directories created under each protein directory.
    """

    root: Path
    grid_search_results_path: Path
    grid_search_inputs_path: Path
    protein_configs_csv: Path
    selections: tuple[str, ...]
    groups: tuple[tuple[float, float], ...]
    trials_per_group: int


def _occ_str(a: float, b: float) -> str:
    """Mirror ``occupancy_to_str`` formatting for ``{A: a, B: b}`` pairs."""
    return f"{a}occA_{b}occB"


def _link(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``, creating parent dirs and replacing any
    existing link. Uses a symlink so the CCP4 is not duplicated per
    trial."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _populate(
    root: Path,
    n_groups: int,
    trials_per_group: int,
    selections: Sequence[str],
) -> RsccFixture:
    assert _REAL_CCP4.exists(), _REAL_CCP4
    assert _REAL_CIF.exists(), _REAL_CIF
    assert 1 <= n_groups <= len(_GROUP_OCCUPANCIES)
    assert trials_per_group >= 1
    assert len(selections) >= 1

    groups = _GROUP_OCCUPANCIES[:n_groups]

    inputs = root / "inputs"
    base_map_dir_rel = Path("1vme_fixture_dir")
    base_map_dir = inputs / base_map_dir_rel

    for a, b in groups:
        s = _occ_str(a, b)
        _link(_REAL_CCP4, base_map_dir / f"{s}.ccp4")
        _link(_REAL_CIF, base_map_dir / f"{s}.cif")
    # Default ``occ_list`` in get_reference_structure_coords looks these up.
    for s in ("1.0occA", "1.0occB"):
        _link(_REAL_CIF, base_map_dir / f"{s}.cif")

    configs_csv = inputs / "protein_configs.csv"
    configs_csv.parent.mkdir(parents=True, exist_ok=True)
    configs_csv.write_text(
        "protein,base_map_dir,selection,resolution,map_pattern,structure_pattern\n"
        f"1vme,{base_map_dir_rel},{';'.join(selections)},1.8,"
        "{occ_str}.ccp4,{occ_str}.cif\n"
    )

    # grid_search_results tree: protein_dir/model_dir/scaler_dir/trial_dir/refined.cif
    results_root = root / "grid_search_results"
    results_root.mkdir(parents=True, exist_ok=True)
    for a, b in groups:
        protein_dir = results_root / f"1vme_{_occ_str(a, b)}"
        scaler_dir = protein_dir / "BOLTZ_2_bench" / "fk_steering"
        for i in range(trials_per_group):
            trial_dir = scaler_dir / f"ens2_gw{1.0 + i}"
            _link(_REAL_CIF, trial_dir / "refined.cif")

    return RsccFixture(
        root=root,
        grid_search_results_path=results_root,
        grid_search_inputs_path=inputs,
        protein_configs_csv=configs_csv,
        selections=tuple(selections),
        groups=groups,
        trials_per_group=trials_per_group,
    )


@pytest.fixture
def rscc_fixture_factory(
    tmp_path: Path,
) -> Callable[..., RsccFixture]:
    """Return a factory that builds an RSCC fixture under ``tmp_path``.

    Parameters:

    - ``n_groups`` (default 2): number of occupancy groups to create.
    - ``trials_per_group`` (default 2): trial subdirectories per group.
    - ``selections`` (default :data:`DEFAULT_SELECTIONS`): selection strings.

    Tests call the factory once to get a :class:`RsccFixture` whose paths
    feed directly into an ``argparse.Namespace`` for the script.
    """

    def _factory(
        *,
        n_groups: int = 2,
        trials_per_group: int = 2,
        selections: Sequence[str] = DEFAULT_SELECTIONS,
    ) -> RsccFixture:
        return _populate(tmp_path, n_groups, trials_per_group, selections)

    return _factory
