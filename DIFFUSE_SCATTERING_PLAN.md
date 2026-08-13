# Plan: Diffuse Scattering as a Guidance Target (lunus.sf)

Status: planned, not started. Written 2026-08-12.

Adds diffuse scattering as an experimental target for guidance, using the
differentiable structure-factor engine in `lunus/lunus/sf/`.

## Why

Every reward in sampleworks today scores a **first moment** of the ensemble.
`RealSpaceRewardFunction.__call__` ends in `.sum(0)` over the batch with
occupancy `1/N` (`core/rewards/real_space_density.py:311`), which is a mean
density; `StructureFactorRewardFunction` does the analogous complex sum. The
ensemble is generated and then averaged away.

Diffuse scattering is the **second moment**: `<|F|²> − |<F>|²`. It is the part
of the experiment that actually reports on conformational spread, and
`lunus.sf.structure_factor_torch.mean_and_diffuse()` computes it
differentiably. lunus's "configurations" axis maps directly onto sampleworks'
ensemble batch axis.

The engine choice is also about scale: the `SFC_Torch` path is memory-bound on
its ASU-grid batch and does not reach large systems. lunus is the scale-oriented
path.

## Decisions (locked)

| Decision | Choice |
|---|---|
| Scope of the lunus reward | **Diffuse only** — score `<\|F\|²> − \|<F>\|²`; all amplitude targets stay with SFC_Torch |
| Existing `StructureFactorRewardFunction` | **Untouched** — separate effort, not in scope here |
| Batched configurations | **Landed in lunus** — `structure_factors_batch()`, see below |
| Packaging (`lunus.sf` importable) | **Done in the lunus repo** — PEP 621 `pyproject.toml`, installable as `lunus[sf]` |
| First target data | **Synthetic**, from a known multi-state ensemble |

## Prior art in this repo — read before building

- `core/rewards/structure_factor.py` — the closest existing analogue. Establishes
  the **two-phase construction** pattern (`__init__` stores config; `prepare()`
  builds everything needing the model atom array and device) that the diffuse
  reward should reuse. Also the pluggable-loss pattern (`AmplitudeLoss`, line 24).
- It is **complete and tested but not wired into the pipeline**: `.prepare()` is
  called only from tests, and `get_reward_function_and_structure()`
  (`utils/guidance_script_utils.py:295-311`) hardcodes `RealSpaceRewardFunction`.
  Phase 4 below builds the missing hook generically, so that effort can use it too.
- `core/rewards/protocol.py` — `RewardInputs`, and the `occupancies = 1/N`
  convention at line 121. See "Correctness traps".
- `utils/atom_reconciler.py` — alignment into the crystal frame. Diffuse is no
  more SE(3)-invariant than density; assume coordinates arrive pre-aligned.

## Upstream state (lunus, as of 2026-08-12)

Both prerequisites have landed. The adapter is written against this API:

```python
from lunus.sf import structure_factors_batch, mean_and_diffuse

structure_factors_batch(
    frac_coords_batch,    # (n_configs, n_atoms, 3), differentiable
    element_idx, occ,     # (n_atoms,) — one atom set shared across configs
    atom_A, atom_lam, elem_offsets, atom_radius_ang,
    grid_shape, orth_matrix, cell_volume, hkl, taper_width,
    blur=0.0, max_atoms_per_batch=50_000, grid_ops=None, compile_core=True,
    use_checkpoint=False,
) -> (n_configs, n_refl) complex
```

`lunus/lunus/sf/__init__.py` re-exports lazily (PEP 562), so the import above
costs nothing until first use. `mean_and_diffuse()` takes the result directly.

**Memory is solved; throughput is not.** `structure_factors_batch` is still a
Python loop over configurations, with `use_checkpoint=True` recomputing each
splat during backward instead of retaining it. Measured (3000 atoms, 90³ grid,
peak RSS above baseline):

| N | retained | checkpointed |
|---|---|---|
| 8 | 839 MB | 361 MB |
| 16 | 1604 MB | 371 MB |

Flat in N rather than linear, for ~2.4× the time, with bit-identical gradients
(`lunus/lunus/sf/tests/test_batch.py`). Fusing the splat into one kernel remains
open upstream and is listed under "Not yet built" in `lunus/lunus/sf/README.md`.
So guidance should assume **N sequential splat+FFT passes per guided step**, with
`use_checkpoint=True` as the default once N or the grid is large.

### Dependency wiring (sampleworks side, not yet done)

lunus now installs as an ordinary package (`pip install lunus[sf]`; extras pull
torch + gemmi, both already in the sampleworks envs). It needs adding to
whichever pixi envs run diffuse guidance. Note that lunus's own `[tool.pixi]`
tables are marked in its `pyproject.toml` as written-but-unresolved — irrelevant
to consuming it as a dependency, but do not copy them as a working example.

## Phases (sampleworks)

### Phase 1 — adapter

New: `core/forward_models/xray/lunus_sf.py`. Resolves the impedance mismatch
between sampleworks' reward inputs and lunus's calling convention:

- Cartesian Å → fractional (`inv(M) @ x`; differentiable, trivial).
- sampleworks scattering indices (`utils/elements.py`) → element symbols for
  `lunus.sf.elements.IT92_COEFFS`.
- Cell + space group → `orth_matrix()`, `grid_shape_for_resolution()`,
  `adjust_grid_for_symmetry()`, `build_grid_ops()`.
- Per-atom B via `build_atom_kernels_torch` (B is fixed at setup; guidance only
  needs ∂/∂coords).

**The one piece with no existing implementation**: `build_grid_ops_from_cctbx`
requires cctbx, which is not in the sampleworks pixi envs. A gemmi-based
equivalent has to be written — gemmi (pinned at `pyproject.toml:214`) exposes the
same rotations and translations.

All of this is coordinate-independent, so it lives behind
`prepare(atom_array, device)`.

### Phase 2 — the reward

New: `core/rewards/diffuse_scattering.py`, satisfying `RewardFunctionProtocol`.
Scores the variance term only. Loss pluggable:

- **MSE** for the synthetic milestone — target and calculation come off the same
  engine, so scales match by construction.
- **Resolution-shell-binned Pearson CC** for real data, which arrives on an
  arbitrary scale. Precedent: `lunus/lunus/sf/xtraj.py:1562` scores experimental
  diffuse by correlation, not residual.

A CC loss has different gradient character than the density reward, so DPS step
sizes will not transfer — expect to re-tune `--step-size`.

### Phase 3 — target ingestion (deferred)

Not needed for the synthetic milestone: targets are generated at integer hkl by
the same path that evaluates them.

For real data later: lunus lattices sample reciprocal space at sub-integer
intervals (`points_per_hkl`), while `compute_fcalc` extracts integer Miller
indices off the FFT grid, so the target needs reduction to integer hkl — see
`xtraj.py:1559` (`common_sets`). Read targets via `reciprocalspaceship`/gemmi
(both already dependencies) rather than cctbx's `any_reflection_file`.

### Phase 4 — pipeline wiring

- `--target-type {density,diffuse}` and dispatch in
  `get_reward_function_and_structure()`.
- A `prepare()` hook for two-phase rewards, in the trajectory scalers right after
  `process_structure_to_trajectory_input()` (`core/scalers/pure_guidance.py:83`,
  `core/scalers/fk_steering.py:108`) — the point where the model atom array and
  device are both known. Build it generically so the SFC reward can use it later.

### Phase 5 — tests

The **recovery test** is the milestone that says the whole thing works: build a
synthetic two-state ensemble via `synthetic/`, compute its diffuse, initialize a
mismatched ensemble, confirm guidance drives the diffuse loss down and the
ensemble spread toward truth.

Cross-engine validation against SFC_Torch is out of scope under diffuse-only —
SFC has no diffuse term. Parity of the underlying `|F|` is already covered by
lunus's own gemmi comparison (`lunus/lunus/sf/tools/compare_icalc_mtz.py`).

### Phase 6 — cost

Per guided step: N **sequential** splat+FFT+backward passes (see "Upstream
state"), ~2.4× that with `use_checkpoint=True`. For 1VME-sized cells the grid is
~90³ and this should be tolerable; a 300³ grid will dominate. The time-dependent
conditioning lever (AGENTS.md §5) applies naturally — coarsen the grid and
truncate resolution at high noise, sharpen as t→0. If wall-clock rather than
memory turns out to bind, the fused batched splat upstream is the lever.

## Correctness traps

### Occupancy convention

`RewardInputs` sets `occupancies = 1/N` (`core/rewards/protocol.py:121`), a
convention the sum-based rewards depend on. The diffuse reward must **divide it
out** and splat each configuration at full occupancy before averaging. Otherwise
both moments scale by `1/N²` and the variance term collapses toward zero — which
presents as "guidance does nothing" rather than as an error. Assert the
convention explicitly rather than silently correcting it.

### Diffuse-only guidance cannot see the mean structure

The variance term is invariant to anything shared across configurations: a common
rigid translation multiplies every `F_b` by the same phase factor, leaving both
`<|F|²>` and `|<F>|²` unchanged. So this reward constrains the ensemble's
*spread* and nothing about where it sits — the mean is held only by the
generative prior and `--align-to-input`.

Fine for the synthetic recovery test. For real work, diffuse will likely need to
compose with density or amplitudes as a second term, and **sampleworks has no
composite reward today** (`RewardFunctionProtocol` is single-valued). A weighted-sum
`CompositeRewardFunction` is the natural addition when that time comes;
deliberately out of scope here.

Confidence ~85% on the invariance argument — verify numerically before betting an
experimental design on it: translate one synthetic ensemble rigidly and confirm
the diffuse loss is unchanged.

## Sequencing

1. lunus repo: packaging + batched entry point.
2. Phases 1-2 (bulk of the work).
3. Phase 4 (small).
4. Phase 5 recovery test — the milestone.
5. Phase 3 and real data after that.
