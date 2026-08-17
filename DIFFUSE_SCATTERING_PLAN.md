# Plan: Diffuse Scattering as a Guidance Target (lunus.sf)

Status: planned, not started. Written 2026-08-12; upstream state and
correctness traps updated 2026-08-16.

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

## Upstream state (lunus, as of 2026-08-16)

The adapter is written against this API:

```python
from lunus.sf import structure_factors_batch, mean_and_diffuse, SolventModel

structure_factors_batch(
    frac_coords_batch,    # (n_configs, n_atoms, 3), differentiable
    element_idx, occ,     # (n_atoms,) shared, OR (n_configs, n_atoms) per-config
    atom_A, atom_lam, elem_offsets, atom_radius_ang,
    grid_shape, orth_matrix, cell_volume, hkl, taper_width,
    blur=0.0, max_atoms_per_batch=50_000, grid_ops=None, compile_core=True,
    use_checkpoint=False, solvent=None, supercell=None,
) -> (n_configs, n_refl) complex
```

`lunus/lunus/sf/__init__.py` re-exports lazily (PEP 562), so the import above
costs nothing until first use. `mean_and_diffuse()` takes the result directly.

### Per-configuration occupancies

`occ` may be `(n_atoms,)` (shared) or `(n_configs, n_atoms)` (per member), and
**gradients flow to `occ` as well as to coordinates**. Two consequences for this
work:

- Unequal ensemble populations are now expressible at the engine level. The
  `1/N` in `RewardInputs` is a sampleworks *choice*, no longer a constraint
  imposed from below. Refining populations is out of scope here, but this is
  where it would start.
- Varying hydration between members is expressed as one array slot at two
  occupancies, never as atoms entering and leaving. The atom array must
  correspond across configurations — same slots, same count — which the
  reconciler already guarantees.

### Bulk solvent

`lunus/lunus/sf/solvent_torch.py` implements the flat/mask model:
`solvent_mask`, `f_solvent`, `f_total`, `calibrate_cutoff`, `mask_occupancy`,
`shell_voxels`, and a `SolventModel` config object. Passing `solvent=None`
leaves today's behaviour bit-identical.

**Masks are applied per configuration, and that is the only choice that reaches
diffuse.** One mask shared across the ensemble has zero variance: it changes
`<F>` but contributes exactly nothing to `<|F|²> − |<F>|²`. Per-configuration
masks fluctuate anti-correlated with the protein — where an atom moves out,
solvent moves in — which is the excluded-volume contrast correction. Note this
decides the `mask(⟨ρ⟩)` vs `⟨mask(ρ)⟩` question the opposite way from SFC's
`bulk_solvent="combined"` default.

Two knobs to be aware of before using it:

- `mask_blur` is this model's probe radius (default σ ≈ 1.13 Å). Setting it to
  `0.0` restores an unsmoothed threshold and does **not** reproduce conventional
  solvent scales on real data.
- `detach_mask` defaults to `False`, which is the consistent choice — measured
  4.8× lower refinement loss on 7FPV with the mask live. See the trap below
  before changing it.

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
open upstream, as does checkpointing only the members that do not fit rather
than all of them; both are listed under "Not yet built" in
`lunus/lunus/sf/README.md`. So guidance should assume **N sequential splat+FFT
passes per guided step**, with `use_checkpoint=True` once N or the grid is large.

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

### Dependency wiring (done)

lunus is declared workspace-level in `pyproject.toml`:

```toml
lunus = {branch = "sf", extras = ["sf"], git = "https://github.com/lanl/lunus.git"}
```

and resolved into all 13 environments on both platforms. Two operational notes:

- **The lock can only be regenerated on Linux.** lunus is a git source dependency
  with no published wheel, so a solve must build it for linux-64, which macOS
  cannot do. Use the **Relock** workflow (Actions → Relock → pick the branch); it
  runs `pixi lock` on ubuntu-latest with pixi pinned to v0.73.0 and commits the
  result back. Do not hand-copy a lock generated elsewhere — a different pixi
  version rewrites far more than the one dependency, including dropping the
  `osx-arm64` entries entirely.
- **The branch pin resolves to a SHA** (currently `60bd44db4`). A later `pixi
  lock` silently advances it if `origin/sf` has moved. Switch `branch` to `rev`
  if the adapter starts depending on specific lunus behaviour, and drop the key
  entirely once `sf` merges upstream.

`scripts/install_lunus.sh` covers environments where the pixi declaration is not
in play; it becomes redundant once the declared route is verified end to end.

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

Note this is now a sampleworks-side convention only: lunus accepts
per-configuration occupancies and differentiates them (see "Upstream state"). The
`1/N` flattening is a choice this pipeline makes, and the place to revisit if
unequal ensemble populations ever become a target.

### The occupancy gradient is incomplete when the mask is detached

With `SolventModel(detach_mask=True)`, `d(F_total)/d(occ)` carries the protein
term but **not** the bulk term that replaces it as an occupancy falls. Forward
the two stay continuous; in the gradient they do not. The default is `False`, so
this only bites if something turns detaching on — most plausibly as a memory
measure inside a guidance run. Assert rather than document: a run that both
refines occupancies and detaches the mask is computing a gradient of a different
model than the one it evaluates.

### Diffuse-only guidance CAN see the mean structure — in a real crystal

An earlier draft of this plan claimed the opposite, at ~85% confidence, and
flagged it for numerical checking. **The check was run on 2026-08-17 and the
claim was wrong.**

The invariance argument holds only in P1. A common rigid translation multiplies
every `F_b` by one phase factor, leaving `<|F|²>` and `|<F>|²` unchanged —
measured relative deviation 9.2e-4 on 1VME chain A recomputed in P1, and that
residual is grid discretization (atoms moving relative to voxel centres), not
round-off: float32 noise in the same pipeline measures 3.1e-8.

In a real space group it fails, because translating the asymmetric unit is not
translating the crystal. A symmetry mate at `Rx + t` moves to `Rx + t + Rd`, so
the packing changes. Equivalently, in

```
F_total(h) = Σ_g exp(2πi h·t_g) · F_ASU(h R_g)
```

a translation by `d` multiplies each term by `exp(2πi (h R_g)·d)` — a phase
depending on `g`, which therefore cannot factor out of the sum. Measured on 1VME
(P 1 21 1) at 1.8 Å: relative deviation 0.72, roughly 800× the P1 residual.

**This is good news for the method.** Packing contacts against symmetry mates make
absolute position observable, so diffuse-only scoring is not the blind-to-the-mean
reward this plan assumed. The argument for needing a composite reward purely to
pin down the mean is correspondingly weaker — a composite may still be wanted for
signal strength or for combining data types, but not for that reason.

Both regimes are pinned by
`tests/synthetic/test_generate_synthetic_sf_lunus.py::TestSelfConsistency`.

## Related work: the lunus-backed synthetic generator

`src/sampleworks/synthetic/generate_synthetic_sf.py` (SFcalculator-backed) is
being ported to lunus as a **separate script first, merged behind an `--engine`
flag later**. It is worth doing before Phase 1 because it exercises every adapter
primitive — cell → orthogonalization matrix, space group → grid ops via gemmi,
elements → IT92, per-atom B → kernels, resolution → grid sizing — forward-only,
non-differentiably, against an oracle (SFC) that already works. Build those
shared pieces in `core/forward_models/xray/lunus_sf.py` and have the script
import them.

Two things settled since that discussion:

- **Bulk solvent is no longer a blocker.** `--simulate-solvent-and-scale` has a
  real lunus path now, so the lunus generator can emit both protein and total
  sets rather than rejecting the flag.
- **The generator should take a multi-model structure, not altlocs.** The
  existing script collapses altloc conformers into one `Fprotein` (deliberately —
  it matches the reward's `bulk_solvent="combined"` mode). That collapse destroys
  the second moment, so it cannot produce a diffuse target at all. An
  `AtomArrayStack` input — the same form guidance *outputs* — maps one model per
  configuration and feeds `structure_factors_batch` directly. Leave the altloc
  path on the SFC generator; the reward fixtures in `tests/rewards/conftest.py`
  depend on it.

Validation: same structure through both engines, compared with lunus's own
`tools/compare_icalc_mtz.py` (correlation and R-factor, overall and by shell).
Expect agreement at the level of lunus-vs-gemmi, ~0.999989 correlation and
R ≈ 0.0077 — the taper is the main source of the difference. That ~0.8% is a
floor under any fit that pairs a lunus-generated target with the SFC-based
reward, so prefer to keep engine pairs consistent.

## Sequencing

1. lunus repo: packaging, batched entry point, bulk solvent, per-config
   occupancies — **all landed as of 2026-08-16**.
2. The lunus-backed synthetic generator (above), which de-risks Phase 1.
3. Phases 1-2 (bulk of the work).
4. Phase 4 (small).
5. Phase 5 recovery test — the milestone.
6. Phase 3 and real data after that.
