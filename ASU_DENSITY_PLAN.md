# Plan: asymmetric-unit density from lunus.sf

Status: planned, future work. Written 2026-08-16.

A lunus-backed alternative to `src/sampleworks/eval/generate_synthetic_density.py`,
producing density on the **crystallographic asymmetric unit** rather than in a
fabricated P1 box around the molecule.

## Approach

1. Take the ASU atoms as deposited.
2. `splat_density()` them onto a **unit cell** grid (fractional coordinates,
   symmetry-commensurate shape).
3. **Fold** the whole cell grid into an ASU-shaped accumulator using the space
   group's grid operations.
4. Write CCP4.

Step 3 replaces the symmetry *expansion* (`symmetrize_sum`) that `xtraj.py` does
today. No expansion is performed at any point.

## Why folding is equivalent to expanding and then cropping

Let `ρ(x)` be the density from splatting the ASU atoms alone — which spills
across ASU boundaries and wraps around the cell through the splat's modulo.

Expansion then restriction gives, for `x` in the ASU:

```
ρ_cell(x) = Σ_g ρ(g⁻¹ x)          # symmetrize_sum, then keep the ASU part
```

Folding gives, for the same `x`:

```
ρ_fold(x) = Σ_g ρ(g x)            # every cell point accumulated onto its ASU representative
```

`g` runs over the whole space group, and `{g⁻¹} = {g}` as a set, so the two sums
are term-for-term the same. **The folded ASU map is the expanded map, restricted
— exactly, not approximately.**

This also disposes of the boundary worry that makes naive cropping wrong. An atom
near an ASU boundary spills density across it, and its symmetry mate spills
density back in. Cropping `ρ` loses the returning contribution; folding collects
it, because the spilled-out voxels are themselves folded back onto their ASU
representatives.

## Why it is cheaper

| | work | peak memory |
|---|---|---|
| expand, then crop | `n_ops × N_cell` gathers | `N_cell` |
| fold | `N_cell` scatter-adds (one pass) | `N_cell` source + `N_cell/n_ops` result |

A factor of `n_ops` in arithmetic, and the result is `n_ops` times smaller. For a
space group of order 8 that is a real difference on a large cell, and it is the
output you actually wanted.

## Where folding does *not* substitute

`xtraj.py`'s structure-factor path needs the **full cell** grid, because
`compute_fcalc()` FFTs it to get `F(hkl)`. Folding to the ASU discards precisely
the periodic array the FFT operates on. So the expansion in xtraj is not
redundant work that folding could replace — the two serve different outputs:

- structure factors → expand, FFT the cell
- ASU density map → fold, write the ASU

Any change to xtraj here would be about *which* it computes, not about replacing
one with the other.

## Implementation

### 1. Grid must be symmetry-commensurate

Use `adjust_grid_for_symmetry()` (already in `lunus.sf.symmetry_torch`) so every
symmetry operation maps grid points exactly onto grid points. Without it the fold
needs interpolation and stops being exact. lunus already enforces this for the
expansion path; the same constraint applies unchanged.

### 2. Build the ASU index map, once

A precomputed integer array of shape `(Nu, Nv, Nw)` giving, for each cell grid
point, the index of its representative in a compact ASU-sized array — plus the
ASU point count. This is the only genuinely new piece of machinery.

Two sources for the ASU definition:

- **gemmi** — has real-space ASU masking on `Grid`. Preferred: gemmi is already a
  sampleworks dependency and cctbx is not. (~75% confident of the exact current
  API; check against the installed version.)
- **cctbx** — `maptbx.asymmetric_map` is the canonical tool and is what
  `build_grid_ops_from_cctbx` already relies on, but cctbx is absent from the
  sampleworks environments.

This is the same cctbx-vs-gemmi split as the Phase 1 grid-ops adapter in
`DIFFUSE_SCATTERING_PLAN.md`; both should land in
`core/forward_models/xray/lunus_sf.py` together.

### 3. The fold itself

One scatter-add over the flattened grid:

```python
asu = torch.zeros(n_asu_points, ...)
asu.scatter_add_(0, asu_index_flat, density_flat)
```

Differentiable, and **linear** — `fold(x + y) == fold(x) + fold(y)` — so protein
density and a solvent mask can be folded separately and combined afterwards, the
same property `fold_supercell()` documents for the supercell case. It is also the
same primitive the splat already uses for its scatter, so it inherits the same
performance characteristics.

### 4. Output

CCP4 with correct cell and symmetry headers. Note what changes relative to the
current generator: no fabricated P1 bounding-box cell, and **no coordinate
shift**. `generate_synthetic_density.py` currently subtracts `xmap_torch.origin`
from the atoms before saving the CIF because CCP4 cannot encode an arbitrary
Cartesian origin (`eval/generate_synthetic_density.py:213-218`). A cell-aligned
map needs none of that.

That is a behavioural difference, not just a simplification: existing paired
outputs such as `tests/resources/1vme/1VME_single_001_density_input.cif` carry
shifted coordinates, and anything consuming those pairs assumes it.

## Correctness notes

**Special positions.** An atom on a symmetry element is mapped onto itself by
part of the group, so its density is accumulated once per stabilizer element.
Folding and expansion behave identically here — both sum over the full group — so
this introduces no new discrepancy, but it is a modelling issue in both, handled
conventionally by reducing the atom's occupancy. Verify against a structure with
an atom on a special position before trusting either map.

**Boundary points.** ASU grid points on a boundary are shared between copies. The
index map must assign each cell point to exactly one representative (the standard
tie-break); if it does, the fold is a clean partition and exact. A consequence:
`sum(ASU) × n_ops == sum(cell)` is **not** an identity, so do not use it as the
validation test — use the index map's own accounting instead.

## Validation

1. **Fold vs expand-and-crop.** Compute both on a small structure in a
   non-trivial space group (P2₁2₁2₁, P4₃) and assert they agree to floating-point
   tolerance. This is the test that pins the claim above, and it is cheap.
2. **Partition check.** Every cell grid point maps to exactly one ASU index;
   `bincount` of the index map has no zeros and the counts sum to `N_cell`.
3. **Against gemmi.** gemmi's own density calculation plus its ASU masking, as an
   independent implementation.
4. **Against the existing path.** `compute_density_from_atomarray` for the same
   structure, acknowledging that grid conventions and kernel treatment differ, so
   this is a sanity comparison rather than a parity test.

## Effort

Small once the gemmi adapter exists — the fold is a few lines, the grid handling
is already in lunus, and CCP4 writing already exists in the repo. The ASU index
map is the only real work, and its cost is dominated by getting the gemmi API and
the boundary tie-break right rather than by volume of code.

## Open questions

- Should the CCP4 hold the compact ASU, or a full-cell array with non-ASU voxels
  zeroed? The compact form is smaller; the masked-cell form is easier for
  downstream tools that assume a cell-shaped grid.
- Does this replace `generate_synthetic_density.py` or sit alongside it? The
  output is a different object (true cell and symmetry vs P1 bounding box), so
  the existing consumers and test resources need auditing before anything is
  swapped.
- `--em` (electron scattering factors) has no lunus path — `elements.py` is IT92
  X-ray only — so that flag must error rather than silently produce X-ray density.
