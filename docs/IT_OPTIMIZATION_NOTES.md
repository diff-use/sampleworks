# IT-Optimization — Consolidated Notes & Migration Reference

**Purpose.** A single, self-contained copy of all the inference-time-optimization
(IT-opt) notes that live in the external `it_opt/` reference tree, gathered here so
this work can be done from inside Sampleworks without opening the `it_opt` workspace.

> **Provenance.** Consolidated on **2026-07-13** from the sibling reference tree
> `../it_opt/` (not part of this repo). Source documents:
> `README.md`, `IT_OPTIMIZATION_MIGRATION_NOTES.md`, `PROTENIX_CHANGES.md`,
> `BUG_AND_NEXT_STEPS.md`, `LATENT_SPACE_OPTIMIZATION.md`, `EDITING_TUTORIAL.md`,
> and `protenix/README.md`.
>
> **Upstream references.**
> - **Paper:** A. Maddipatla, A. Rzayev, M. Pegoraro, M. Pacesa, P. Schanda, A. Marx,
>   S. Vedula, A. M. Bronstein, *"Inference-time optimization for experiment-grounded
>   protein ensemble generation,"* arXiv:[2602.24007](https://arxiv.org/abs/2602.24007)
>   (27 Feb 2026) — the authoritative description of the method these notes port.
> - **Source (public backup of the reference tree):** https://github.com/sai-advaith/it_opt
>   — clone-able anywhere; see §0.
>
> **About file paths & links below.** All `path/...` references, line numbers, and
> `[link](...)` targets in the quoted sections are relative to the **external
> `it_opt/` tree**, *not* to Sampleworks. They are preserved verbatim for
> traceability and will **not** resolve as clickable links from here. When porting,
> treat line numbers as approximate — the migration plan (§2) is deliberately written
> against *code patterns to find*, which survive the port; the other sections carry
> `it_opt` line numbers for cross-reference only.
>
> **The one thing to read first:** §2, and within it the TL;DR — the reference loop
> is *not actually running Adam*. Fix that (Tier 1) before tuning anything else.

---

## Contents

0. [Where the IT-opt source lives (for review & porting)](#0-where-the-it-opt-source-lives-for-review--porting)
1. [Overview — the shared IT-opt idea](#1-overview--the-shared-it-opt-idea)
2. [Migration & port plan (the headline)](#2-migration--port-plan-the-headline)
3. [Protenix change map & bug review](#3-protenix-change-map--bug-review)
4. [Full bug report & next steps (all three forks)](#4-full-bug-report--next-steps-all-three-forks)
5. [Latent-space optimization deep-dive (AF3 method + AF2/OpenFold framing)](#5-latent-space-optimization-deep-dive)
6. [Per-function reference & safe-editing notes](#6-per-function-reference--safe-editing-notes)
7. [Appendix — Protenix fork: how it works & loop map](#7-appendix--protenix-fork-how-it-works--loop-map)

---

## 0. Where the IT-opt source lives (for review & porting)

**An assistant working *only* in the Sampleworks workspace can still read and port the IT-opt
source — but that source is not in this repo.** It lives in a sibling reference tree that is
**not tracked in git** and (currently) exists only on the original local machine:

- **Root (absolute, this machine):** `/Users/fengyu/Sampleworks_0622/it_opt/`
- **Relative to the Sampleworks repo root:** `../it_opt/`
- **Public backup (clone-able anywhere):** https://github.com/sai-advaith/it_opt

### Portable source — public backup on GitHub

The reference tree is mirrored publicly at **https://github.com/sai-advaith/it_opt**
(default branch `main`; verified to contain `protenix/it_optimization_manager.py` and the full
`src/losses/` layer). This is the source-of-truth whenever the local `../it_opt/` path is
absent — a remote pod, CI, or a fresh clone of Sampleworks:

```bash
git clone https://github.com/sai-advaith/it_opt.git
# port target is then:  it_opt/protenix/it_optimization_manager.py
```

An assistant can Read/Grep these by **absolute path** even though they sit outside the
workspace file tree (reads of the wider filesystem are allowed, subject to the session's
permission mode — the first access may prompt). The frictionless setup is to **add `../it_opt`
as an extra working directory for the session** (in Claude Code: the add-directory command, or
the `additionalDirectories` setting) so reads/greps there never prompt.

### The files that matter for the port (Protenix IT-opt layer)

| Purpose | Absolute path |
|---|---|
| **Driver / the loop** (port target) | `/Users/fengyu/Sampleworks_0622/it_opt/protenix/it_optimization_manager.py` |
| Loss modules (directory) | `/Users/fengyu/Sampleworks_0622/it_opt/protenix/src/losses/` |
| ↳ backbone-RMSD (default objective) | `.../src/losses/backbone_rmsd_loss_function.py` |
| ↳ anchor (L2 prior to trunk baseline) | `.../src/losses/anchor_loss_function.py` |
| ↳ NMR / bond-length / violation | `.../src/losses/{nmr,bond_length,violation}_loss_function.py` |
| Model wrapper (stepwise, differentiable access) | `/Users/fengyu/Sampleworks_0622/it_opt/protenix/src/utils/non_diffusion_model_manager.py` |
| Run config (weights, lr, steps) | `/Users/fengyu/Sampleworks_0622/it_opt/protenix/pipeline_configurations/rmsd_baseline.yaml` |
| Reference PDBs (RMSD / fold-switch experiment) | `/Users/fengyu/Sampleworks_0622/it_opt/protenix/it_optim_inputs/` |

Everything under `it_opt/protenix/src/protenix/**` is **stock upstream Protenix** — do *not*
port it; Sampleworks has its own model plumbing. The IT-opt layer to study is just the driver +
`src/losses/` + the model wrapper.

### Locating the exact code patterns the migration plan (§2) refers to

```bash
IT=/Users/fengyu/Sampleworks_0622/it_opt/protenix
grep -n "torch.optim.Adam"   "$IT/it_optimization_manager.py"   # re-created optimizer (Tier 1 / #1)
grep -n "clip_grad_norm_"    "$IT/it_optimization_manager.py"   # the two separate clips (Tier 1 / #3)
grep -n "denoise_net_batched\|get_x_0_hat_from_x_noisy_batched" \
                             "$IT/it_optimization_manager.py"   # the two forward passes (Tier 2 / #4)
grep -n "== 160\|save_msa_every_n_steps" "$IT/it_optimization_manager.py"  # hardcoded save step (Tier 4 / #8)
```

### ⚠️ Two caveats before relying on this

1. **Portability.** The absolute path above only exists on the original machine. On a remote
   pod, in CI, or a fresh clone of Sampleworks, `it_opt/` will **not** be present — an
   assistant will find *these notes* (they are in the repo) but not the *source*. Resolve it by
   cloning the public backup first: `git clone https://github.com/sai-advaith/it_opt.git`
   (then point the paths above at that clone). This is why vendoring the source into this repo
   is **not** necessary.
2. **License.** Protenix is **CC BY-NC 4.0 (non-commercial)**. Prefer **re-implementing the
   technique** into Sampleworks' own code (guided by §2–§6) over copying Protenix source
   verbatim into this repo. Do not vendor the stock `src/protenix/**` package; if you add a
   reference snapshot at all, confine it to the small custom IT-opt layer and mark it clearly.

---

## 1. Overview — the shared IT-opt idea

*Source: `it_opt/README.md`.*

The `it_opt/` reference tree demonstrates the **same inference-time optimization
(IT-opt) idea across three different AlphaFold 3 reimplementations** ("remakes"):
**AlphaFold 3** itself, **Boltz-2**, and **Protenix**. The point is that IT-opt is
not specific to one codebase — because all three share the same trunk → diffusion
architecture, the same embedding-optimization recipe ports to each. Each subdirectory
is a self-contained fork (or snapshot) of the upstream model with the IT-opt code
added on top.

### The shared idea

All three models share the same backbone: a trunk (Evoformer / Pairformer) produces
single (`s`) and pair (`z`) embeddings, and a diffusion module turns those embeddings
into 3-D coordinates.

```
trunk(MSA, templates, …) ──► (s, z) ──► diffusion ──► coords ──► confidence head
                              ▲ frozen during standard inference
```

Standard inference freezes `(s, z)` and samples coordinates once. **IT-opt instead
treats `(s, z)` (or a structured perturbation of them) as trainable parameters at
inference time**, unrolls the differentiable diffusion sampler, and runs gradient
descent on the embeddings to optimize an objective evaluated on the *denoised*
structure — no retraining, no weight updates. After optimization, a final sampling
pass with the optimized embeddings produces the structures that get saved.

The objective is task-dependent and is the main thing that differs between experiments:

| Objective | What it does | Where |
|---|---|---|
| Confidence (`iPTM` / `pTM` / `pLDDT`) | push the model toward higher self-confidence; useful for complex docking | af3, boltz |
| NMR restraints | bias sampling toward NMR-consistent structures (NOE / RDC / relaxation) | boltz, protenix |
| Backbone RMSD to a target | steer toward a specific conformation (e.g. a fold-switched state) | protenix |
| Anchor / regularization | keep the optimized embeddings near the trunk baseline (added to any objective) | all three |

A common regularization theme across the forks is an **anchor (L2) term** that keeps
the optimized `(s, z)` from drifting off the learned manifold, plus **early-stopping /
freezing** late in the diffusion schedule to limit distribution shift.

### Where the optimization loop lives in each fork

The heart of every fork is the same nested loop — *outer* (full diffusion passes /
fresh noise) → *diffusion timesteps* → *inner* (gradient steps on the embeddings):

| Fork | Entry point | Outer loop | Inner loop |
|---|---|---|---|
| af3 | `optimize_embeddings()` in `af3/src/alphafold3/model/embedding_optimization.py:817` | `:1276` | `:1330` (+ diffusion loop `:1288`) |
| boltz | `main()` in `boltz/scripts/experiments/boltz_embedding_{iptm,nmr}_optimization.py` | iPTM `:514` / NMR `:624` | iPTM `:538` / NMR `:654` (+ diffusion loop `:519`/`:635`) |
| protenix | `run_it_optimization()` in `protenix/it_optimization_manager.py:288` | `:321` | per-timestep step `:335` (2-level: one Adam step per diffusion step) |

### Reference directory layout (external)

```
it_opt/
├── af3/        AlphaFold 3 IT-opt (embedding optimization + coordinate guidance)
├── boltz/      Boltz-2 fork with embedding-space IT-opt (iPTM + NMR) and a
│               template↔diffusion fixed-point experiment
└── protenix/   Protenix fork with an IT-opt manager (RMSD / NMR / anchor losses)
```

- **`af3/`** — a *snapshot* of just the files needed to reproduce IT-opt of AlphaFold 3
  trunk embeddings: two model modules (`embedding_optimization.py`,
  `coordinate_guidance.py`), the SLURM/shell sweep scripts, and the base AF3 checkpoint.
- **`boltz/`** — a full *fork of Boltz-2* with IT-opt experiments under
  `scripts/experiments/` and custom loss modules under `src/boltz/experiments/`.
- **`protenix/`** — a *fork of Protenix* with a single-entry-point IT-opt driver
  (`it_optimization_manager.py`), YAML run configs, and a `src/losses/` package
  (backbone-RMSD, anchor, NMR, bond-length, violation). **This is the fork the
  Sampleworks port is based on.**

### Shared NMR data: `*/nmr_pipeline_inputs/`

Both `boltz/` and `protenix/` carry an identical `nmr_pipeline_inputs/` dataset of NMR
targets used by NMR-restraint optimization. Each target is keyed by its PDB ID:

```
nmr_pipeline_inputs/
├── metadata/<pdb_id>/<pdb_id>.json     # {"pdb_id", "seq"} — the chain sequence
├── pdbs/<pdb_id>/<pdb_id>.pdb          # original NMR reference structure
│                 <pdb_id>_fixed.pdb    # cleaned (e.g. PDBFixer'd) reference
└── restraints/<pdb_id>/<pdb_id>.str    # original BMRB/PDB restraint file
                        <pdb_id>.csv    # parsed restraints (one row per member)
```

Parsed restraint CSV columns: `type, constrain_id, member_id, member_logic,
residue1_num, residue1_id, atom1, heavy_atom1, residue2_num, residue2_id, atom2,
heavy_atom2, distance, lower_bound, upper_bound`.

### Attribution

Derivative works built on three upstream projects, retained under their respective
licenses: **AlphaFold 3** — Google DeepMind; **Boltz-1 / Boltz-2** — MIT License;
**Protenix** — ByteDance, CC BY-NC 4.0.

---

## 2. Migration & port plan (the headline)

*Source: `it_opt/IT_OPTIMIZATION_MIGRATION_NOTES.md`. This is the actionable plan for
porting `protenix/it_optimization_manager.py` (`ITOptimizationManager.run_it_optimization`)
into a new codebase. Each item is described by the **code pattern to find** (not line
numbers, which won't survive the port), plus whether it **changes behavior**.
Behavior-changing items must be A/B-validated on the actual loss (backbone RMSD) after
landing — do not assume they improve results.*

> Context: findings were produced by fanning out five review lenses (optimizer
> dynamics, compute/memory, autograd correctness, method design, engineering) and
> adversarially verifying each proposal against the source. 24 proposed → 13 survived
> verification. The single most important finding is Tier 1 below.

### TL;DR — the headline finding

The loop is **not actually running Adam optimization**. Because the optimizer is
re-created inside the per-step loop, every step starts from zeroed moments, so the
update degenerates to `lr · sign(g)` (signSGD / RProp). As a direct consequence,
**three configured knobs are silently inert**:

- `max_grad_norm` — provably a no-op (signSGD is invariant to positive gradient rescaling).
- `learning_rate` — really a fixed per-element step *size*, not an Adam learning rate.
- per-tensor vs. joint clipping — moot for the same reason.

Fix the optimizer lifetime first (Tier 1); most other tuning is meaningless until then.

### Tier 1 — Correctness: make the optimizer real (fix first)

**1. Do not re-create the optimizer inside the step loop**
- **Find:** `optimizer = torch.optim.Adam([optimized_s, optimized_z], lr=...)` located
  *inside* the `for diffusion_step` loop.
- **Change:** construct it **once per outer loop** — just before the inner
  `for diffusion_step`, inside `for diffusion_loop_iteration`. Keep `zero_grad()` and
  `step()` where they are.
- **Why per-outer-loop (not global):** the coordinates are resampled each outer loop,
  so momentum should not leak from the low-noise tail of one sweep into the high-noise
  start of the next.
- **Safe reference-wise:** `optimized_s` / `optimized_z` are created once and only
  mutated in place (never reassigned), so a hoisted optimizer keeps valid parameter
  references.
- **Behavior-changing: YES.**

**2. Re-tune the learning rate (couple with #1)**
- Current `~0.06` was effectively a signSGD step size. Real Adam momentum will overshoot
  at that value — sweep it down and validate on the loss trajectory.
- **Behavior-changing: YES** (validate together with #1).

**3. Use a single joint gradient clip**
- **Find:** two separate calls, `clip_grad_norm_([optimized_s], ...)` and
  `clip_grad_norm_([optimized_z], ...)`, to the same threshold.
- **Change:** one `clip_grad_norm_([optimized_s, optimized_z], max_grad_norm)`.
- **Why:** clipping `s` and `z` independently to the same norm distorts their relative
  gradient scale (`z` has ~30× more elements than `s`). Note: this clip does **nothing**
  until #1 is done.
- **Behavior-changing: YES** (only takes effect after #1).

### Tier 2 — Compute: one of the two forward passes is nearly redundant

**4. Advance the trajectory with the differentiable output instead of a second forward**
- **Find:** two denoiser calls per step — a differentiable `denoise_net_batched(...)`
  for the loss, then a separate `no_grad` `get_x_0_hat_from_x_noisy_batched(...)` to
  advance the trajectory.
- **Change:** delete the second call; advance with `it_optim_x0_hat.detach()` fed into
  `get_x_t_from_x_0_hat`.
- **Why:** the two passes use identical coordinates and `t_hat`; the only difference is
  the second uses embeddings one Adam step fresher. Removing it is ~2× on the inner loop.
- **Behavior-changing: YES** — introduces a one-step embedding lag that recurs every
  step. Validate output equivalence, or keep the exact post-step forward only for the
  last K low-noise steps.

### Tier 3 — Safe cleanups (no behavior change — port freely)

**5. Build the save-snapshot dict only at the save step**
- **Find:** `msa_dict_t = {s_inputs.clone().detach(), s_trunk..., z_trunk...}` built
  unconditionally every step, but consumed only inside the `if (diffusion_step+1)==160:`
  save guard.
- **Change:** move the dict construction inside the save guard. Removes ~199 discarded
  full-`z` clones (the ~0.5 GB tensor) per outer loop. Byte-identical output.

**6. Delete the dead re-detach line**
- **Find:** `noisy_structures = noisy_structures.clone().detach()` after `optimizer.step()`.
- **Change:** delete it — `noisy_structures` is never read again (the next iteration
  overwrites it, and the `no_grad` block below uses `structures`).

**7. Drop the redundant `.clone().detach()` on embeddings in the `no_grad` advance pass**
- **Find:** `s_inputs=optimized_s_inputs.clone().detach()` (and `s`/`z`) passed into the
  `no_grad` `get_x_0_hat_from_x_noisy_batched(...)` call.
- **Change:** pass `optimized_s_inputs, optimized_s, optimized_z` directly. Under
  `no_grad`, `detach()` is a no-op and the callee only reads them (`inplace_safe=False`).

### Tier 4 — Config / hygiene (wires up dead knobs; mostly safe)

**8. Honor the save-cadence config knob**
- **Find:** hardcoded `if (diffusion_step + 1) == 160:` (with a stale "between 160 and
  170" comment) while a `it_optimization_parameters.save_msa_every_n_steps` knob exists
  in config but is never read.
- **Change:** `if (diffusion_step+1) % save_cadence == 0 or (diffusion_step+1) == diffusion_N:`,
  hoist `save_cadence = int(...save_msa_every_n_steps)` above the loop, and
  `assert save_cadence > 0` (guards against divide-by-zero in the new modulo). Remove the
  magic `160` and fix the comment.

**9. Remove (or genuinely wire) the dead resume branch**
- **Find:** `checkpoint = None` hardcoded, making the `else: # Loading checkpoint` path
  unreachable.
- **Change:** delete the dead branch. If resume IS wanted in the new codebase, guard
  against the **batched** shapes (`[batch, N, c_s]` / `[batch, N, N, c_z]`), not the
  unbatched init tensors. Remember `s_inputs` is frozen, so restoring
  `checkpoint['s_inputs']` is a no-op unless deliberately different.

**10. Fix logging**
- Honor the `log_grads` knob: `clip_grad_norm_` already **returns** the pre-clip norm —
  capture and log it (free gradient-health signal; tells you whether the clip is actually
  biting). Gate the `.item()` sync on the knob so the disabled path stays free.
- Replace the unconditional per-step `print(...)` (~4000 lines/run, corrupts the tqdm
  bar) with `diffusion_steps.set_postfix(loss=..., s_gn=..., z_gn=...)`.

### Tier 5 — Method upgrades (optional, higher effort; from §5 / `LATENT_SPACE_OPTIMIZATION.md`)

**11. Optimize at intervals with multiple inner Adam steps**
- Instead of one update per denoise step, run `inner_steps > 1` Adam updates against a
  **fixed** noisy structure at optimizing steps only, and **skip the differentiable
  forward** on non-optimizing steps.
- **Why:** running several steps on a fixed landscape is what actually makes Adam's
  second moment engage; skipping pass (a) elsewhere is a large compute saving. Verified
  safe — the loss-forward feeds nothing downstream except the loss. Make `opt_interval`
  and `inner_steps` config params.
- For genuine noise-averaging, re-sample noise per inner draw (`get_x_noisy(structure,
  step)` with fresh randn) — re-detaching the same `noisy_structures` does **not**
  average over noise.
- **Behavior-changing: YES.**

**12. Early-stopping "cliffs"**
- Freeze a batch sample's embedding once its `‖z_i − z_init‖ / ‖z_init‖` exceeds a
  budget, by zeroing that row's grad before `step()` (between the clip and
  `optimizer.step()`).
- **Requires:** exposing the **per-sample** z-drift from the anchor loss — currently it
  computes `per_sample_loss_z` but reduces it to a mean before storing. Persist the
  per-sample vector.
- **Why it's safe:** per-sample losses are mean-reduced and the denoiser has **no
  batch-mixing layers** (no BatchNorm), so zeroing row `i` cleanly stops optimizing
  sample `i` only. Add a comment so a future batch-mixing layer doesn't silently break
  this invariant.
- **Behavior-changing: YES.**

**13. Structured / low-rank pair updates**
- Replace the fully free `optimized_z` (~16.6M params for a ~90-token protein at batch 16)
  with a **zero-initialized** structured delta over a frozen `z_init`, e.g. low-rank
  `Δ = UᵀV` on the residue axes.
- Wire the reconstructed `z` into **all** consumers: differentiable denoise, anchor loss,
  the trajectory advance pass, and the returned `optimized_dict`. Optimize `U,V` (not `z`)
  and clip *their* grads.
- Payoff is statistical (lower-variance, drift-resistant), not runtime. Re-tune
  `lambda_z` since the deviation-metric geometry changes. Prefer low-rank over
  per-channel `element_scale` (the latter can't change pairwise structure and will likely
  under-fit).
- **Behavior-changing: YES.**

### Suggested sequencing

1. **Tier 3** safe cleanups — bring over as-is.
2. **Tier 1** optimizer lifetime + joint clip + lr retune — one coupled change; A/B on
   backbone-RMSD loss. This is the correctness crux.
3. **Tier 2** single-forward compute win — behind an A/B for output equivalence.
4. **Tier 4** config / logging hygiene.
5. **Tier 5** method upgrades once the base loop is validated.

### Things to keep straight

- Tier 1's three items are **one** coupled change — validating them separately is
  misleading because the clip is inert until the optimizer is persistent.
- Everything in Tier 3 is genuinely free and safe.
- The coordinate-guidance path (`guidance_direction` in `get_x_t_from_x_0_hat`) is
  **dormant** in the current code (always `None`). If the new codebase intends a
  coordinate-guidance "tail," that is a separate feature to wire deliberately — it is not
  currently active.

---

## 3. Protenix change map & bug review

*Source: `it_opt/PROTENIX_CHANGES.md`. What was added on top of stock Protenix (PyTorch
AF3) to do IT-opt of the Pairformer "MSA" embeddings against an experimental/reference
objective — extracted and labeled — plus a focused bug review of the changed files.*

> **How the labels were derived.** The `it_opt` checkout is **not a git repo** and every
> file shares the same mtime, so upstream Protenix can't be diffed. Labels are inferred
> from (a) import topology — custom code imports from `src.utils` / `src.losses`, stock
> code lives under `src/protenix/**`; and (b) naming (`it_optimization*`, `*_loss_function`).
> Treat `[CUSTOM]` as "part of the IT-Opt layer," not a byte-level diff claim.

### 3.1 Change map — custom layer vs. stock Protenix

| Path | Label | Role in IT-Opt |
|---|---|---|
| `it_optimization_manager.py` | **[CUSTOM]** | **Entry point + driver.** The whole optimization loop. |
| `src/utils/non_diffusion_model_manager.py` | **[CUSTOM]** | Glue: `ProtenixModelManager` + `MSA` wrapper around the stock model; batched denoise helpers. |
| `src/losses/` | **[CUSTOM]** | All objective terms (main + anchor + validity). |
| `src/utils/io.py`, `pdb_parsing.py`, `mmseqs_query.py` | **[CUSTOM]** | Config load, PDB/bond parsing, MSA server query. |
| `pipeline_configurations/` | **[CUSTOM]** | YAML configs (`rmsd_baseline.yaml`). |
| `it_optim_inputs/` | **[CUSTOM]** | Reference structures (`2qke_fixed.pdb`, `5jyt_fixed.pdb`). |
| `src/protenix/**` | `[STOCK]` | Upstream Protenix AF3 (model, data, metrics). **Unchanged plumbing.** |
| `src/protenix/openfold_local/**` | `[STOCK]` | Vendored OpenFold utilities. |
| `src/af3-dev/**` | `[STOCK]` | Released model checkpoint + data. |
| `src/runner/inference.py` | `[STOCK]` | Stock inference entry; **not on the IT-Opt path** (IT-Opt runs through `it_optimization_manager.py`). |

**The IT-Opt layer is small:** one driver + one model wrapper + the `losses/` package +
a couple of util/config files. Everything under `src/protenix/` is untouched upstream.

### 3.2 The optimization loop — `run_it_optimization` (`:288`)

The Pairformer outputs `(s_trunk, z_trunk)` (what the paper calls the "MSA embeddings")
are made into **leaf tensors with `requires_grad=True`** and optimized by Adam; the
diffusion trajectory is advanced under `no_grad`.

```python
optimized_s.requires_grad = True          # the free variables = the embeddings themselves
optimized_z.requires_grad = True          # (raw tensors — NOT a structured Δ as in the JAX arm)

for diffusion_loop_iteration in range(outer_diffusion_steps):     # OUTER: resample noise
    structures = get_initial_latents()
    for diffusion_step in diffusion_steps:                         # INNER: reverse diffusion
        optimizer = torch.optim.Adam([optimized_s, optimized_z], lr=...)   # (see BUG-02)
        x0_hat = denoise_net_batched(noisy, s=optimized_s, z=optimized_z)  # denoise w/ current Z
        total_loss, _, main = get_loss_values(x0_hat, ...)         # objective
        total_loss.backward()
        clip_grad_norm_([optimized_s]); clip_grad_norm_([optimized_z])
        optimizer.step()                                           # update embeddings
        with torch.no_grad():                                      # advance trajectory
            x_0_hat = get_x_0_hat_from_x_noisy_batched(structures, s=..., z=...)
            structures = get_x_t_from_x_0_hat(structures, x_0_hat, step, step+1)
            structures = get_x_noisy(structures, step+1)
```

### 3.3 The objective — `get_loss_values` (`:254`)

```python
main_loss        = self.loss_function(x0_hat)            # BackboneRMSD / NMR / density
adversarial_s, _ = self.adversarial_loss_function(...)   # anchor: λ_s·‖s − s₀‖
adversarial_z    =                                       # anchor: λ_z·‖z − z₀‖
total_loss = main_loss + adversarial_s + adversarial_z
```

Same shape as the JAX arm's `loss = −score + anchor_s + anchor_z`, but `main_loss` is a
**physical / reference objective**, not ipTM. (The "adversarial" name is a misnomer —
it's the on-manifold anchor prior `‖Z − Z₀‖`.)

### 3.4 Objective options (`src/losses/`)

| File | Label | Objective |
|---|---|---|
| `backbone_rmsd_loss_function.py` | **[CUSTOM]** | RMSD of backbone to a **reference structure** (the `rmsd_baseline.yaml` default; semi-GT = a target conformation). |
| `nmr_loss_function.py` | **[CUSTOM]** | NOE flat-bottom distance restraints (+ RDC / S² order params). |
| `anchor_loss_function.py` | **[CUSTOM]** | Embedding prior `λ·‖Z − Z₀‖`. |
| `bond_length_loss_function.py`, `violation_loss_function.py` | **[CUSTOM]** | Validity (bond length / clash). |
| `density_loss_function.py` | **[MISSING]** | Imported by `__init__` but **absent** — see BUG-P1. |

### 3.5 Where the MLP idea would graft

Replace the two leaf tensors in `run_it_optimization` with an MLP output and optimize its
weights — nothing else changes:

```python
optimized_z = z_init + g_phi(z_init)            # your net
optimizer   = torch.optim.Adam(g_phi.parameters(), lr=...)
```

`get_loss_values` and the loop stay identical. *(In the Sampleworks port this is the role
of the latent-injection adapter — see the `latent_adapter` scaffold.)*

### 3.6 Bug review (Protenix IT-Opt path)

**🔴 BUG-P1 — NEW — missing loss modules → fatal `ImportError` at startup.** Not in the
full bug list (§4). `src/losses/__init__.py` imports four modules that **do not exist**:

```python
from .density_loss_function import DensityGuidanceLossFunction   # line 4 — FILE MISSING
from .nmr_loss_function import NMRLossFunction                   # line 5 — its own imports also missing:
#   nmr_loss_function.py → from .s_2_loss_function import S2LossFunction        (MISSING)
#   nmr_loss_function.py → from ..utils.hydrogen_addition import ...            (MISSING)
from .violation_loss_function import ViolationLossFunction       # line 6
#   violation_loss_function.py → from ..utils.openfold_violations.violations import ...  (MISSING)
```

Missing files confirmed absent: `density_loss_function.py`, `s_2_loss_function.py`,
`src/utils/hydrogen_addition.py`, `src/utils/openfold_violations/violations.py`. Because
`it_optimization_manager.py:6` does `from src.losses import *`, the **entry point dies at
import time** — before any optimization runs. NOE/density objectives are therefore
**non-functional** in the checkout; only `backbone_rmsd` + `anchor` have all their code
present. **Fix:** restore the four modules, or make `losses/__init__.py` import lazily /
guard optional objectives behind `try/except ImportError` so `backbone_rmsd` runs standalone.

**🔴 BUG-P2 — `_get_main_loss_function` ignores the config (= existing BUG-14).**
`_get_main_loss_function` (`:80`) **hardcodes** `BackboneRMSDLossFunction` and never reads
`config.loss_function.loss_function_type`. Selecting `"nmr"`/`"density"` in YAML silently
does nothing. (Conveniently masks BUG-P1 for the NMR/density paths — they're never constructed.)

**🔴 Cross-referenced criticals (already in §4):**

| ID | Where | Issue |
|---|---|---|
| BUG-01 | `it_optimization_manager.py` ~:301 | `checkpoint = None` hardcoded → resume branch is dead code. |
| BUG-02 | `it_optimization_manager.py:337` | **`Adam` rebuilt every diffusion step** → momentum reset to zero each step; lr never accumulates. |
| BUG-03 | `it_optimization_manager.py` | `optimized_s_inputs` never gets `requires_grad=True` → that embedding is silently frozen. |
| BUG-04 | `backbone_rmsd_loss_function.py:84` | Alignment uses all atoms but loss indexes backbone-only masks; count mismatch broadcasts wrong / errors. Also "RMSD" is a per-structure Frobenius norm, not normalized RMSD. |
| BUG-16 | `it_optimization_manager.py` (`==160`) | Intermediate embedding save hardcodes step 160; never triggers if `diffusion_N` < 160. |

**🟡 Additional smells (low severity, not in the doc):**
- **Dead code:** `noisy_structures = noisy_structures.clone().detach()` after
  `optimizer.step()` is reassigned but never used.
- **Copy-paste docstring:** `AnchorLossFunction.__init__` is documented as "RMSD loss
  function for the backbone atoms" — it's actually the embedding anchor.
- **Unused config key:** `it_optimization_parameters.save_msa_every_n_steps: 40` is set in
  `rmsd_baseline.yaml` but the save is gated on the hardcoded `==160` (BUG-16), so the key
  is dead.
- **Double denoise per step:** one forward for the loss (with grad) + a second under
  `no_grad` to advance the trajectory → ~2× denoise cost per step (see Tier 2 / #4).

### 3.7 Bottom line

- The IT-Opt **layer** is compact and cleanly separated from stock Protenix: one driver,
  one model wrapper, the `losses/` package, a config, and reference PDBs.
- **As shipped it does not run**: `from src.losses import *` fails on missing modules
  (BUG-P1). The only end-to-end-present objective is `backbone_rmsd` + `anchor`, and even
  that path carries the criticals (BUG-02 momentum reset, BUG-03 frozen `s_inputs`,
  BUG-04 mask mismatch).
- Minimum to get a working run: (1) guard/restore the missing loss imports, (2) move
  `Adam` construction outside the diffusion-step loop, (3) set
  `optimized_s_inputs.requires_grad` (or document it frozen), (4) reconcile the backbone
  masks in the RMSD loss.

---

## 4. Full bug report & next steps (all three forks)

*Source: `it_opt/BUG_AND_NEXT_STEPS.md`. Generated by deep code inspection of all three
forks; line numbers refer to files as they exist in the `it_opt` tree.*

### Severity legend

| Symbol | Meaning |
|---|---|
| 🔴 **Critical** | Produces silently wrong results or crashes under normal use |
| 🟠 **High** | Significant correctness or reliability problem, not always triggered |
| 🟡 **Medium** | Performance, maintainability, or edge-case correctness issue |
| 🟢 **Low** | Minor quality issue, naming, or hardcoded constant |

### Critical bugs 🔴

**BUG-01 · Protenix · Dead checkpoint-resume code** — `it_optimization_manager.py:301–310`.
`checkpoint = None` is hardcoded, so the `else: # Loading checkpoint` branch (including
`checkpoint["s_inputs"]`) is unreachable dead code. Any interrupted long run cannot be
resumed. **Fix:** load from an optional YAML `checkpoint_path`:
```python
checkpoint_path = self.config.it_optimization_parameters.get("checkpoint_path", None)
checkpoint = torch.load(checkpoint_path) if checkpoint_path else None
```

**BUG-02 · Boltz (both scripts) + Protenix · Adam rebuilt every diffusion step — momentum
always reset** — `boltz_embedding_iptm_optimization.py:536`,
`boltz_embedding_nmr_optimization.py:649`, and `protenix/it_optimization_manager.py:337`.
```python
if do_opt:
    optimizer = torch.optim.Adam(params, lr=cfg.lr)   # <- rebuilt every step!
```
Adam's first/second moments (`m`, `v`) are discarded every timestep, so "Adam" is
effectively zero-momentum gradient descent (raw grad ÷ flat step size). **This is the same
finding as the migration plan's Tier 1 / TL;DR.** **Fix:** construct the optimizer outside
the diffusion loop (per the migration plan, once per outer loop).

**BUG-03 · Protenix · `optimized_s_inputs` silently never receives gradients** —
`it_optimization_manager.py:303–313`. `requires_grad = True` is set on `optimized_s` and
`optimized_z` but never on `optimized_s_inputs`, though it is named/cloned as if optimized.
**Fix:** either set `requires_grad = True` and add it to clipping + optimizer, or rename it
`frozen_s_inputs` to make intent explicit.

**BUG-04 · Protenix · Backbone alignment uses all atoms; loss uses backbone-only — mask
mismatch** — `backbone_rmsd_loss_function.py:84–89`. The rigid alignment is computed over
all heavy atoms, but the loss is evaluated only on backbone atoms, so the gradient
optimizes backbone deviation relative to an all-atom alignment — a different problem than
backbone RMSD. Also `per_structure_norm` computes `||aligned - target||_F` (Frobenius),
not RMSD (true RMSD = `||...||_F / sqrt(N_atoms)`), inflating the value by
`sqrt(N_backbone_atoms)`. **Fix:** use the backbone mask for *both* alignment and loss, and
divide by `sqrt(n_backbone)`.

**BUG-05 · Boltz NMR · `KeyError` crash if `nmr_loss_fn.wandb_log()` fails** —
`boltz_embedding_nmr_optimization.py:738–748`. A `try/except: pass` swallows any exception
from `wandb_log()`, then the next lines index `log_payload["nmr/constraints_satisfied_ub"]`
which was never written → `KeyError` kills the run. **Fix:** use
`log_payload.get("nmr/...", float("nan"))`.

### High priority 🟠

**BUG-06 · AF3 · Thread-unsafe module monkey-patch; original not restored on exception** —
`embedding_optimization.py:857–867` (patch) / `:1584` (restore).
`template_modules.dgram_from_positions` is globally replaced with a soft version; if any
exception is raised before the restore, all subsequent evaluations silently use soft
binning, and concurrent calls race on the global attribute. **Fix:** `try/finally` +
`threading.Lock`.

**BUG-07 · Boltz NMR · `grad_norm()` called 10× with `retain_graph=True` — 10× backward
cost** — `boltz_embedding_nmr_optimization.py:689–698`. Ten separate
`torch.autograd.grad(..., retain_graph=True)` traversals per inner step. **Fix:** call
`.backward()` once and read `.grad` afterwards; do per-component attribution in a separate
profiling run.

**BUG-08 · Protenix · `calculate_within_chain_clash` — exponential gradient explosion** —
`it_optimization_manager.py:131–136`. `loss = (torch.relu(threshold - distances) /
0.25).exp()` — for severe clashes (`dist ≈ 0`, threshold 1.2) this is `exp(4.8) ≈ 121`;
dozens of clashes sum to tens of thousands and overwhelm the embedding signal. **Fix:**
bounded penalty, e.g. quadratic hinge `(relu(threshold - distances)**2).mean()`.

### Medium 🟡

- **BUG-09 · AF3** — `feat_batch.Batch` reconstructed every inner step
  (`embedding_optimization.py:692–693`, called `:1152`); build it once before the loop.
- **BUG-10 · AF3** — Pairformer projection param re-keying relies on brittle string search
  (`:1192–1227`); fail loudly (raise `ValueError`) if the param path isn't found while
  projection is enabled.
- **BUG-11 · AF3** — `EmbeddingOptimizationConfig` has no cross-parameter validation
  (`:105–168`): `perturbation_cliff_pct=0.0` silently disables the cliff guard;
  `low_rank=0` with `update_mode='low_rank'` gives a zero-parameter model; `score_threshold`
  shares the `cliff_reached` variable making logs ambiguous. **Fix:** add a `validate()` /
  `__post_init__` and separate the two early-stop flags.
- **BUG-12 · AF3** — `parse_scoring_type` (`:622–666`) doesn't validate/normalize weights;
  `"0.0*iptm"` is silently accepted (zero gradient, no-op). **Fix:** raise on zero total
  weight, warn if far from 1.0.
- **BUG-13 · AF3** — custom Adam re-implementation (`:773–814`) instead of `optax.adam`;
  extra code surface, and the integer `step` in bias correction means `adam_state` can't be
  passed to `jax.lax.scan`. **Fix:** use `optax.adam`.
- **BUG-14 · Protenix** — `_get_main_loss_function` (`:80–85`) hardcodes
  `BackboneRMSDLossFunction`; NMR loss can't be selected via YAML (= BUG-P2). **Fix:**
  dispatch on a `loss_type` YAML key.

### Low 🟢

- **BUG-15 · Boltz (both)** — loss checkpoint files saved to CWD, not `--out_dir`
  (`:673/675`, `:841/843`). **Fix:** save to `cfg.out_dir / ...`.
- **BUG-16 · Protenix** — magic number `160` hardcoded for intermediate embedding save
  (`:388`); never triggers if `diffusion_N` reduced. **Fix:** move to config
  `embedding_save_step` (see migration plan Tier 4 / #8).
- **BUG-17 · Boltz iPTM** — `--optimize_s` flag has undocumented silent correctness risk
  (`:155–157`, `:591–592`): some kernels leave no gradient path through `s`, so the flag
  silently produces zero gradients. **Fix:** startup gradient check that raises if `s_opt`
  gets a zero gradient after a test forward pass.

### Summary table

| ID | Fork | File | Severity | One-line |
|---|---|---|---|---|
| BUG-01 | Protenix | `it_optimization_manager.py:301` | 🔴 | Checkpoint resume is dead code — `checkpoint = None` hardcoded |
| BUG-02 | Boltz ×2 + Protenix | `*_optimization.py:536/:649`, `it_opt_mgr:337` | 🔴 | Adam rebuilt each diffusion step — momentum always reset |
| BUG-03 | Protenix | `it_optimization_manager.py:303` | 🔴 | `optimized_s_inputs` never gets `requires_grad=True` |
| BUG-04 | Protenix | `backbone_rmsd_loss_function.py:84` | 🔴 | All-atom alignment ≠ backbone-only loss; Frobenius not RMSD |
| BUG-05 | Boltz NMR | `boltz_embedding_nmr_optimization.py:747` | 🔴 | `KeyError` when `wandb_log()` exception swallowed |
| BUG-06 | AF3 | `embedding_optimization.py:867` | 🟠 | Thread-unsafe monkey-patch; not restored on exception |
| BUG-07 | Boltz NMR | `boltz_embedding_nmr_optimization.py:689` | 🟠 | 10× `grad_norm()` with `retain_graph=True` |
| BUG-08 | Protenix | `it_optimization_manager.py:134` | 🟠 | `exp(relu(...)/0.25)` clash penalty → gradient explosion |
| BUG-09 | AF3 | `embedding_optimization.py:693` | 🟡 | `feat_batch.Batch` rebuilt every inner step |
| BUG-10 | AF3 | `embedding_optimization.py:1192` | 🟡 | Pairformer projection re-keying via string search |
| BUG-11 | AF3 | `embedding_optimization.py:105` | 🟡 | No cross-parameter validation in config |
| BUG-12 | AF3 | `embedding_optimization.py:622` | 🟡 | `parse_scoring_type` accepts zero-weight scoring |
| BUG-13 | AF3 | `embedding_optimization.py:773` | 🟡 | Custom Adam instead of `optax.adam` |
| BUG-14 | Protenix | `it_optimization_manager.py:80` | 🟡 | Main loss hardcoded — NMR not selectable via YAML |
| BUG-15 | Boltz ×2 | `*_optimization.py:673/841` | 🟢 | Loss checkpoint saved to CWD not `--out_dir` |
| BUG-16 | Protenix | `it_optimization_manager.py:388` | 🟢 | Diffusion step 160 hardcoded magic number |
| BUG-17 | Boltz iPTM | `boltz_embedding_iptm_optimization.py:155` | 🟢 | `--optimize_s` silently zero-gradient on some kernels |

### Prioritized roadmap

1. **Phase 1 — correctness blockers (before trusting results):** BUG-01 (checkpoint resume),
   BUG-02 (Adam outside loop), BUG-04 (mask mismatch — rerun fold-switching after), BUG-05
   (`.get()` for NMR keys), BUG-03 (decide `s_inputs` trainable or rename).
2. **Phase 2 — performance (before large sweeps):** BUG-07 (drop 10× `grad_norm`), BUG-09
   (cache `Batch`), BUG-08 (bounded clash penalty).
3. **Phase 3 — harden infra (before sharing/publishing):** BUG-06 (`try/finally` + lock),
   BUG-10 (explicit param path or fail loudly), BUG-11 (`validate()`), BUG-12 (weight-sum
   warnings), BUG-14 (YAML-selectable loss), BUG-13 (`optax.adam`).
4. **Phase 4 — polish/reproducibility:** BUG-15 (`out_dir`), BUG-16 (config the save step),
   BUG-17 (startup gradient check).

**Longer-term:** unified environment (single conda/Docker with pinned deps across forks);
shared loss library (the NMR stack is duplicated nearly identically in
`boltz/src/boltz/experiments/nmr/` and `protenix/src/losses/`); seed reproducibility (AF3's
`optimize_embeddings` doesn't seed the JAX RNG); a tiny per-fork CI smoke test (1 outer step,
5 diffusion steps, batch 1, small protein).

---

## 5. Latent-space optimization deep-dive

*Source: `it_opt/LATENT_SPACE_OPTIMIZATION.md`. Analysis of the AF3 IT-opt strategy in
`af3/src/alphafold3/model/embedding_optimization.py` (latent method: gradient descent on the
trunk pair embedding to maximize confidence) and `coordinate_guidance.py` (baseline:
classifier-guidance drift on the denoised coordinates), and how it maps onto the
AF2 / OpenFold tradition.*

> Note: channel dims below (256 / 384 / 128) are AF2/AF3 architecture-spec values. The
> trunk/network modules are not present in the `af3` checkout, so they are not verified
> against local source.

### 5.1 The core strategy: test-time optimization

Both files implement inference-time optimization to steer the diffusion sampler toward
structures that score higher on confidence metrics (iPTM / pTM / pLDDT / ipSAE). Two
variants of the same idea at different points in the pipeline:

| File | What it optimizes | Analogy |
|---|---|---|
| `coordinate_guidance.py` | the **denoised coordinates** at each diffusion step | classifier guidance (a drift term on `x`) |
| `embedding_optimization.py` | the **trunk embeddings** (`single` / `pair` latents) | optimizing the *conditioning* itself |

The diffusion loop is **manually unrolled** (no `lax.scan`) so gradients can be injected
per step:

```
for outer_step in outer_steps:
  positions = initial_noise
  for diff_step in diffusion_steps:
    positions_noisy = augment_and_noise(positions)
    if diff_step % interval == 0:
      for opt_step in optimization_steps:
        grads = ∇_embeddings (-score)
        embeddings = adam_update(embeddings, grads)
    denoised = denoise(embeddings, positions_noisy)
    positions = euler_update(...)
```

Crucially, embeddings updated at step `t` **persist** to step `t+1` — the optimization
compounds along the trajectory rather than being a per-step correction.

### 5.2 What makes it differentiable (the key trick)

Confidence metrics depend on coordinates through **hard distance binning** (a distogram),
which has zero gradient. The fix is **soft binning** (`soft_dgram_from_positions`): replace
hard bin indicators with a product of sigmoids.

```python
soft_lower = sigmoid((dist² - lower_break) / temp)
soft_upper = sigmoid((upper_break - dist²) / temp)
dgram = soft_lower * soft_upper
```

It **monkey-patches** `template_modules.dgram_from_positions` with the soft version during
optimization and restores the hard version for final scoring (see BUG-06). Every metric is
reimplemented as a differentiable surrogate: iPTM/pTM replace the `max` over alignments with
a temperature-scaled softmax-weighted sum; ipSAE replaces the hard PAE cutoff with a soft
sigmoid mask; scoring is composable — `parse_scoring_type` parses `"0.8*iptm+0.2*ptm"` into
weighted terms.

### 5.3 The gradient target: embeddings, not coordinates

The loss (`step_loss_fn`):

```
loss = -score + anchor_λ_single·‖s - s_init‖ + anchor_λ_pair·‖z - z_init‖
```

Gradients are taken **w.r.t. the embeddings** (`argnums=0`) and flow through the entire
DiffusionHead → ConfidenceHead chain. By default it **freezes single and optimizes pair
only** (`optimize_single=False`, following FKSFold-Boltz) — `pair` carries inter-residue
geometry, the most direct structural lever.

### 5.4 Structured updates (regularization by construction)

Rather than freely perturbing the full `(N, N, C)` pair tensor, the optimized variable is a
compact `Δ` added to a frozen `z_init` (`init_update_params` / `apply_update`). Modes:

- **`full`** — perturb the whole tensor (most expressive, most prone to drift).
- **`low_rank` / `low_rank_diagonal`** — `Δ = UᵀV`; useful directions in a low-rank subspace.
- **`per_residue_bias`, `channel_scale`** — additive / scaling biases.
- **`element_scale*`** — multiplicative `z = (1+β)·z_init` (scalar → per-channel → rank-1
  residue outer product); the trunk already laid down the right basis, you just reweight it.
- **`zero_sum` / `zero_sum_channel`** — perturbations that preserve marginal sums.

All structured params are **initialized to zero** so the initial reconstruction exactly
equals `z_init`. A **`cross_chain_only`** mask restricts edits to inter-chain pair entries —
focusing on the protein–protein interface. *(This is the basis for migration plan Tier 5 /
#13.)*

### 5.5 Keeping the latent on-manifold (three guardrails)

Free optimization pushes embeddings into adversarial regions that fool the confidence head
without producing real structure. Three mechanisms fight this:

1. **Anchor regularization** — Frobenius penalty pulling embeddings toward `z_init`.
2. **Pairformer projection** (`pairformer_projection_fn`) — periodically re-run the 48-layer
   trunk Pairformer on the edited embeddings, **with `stop_gradient`**, to project them back
   onto the learned manifold. (Notable detail: param re-keying rewrites
   `__layer_stack_no_per_layer_1/` → `__layer_stack_no_per_layer/` — brittle, see BUG-10.)
3. **Early-stopping "cliffs"** — stop optimizing once the perturbation magnitude
   (`delta_z_pct` / `beta_norm`) exceeds `perturbation_cliff_pct`, or the score crosses
   `score_threshold`. *(Basis for migration plan Tier 5 / #12.)*

### 5.6 Optimizer & robustness details

- **Hand-rolled Adam** (`adam_update`) with bias correction; keys prefixed `_` are constants.
- **Gradient clipping** by global norm.
- **Float32 accumulation** even though DiffusionHead runs bfloat16 internally.
- **Noise-averaged gradients** — `optimization_batch_size > 1` averages the score over
  multiple noise realizations, via `vmap` (parallel) or sequential gradient accumulation
  (constant memory).
- **Memory management** — `jax.checkpoint` applied separately to denoise and score passes
  so peak memory is `max(diffusion_bwd, confidence_bwd)`, not the sum.

### 5.7 The coordinate-guidance baseline (for contrast)

`coordinate_guidance.py` is the simpler sibling. Same soft-binning + differentiable-metric
machinery, but: differentiates score **w.r.t. denoised coordinates** and adds
`scale · ∇score` directly (classifier-guidance drift, no Adam, no persistent state); adds
direct geometric scores that bypass the confidence head (`compute_interface_contact_score`,
`compute_pae_proxy_score`); has windowing (`guidance_start/stop_fraction`,
`guidance_interval`), optional iterative refinement, and numerical gradient checks.

### 5.8 Framing it as AF2 / OpenFold latent-space optimization

**Representation correspondence:**

| Shape | AF2 / OpenFold | AF3 (this code) | Role |
|---|---|---|---|
| `[N_s, N, 256]` | `m` — MSA representation | *— absent —* | coevolution / sequence ensemble |
| `[N, 384]` | `s` — single | `single` (s_trunk) | per-residue / per-token node feature |
| `[N, N, 128]` | `z` — pair | `pair` (z_trunk) ← **optimized** | inter-token geometry (the lever) |
| module | IPA — deterministic | Diffusion — stochastic | latent → 3D coordinates |

In AF2 the **MSA representation `m`** is the richest latent and `s` is essentially its first
row after the Evoformer. AF3 collapses this: the trunk emits only `single` + `pair`, and the
MSA is consumed inside the trunk. So the AF3 analog of "optimize the MSA/single
representation" is **optimize `single`/`pair` directly**.

**The three classic AF2/OpenFold paradigms:**
- **(a) Input/sequence-space (ColabDesign / AfDesign).** Free variable = input sequence;
  backprop through the whole Evoformer + structure module. Expensive, but on-manifold "for
  free" (always a valid sequence).
- **(b) Latent-space.** Free variable = internal `m`/`s`/`z` tensors; backprop only through
  the cheaper downstream module. Cheap and expressive, but the latent can drift adversarially.
  **← this code is here.**
- **(c) Recycling.** Gradient-free fixed-point re-embedding; cleans up representations but
  can't be steered.

**What carries over (same as AF2):** node + edge split (`single`/`pair` at ~384 / ~128);
pair is the workhorse (edit the edge tensor, freeze `single`); manifold = geometric
consistency (AF2 triangle attention ↔ AF3 Pairformer; projection re-imposes it); recycling
repurposed (AF2's gradient-free re-embedding returns as the stop-grad Pairformer projection);
same hallucination hazard and the same regularize-back-to-manifold answer.

**What's different (forced by AF3):** no MSA latent to optimize (edit distilled geometry,
not the coevolutionary ensemble — less expressive, lower-dim, more stable); the latent
conditions a *distribution* (diffusion is stochastic → optimization is unrolled across
denoising steps and noise-averaged); differentiability is hand-built (`soft_dgram` sigmoid
patch); trunk strictly frozen (gradient never re-enters the encoder); token-level,
interface-aware (`pair` spans protein/nucleic/ligand tokens; `cross_chain_only` + ipSAE
target the interface).

### 5.9 One-sentence summary

This is **AF2/OpenFold-style latent-space optimization of the single/pair representation**,
ported to AF3 by (a) backpropping differentiable confidence surrogates through the diffusion
sampler instead of IPA, (b) freezing `single` and editing `pair` as the geometric lever, and
(c) using a stop-gradient Pairformer pass as a steerable substitute for AF2 recycling to keep
the optimized latent on-manifold.

---

## 6. Per-function reference & safe-editing notes

*Source: `it_opt/EDITING_TUTORIAL.md`. Focused on Protenix (the port source); the AF3/Boltz
tables are summarized. Line numbers are `it_opt`-relative.*

### 6.1 Protenix — `it_optimization_manager.py` (the port source)

| Function | Lines | What it does | Risks / suggestions |
|---|---|---|---|
| `ITOptimizationManager.__init__` | 25–42 | Loads model, sets up losses, wandb, output dirs | `_get_main_loss_function` hardcodes `backbone_rmsd` — to switch to NMR you must edit code, not YAML (BUG-14). |
| `get_msa_features` | 102–110 | Returns `{s_inputs, s_trunk, z_trunk}` from the cached MSA run | Returns the same cached features every call — no refresh if the sequence changes. |
| `run_it_optimization` | 288–399 | 2-level loop: outer diffusion passes × one Adam step per timestep | (1) checkpoint dead code (BUG-01); (2) Adam rebuilt each step (BUG-02); (3) `optimized_s_inputs` never `requires_grad` (BUG-03); (4) magic step 160 save (BUG-16); (5) `s`/`z` clipped separately — joint clip is more principled (migration Tier 1 / #3). |
| `calculate_within_chain_clash` | 130–136 | Soft clash penalty `exp(relu(...))` | Exponential → gradient explosion for severe clashes (BUG-08); prefer a bounded penalty. |
| `run_full_diffusion_process` | 117–128 | Unguided diffusion baseline | `torch.no_grad()` diagnostics utility only; not connected to `main()`; may be stale vs. the batched version. |

### 6.2 AF3 & Boltz — summary of the per-function notes

- **AF3 `embedding_optimization.py`** — `soft_dgram_from_positions` (soft distogram bins;
  `temperature` borrowed from `ipsae_soft_temperature` — consider a dedicated knob);
  `EmbeddingOptimizationConfig` (20+ hyper-params, no cross-validation — BUG-11);
  `init_update_params`/`apply_update` (11 structured modes; `full` allocates N×N×C — prefer
  `low_rank`/`per_residue_bias` first); differentiable metric functions (iPTM softmax
  approximation diverges at low temp); `parse_scoring_type` (no weight-sum check — BUG-12);
  `clip_grads` (single global scale; low-rank `U`/`V` clipped independently); `adam_update`
  (custom, prefer `optax` — BUG-13); `optimize_embeddings` (the main function; monkey-patch
  not thread-safe — BUG-06; brittle projection re-keying — BUG-10).
- **AF3 `coordinate_guidance.py`** — `compute_interface_contact_score` (single-chain
  compactness fallback is a heuristic); `compute_pae_proxy_score` (not real PAE, not
  comparable to iPTM/pTM); `run_guided_diffusion` (guidance scale very sensitive; late
  low-noise steps can add artifacts — use `--coord_guidance_stop_fraction`).
- **Boltz `boltz_embedding_iptm_optimization.py`** — `pairformer_trunk_only` (silently skips
  templates); `denoise_at_timestep` (memory scales with batch — may OOM before clipping);
  `grad_norm` (`retain_graph=True` — BUG-07); `main()` (Adam rebuilt each step — BUG-02; no
  anchor in iPTM mode → drift; `--optimize_s` silent gradient risk — BUG-17).
- **Boltz `boltz_embedding_nmr_optimization.py`** — the NMR loss stack (NOE + RDC +
  relaxation + geometry + anchor; 4+ λ weights hard to balance — always keep
  `lambda_anchor_s/z > 0` or `z_opt` drifts to a degenerate minimum); `main()` (same
  Adam-rebuild issue; NMR CSVs use a strict column schema).

### 6.3 Safe editing workflow (applies to the port too)

1. Read the relevant README first.
2. Make **one focused change at a time** (a hyperparameter, an objective, or loop control).
3. Run a **short/small experiment** (few outer steps, small batch) before long jobs.
4. Check produced metrics and structures in the output directory.
5. Only then scale up sweep size, batch size, or runtime.

**What to avoid:** editing all forks at once unless intentionally porting a method; removing
anchor/regularization terms during early experiments (destabilizes optimization); increasing
both step counts and learning rate together on a first trial; mixing environments across
forks.

**Edit-type map (Protenix):** optimization behavior (objective weights, lr, clipping, step
counts, anchor weight, stop timestep) → mostly `pipeline_configurations/*.yaml`; add/modify a
loss term → `src/losses/` (keep outputs scalar and differentiable; wire into `total_loss`);
run orchestration → YAML config + `it_optimization_manager.py`.

---

## 7. Appendix — Protenix fork: how it works & loop map

*Source: `it_opt/protenix/README.md`. The conda/pip install wall and model-download commands
are **omitted** here (they target the standalone `guided_af3` environment, not Sampleworks'
pixi/pod setup — see the original file if you need them). Kept: the mechanism and the loop map.*

A fork of [Protenix](https://github.com/bytedance/Protenix) with an inference-time
optimization driver. Rather than retraining, it treats the trunk features
`(s_inputs, s_trunk, z_trunk)` as optimization variables and walks them through the
(differentiable) diffusion process under a configurable objective evaluated on the denoised
structure.

### Contents (Protenix fork)

| Path | Role |
|---|---|
| `it_optimization_manager.py` | The IT-opt driver. Builds the model, runs the optimization loop over the diffusion process, saves structures/metrics. Entry point for all runs. |
| `pipeline_configurations/*.yaml` | Run configs: sequence + reference PDBs, model-manager settings (recycles, diffusion steps, checkpoint path), optimizer hyperparameters, and the loss stack with weights. |
| `it_optim_inputs/` | Reference PDBs for the RMSD / fold-switching experiment (`2qke_fixed.pdb`, `5jyt_fixed.pdb`). |
| `nmr_pipeline_inputs/` | NMR target dataset used by the NMR loss (see §1). |
| `src/losses/` | Loss modules: backbone-RMSD, anchor (L2 to trunk baseline), NMR (NOE/RDC/relaxation), bond-length + clash, OpenFold-style violations. |
| `src/utils/non_diffusion_model_manager.py` | `ProtenixModelManager` — wraps the model so the trunk and each diffusion step can be called individually and differentiably. Also handles MSA querying and structure I/O. |
| `src/utils/{io,mmseqs_query,pdb_parsing}.py` | Config loading, MSA-server queries, PDB parsing helpers. |
| `src/runner/inference.py`, `src/configs/`, `src/protenix/` | Upstream Protenix inference runner, config defaults, model package. |
| `src/af3-dev/release_model/` | Protenix model checkpoint (`model_v0.2.0.pt`). |

### How it works

The manager (`ITOptimizationManager`) loads the trunk features once, then for a number of
**outer diffusion passes** it optimizes `(s_trunk, z_trunk)` with Adam against a weighted
loss. Two loss families are wired up:

- **`backbone_rmsd`** — RMSD of the diffused backbone to a target reference structure,
  optionally combined with an **anchor** term keeping the optimized embeddings close to the
  trunk baseline. Steers toward a specific conformation (e.g. a fold-switched state).
- **NMR / geometry losses** (`nmr`, `bond_length`, `violation`) — bias sampling toward
  NMR-restraint-consistent, geometrically sane structures.

Which loss is active, its weights, the learning rate, the number of outer steps, batch size,
and gradient clipping are all set in the YAML config.

### The loop map

`ITOptimizationManager.run_it_optimization()` — definition at `it_optimization_manager.py:288`:

| Loop | Code | Line |
|---|---|---|
| Outer — full diffusion passes (`outer_diffusion_steps`) | `for diffusion_loop_iteration in range(num_full_diffusion_optimization_steps):` | `:321` |
| Inner — one optimization step per diffusion timestep | `for diffusion_step in diffusion_steps:` | `:335` |

Where the optimization actually happens (inside the inner loop):

- **Adam over the embeddings** — `torch.optim.Adam([optimized_s, optimized_z], lr=learning_rate)`
  at `:337` (rebuilt each step — BUG-02), `optimizer.zero_grad()` at `:340`.
- **Differentiable forward** — `denoise_net_batched(...)` → `it_optim_x0_hat` at `:345–354`
  (the call gradients flow through).
- **Loss** — `get_loss_values(...)` → `total_loss` at `:356` (main loss + anchor terms).
- **Backprop → clip → step** — `total_loss.backward()` (`:357`) → `clip_grad_norm_`
  (`:360–361`) → `optimizer.step()` (`:367`).

The diffusion-step forward is then taken under `torch.no_grad()` at `:372–392` — that part is
*not* optimized. Unlike the AF3 and Boltz forks (which add a third, innermost
`optimization_steps` / `inner_steps` loop), Protenix takes exactly **one Adam step per
diffusion timestep**, so its loop nest is two levels deep rather than three.

### Usage (RMSD optimization example)

Computes RMSD to the ground-state reference (`2qke`). In this setup the model predicts the
fold-switched state (`5jyt`) even when the input sequence is from the ground state (`2qke`);
IT-opt steers it back toward the ground state.

```
python3 it_optimization_manager.py -- configuration_file pipeline_configurations/rmsd_baseline.yaml
```

Model weights (`model_v0.2.0.pt`) and the CCD component dictionaries are downloaded into
`src/af3-dev/release_model/` and `src/af3-dev/release_data/`; the checkpoint path is wired
into the run configs as `model_checkpoint_path` (see `pipeline_configurations/rmsd_baseline.yaml`).
