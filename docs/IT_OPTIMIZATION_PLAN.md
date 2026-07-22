# IT-Optimization — Sampleworks Implementation Plan

**Companion to** [IT_OPTIMIZATION_NOTES.md](IT_OPTIMIZATION_NOTES.md) (the consolidated
reference/migration notes) and [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) (the call graph). This
document is the *how-we-build-it-here* plan: it takes the method described in the notes and lands
it inside Sampleworks' protocol-driven architecture.

> **One-sentence summary.** Treat the frozen structure model as a differentiable sampler,
> make its cached post-trunk latents (`s_trunk`, `z_trunk`) the free variables, and run Adam on
> them against an experimental reward evaluated on the *denoised* structure — no weight updates,
> no retraining. This is AF2/OpenFold-style latent-space optimization, adapted to the AF3 family.

> **As-built note (updated).** This document is the original design plan. The implementation
> that landed is a **faithful port of the reference `run_it_optimization`** (outer resample loop +
> inner per-step `s/z` Adam + a final clean sampling pass), with the reference's bugs fixed. The
> authoritative "what was actually built and how it maps to the reference" doc is
> [IT_OPT_REFERENCE_COMPARISON.md](IT_OPT_REFERENCE_COMPARISON.md); the test/debug recipe is
> [IT_OPT_TESTING_PROTENIX.md](IT_OPT_TESTING_PROTENIX.md). Two specifics changed from the plan
> below: the anchor is a small `LatentAnchor` **co-located in the scaler** (it regularizes
> *latents*, so it can't be a coordinate `RewardFunctionProtocol`), and there is an outer
> optimization loop + final sampling pass (not a single pass).

---

## 0. The make-or-break finding (read this first)

Whether a gradient can reach the trunk latents from the denoised structure is **not uniform
across models** — it silently breaks on two of the four. This was verified against current code
(2026-07-13) and dictates the whole rollout order.

| Model | `step()` latent path | IT-opt readiness |
|---|---|---|
| **Boltz1** | passes `s_trunk=cond.s`, `z_trunk=cond.z` directly, no `.detach()`, `diffusion_conditioning=None` ([boltz/wrapper.py:1262](../src/sampleworks/models/boltz/wrapper.py#L1262)) | ✅ **clean — start here** |
| **RF3** | passes `S_trunk_I=cond.s_trunk`, `Z_trunk_II=cond.z_trunk` directly, no `.detach()` ([rf3/wrapper.py:597](../src/sampleworks/models/rf3/wrapper.py#L597)) | ✅ **clean — start here** |
| **Protenix** | `step()` **unconditionally `.detach()`s** all conditioning latents when grad is enabled ([protenix/wrapper.py:679-685](../src/sampleworks/models/protenix/wrapper.py#L679)); `pair_z/p_lm/c_l` are `z`-derived caches | 🔴 **gradient silently zeroed** — needs opt-in bypass + cache recompute |
| **Boltz2** | never passes `z`; `z` and part of `s` are baked into cached `diffusion_conditioning` (q/c/biases) at featurize ([boltz/wrapper.py:823-839](../src/sampleworks/models/boltz/wrapper.py#L823)) | 🟠 **partial/stale gradient** — needs conditioning-recompute hook |

**Consequence:** the Step-1 `latent_adapter` `training_adapter=True` path (which keeps the injected
latent in the autograd graph) is **silently defeated on Protenix and partial on Boltz2 today.** It
only learns on Boltz1/RF3. Any first version must target those two.

Two more per-model gotchas (verified):
- Protenix/RF3 run the trunk under `torch.set_grad_enabled(False)` at featurize, so cached
  `s_trunk`/`z_trunk` are `requires_grad=False` non-leaves → IT-opt must
  `clone().requires_grad_(True)` to get an optimizable leaf.
- Boltz's `_pairformer_pass` has **no** grad guard → if featurize runs under grad, cached `s/z`
  retain a graph back to the (frozen) trunk → detach before making leaves, or risk
  double-backward / memory blow-up.
- All three `Conditioning` objects are `@dataclass(frozen=True, slots=True)` → latents are swapped
  via `dataclasses.replace` (which `AttrLatentIO.write_single` already does correctly).

---

## 1. v1 architecture (chosen)

**Per-diffusion-step latent optimization**, integrated into the diffusion trajectory — not a
decoupled optimize-then-sample. At each diffusion step, after (or instead of) any coordinate
guidance, we take one Adam step on **both** `s` and `z`, then advance the trajectory with the
updated latent frozen.

Design decisions locked for v1:
- **Coordinate DPS guidance is disabled** (`scaler=None` into the sampler). The only steering in
  v1 comes from the latent update. (This is the "comment out the guidance section" request,
  realized as a clean disable so it stays reversible.)
- **Optimize `s` and `z` together**, but **clip each latent's gradient separately** (matching the
  reference). A single joint `[s, z]` clip was originally tried as a "harmonization," but since
  `z`'s gradient is ~1e4× `s`'s, the shared coefficient is set by `z` and scales `s`'s step down to
  near-nothing — starving `s`. Per-latent clips cap each step without coupling them; Adam's
  per-parameter normalization is what actually commensurates `s` and `z`. (Corrected 2026-07-20.)
- **One persistent Adam optimizer** built once for the whole diffusion pass — *not* rebuilt each
  step. Rebuilding per step is the notes' headline bug (BUG-02 / Tier 1 #1): it degenerates Adam
  to `lr·sign(g)` because the moments reset every step. We start correct.
- **Objective** = `RealSpaceRewardFunction` (density fit), evaluated on the reconciler-aligned
  denoised structure `x̂₀`, **plus an anchor** `λ_s·‖s − s₀‖² + λ_z·‖z − z₀‖²` that keeps the
  latents on the learned manifold (notes §5.5 — required, not optional).
- **Models** = Boltz1 and RF3 (verified clean). Protenix and Boltz2 come in Phase 3.

### The v1 loop

```
features = model.featurize(structure)                 # ONCE — runs the frozen trunk
s0 = read s_trunk(features); z0 = read z_trunk(features)      # baseline, detached (anchor target)
latent_s = s0.clone().requires_grad_(True)
latent_z = z0.clone().requires_grad_(True)
optimizer = Adam([latent_s, latent_z], lr)            # persistent, built ONCE
coords    = model.initialize_from_prior(ensemble_size, features)
schedule  = sampler.compute_schedule(num_steps)

for i in range(num_steps):
    context = sampler.get_context_for_step(i, schedule).with_reconciler(reconciler, ref)
    if i >= guidance_start:
        feats_i = inject(features, latent_s, latent_z)        # re-inject leaves (fresh graph each step)
        optimizer.zero_grad()
        x̂₀   = model.step(noisy(coords, context), t, feats_i) # ONE differentiable forward
        x̂₀   = reconciler.align(x̂₀, ref)                      # experimental frame — reward is frame-dependent
        loss = reward(x̂₀) + λ_s·‖latent_s − s0‖² + λ_z·‖latent_z − z0‖²
        loss.backward()
        clip_grad_norm_(latent_s, max_grad_norm); clip_grad_norm_(latent_z, max_grad_norm)  # separate
        optimizer.step()
    feats_adv = inject(features, latent_s.detach(), latent_z.detach())
    coords = sampler.step(coords, model, context, scaler=None, features=feats_adv).state   # advance, NO coord guidance
```

**Why this is sound in Sampleworks (all verified):**
- `featurize()` is called once and reused every step ([pure_guidance.py:74](../src/sampleworks/core/scalers/pure_guidance.py#L74)) — the cache seam for the trainable latent.
- `model.step()` is a real differentiable forward under `torch.set_grad_enabled` ([edm.py:426-427](../src/sampleworks/core/samplers/edm.py#L426)); the per-step gradient reward(x̂₀)→latent bypasses the coordinate state recursion entirely, so the sampler's per-step detach ([edm.py:421](../src/sampleworks/core/samplers/edm.py#L421)) does **not** block it.
- Re-injecting the leaves into a *fresh* `features` each step (a new `dataclasses.replace`) avoids the "backward through the graph a second time" hazard that would arise from reusing one featurize graph across many `.backward()` calls.

**Known v1 approximations (accept now, refine later):**
- The advance uses the sampler's own forward, so there are ~2 denoiser forwards per step (one for
  the latent gradient, one to advance). The notes' Tier 2 #4 removes the second by advancing with
  the detached `x̂₀`; deferred.
- Per-step augmentation (random SO(3)) + fresh noise make `x̂₀` stochastic across the two forwards.
  For a stable objective, run v1 with `augmentation=False` (and/or a fixed noise seed) during the
  optimization window.
- The density reward reduces over the ensemble with `.sum(0)` ([real_space_density.py:330](../src/sampleworks/core/rewards/real_space_density.py#L330)) → one joint scalar, not per-member. A shared latent across the ensemble is consistent with that; per-member latents would need a per-member reward.

### Why not the faithful Protenix port, and why not decoupled two-phase

- **Faithful port (unroll + backprop across timesteps):** impossible without a new sampler — the
  EDM sampler detaches state and re-noises every step ([edm.py:419-421](../src/sampleworks/core/samplers/edm.py#L419)), and it would cost O(steps · N²·d) memory. The v1 loop keeps the *shape* of the Protenix loop (update the latent each diffusion step) but takes a **single-forward per-step gradient**, which is what the architecture actually supports.
- **Decoupled optimize-then-sample (originally recommended):** cleaner separation, but the user
  chose per-step integration for v1. Kept as a documented future variant (§4).

---

## 2. Component inventory — what exists vs. what's net-new

| Piece | Status | Action |
|---|---|---|
| `LatentAdaptedWrapper` (single-rep injection at featurize→step) | exists ([latent_adapter.py](../src/sampleworks/models/latent_adapter.py)), **not wired in**, single-rep only, injector is a scalar affine | extend for pair `z`; add a tensor/delta injector; wire in |
| `AF3EDMSampler` (frozen diffusion solver) | exists, reused unchanged for the advance | none for v1 |
| `RealSpaceRewardFunction` (density objective) | exists, differentiable on `x̂₀` | reuse as-is |
| Anchor / L2-to-baseline regularizer | **does not exist anywhere** | net-new `core/rewards/anchor.py` |
| `LatentOptimization` trajectory scaler | **does not exist** | net-new `core/scalers/latent_optimization.py` |
| `GuidanceType.LATENT_OPT` + CLI/spine wiring | **does not exist** | add enum + `_run_guidance` branch + args |

---

## 3. Phased rollout

### Phase 0 — Injection surface (pure code, unit-testable, no GPU)
1. Extend `latent_adapter.py`: add `read_pair`/`write_pair` to `LatentIO`; a `DEFAULT_PAIR_REP_ATTR`
   map (`{boltz1/boltz2: "z", protenix/rf3: "z_trunk"}`); a pair-capable `AttrLatentIO`.
2. Add a `DeltaInjector` (zero-initialized additive delta `latent + Δ`, `Δ` the trainable leaf) —
   a scalar affine cannot represent a latent perturbation. Zero-init = exact identity at start
   (on-manifold discipline). Keep it swappable for a low-rank `Δ = UᵀV` on `z` later.
3. Fix `featurize(**kwargs)` pass-through drop ([latent_adapter.py:171](../src/sampleworks/models/latent_adapter.py#L171)).

### Phase 1 — Scaler + anchor (Boltz1/RF3)
4. `core/rewards/anchor.py`: `AnchorReward` implementing `RewardFunctionProtocol`,
   `λ·‖latent − latent₀‖²`. Separate `λ_s`, `λ_z`.
5. `core/scalers/latent_optimization.py`: the v1 loop above. Persistent Adam, joint clip, density +
   anchor objective, `scaler=None` advance. Optimization diagnostics (loss trajectory, `‖Δ‖`
   drift per step) into `GuidanceOutput.metadata`.
6. Validate on Boltz1 + RF3 with the density reward. Short run first (few steps, small protein).

### Phase 2 — CLI / spine wiring
7. `GuidanceType.LATENT_OPT` ([guidance_constants.py](../src/sampleworks/utils/guidance_constants.py)); a branch in `_run_guidance` ([guidance_script_utils.py:527](../src/sampleworks/utils/guidance_script_utils.py#L527)) that wraps the model in `LatentAdaptedWrapper(training_adapter=True)` and selects the scaler; `add_latent_opt_args` (`--learning-rate`, `--num-opt-steps`/interval, `--anchor-weight-s`, `--anchor-weight-z`, `--which-latent {single,pair,both}`, `--max-grad-norm`) + `_DYNAMIC_ATTRS`; a `save_trajectory` dispatch branch.

### Phase 3 — Enable the hard models
8. **Protenix:** gate the detach block ([protenix/wrapper.py:679-685](../src/sampleworks/models/protenix/wrapper.py#L679)) behind an `it_opt`/grad-passthrough flag; for `z`-opt, recompute `pair_z`/`p_lm`/`c_l` from the optimized `z` each step, or disable `enable_diffusion_shared_vars_cache`.
9. **Boltz2:** re-call `self.model.diffusion_conditioning(s_trunk, z_trunk, …)` ([boltz/wrapper.py:823](../src/sampleworks/models/boltz/wrapper.py#L823)) on the optimized `(s,z)` before the denoiser step so `q/c/biases` aren't stale; detach cached `s/z` before making leaves.

---

## 4. Future refinements (notes Tier 5 / method upgrades)
- **Interval optimization** with `inner_steps > 1` against a *fixed* noisy structure at optimizing
  steps only (what actually makes Adam's second moment engage; notes Tier 5 #11).
- **Early-stopping "cliffs":** freeze a batch member once its `‖z_i − z₀‖/‖z₀‖` exceeds a budget
  (notes Tier 5 #12).
- **Structured low-rank `z` delta** (`Δ = UᵀV`) to cut the O(N²·d) `z` memory and resist drift
  (notes Tier 5 #13).
- **Decoupled optimize-then-sample** as an alternative guidance mode.
- **Tier 2 single-forward advance** to remove the redundant denoiser pass.

## 5. Correctness lessons baked in from the notes' bug review
- Persistent optimizer, real Adam (BUG-02 / Tier 1 #1) — done in v1.
- Per-latent `s`/`z` clip (Tier 1 #3) — separate clips; a joint clip starved `s` (corrected 2026-07-20).
- Objective is config-selectable via `RewardFunctionProtocol`, never hardcoded (BUG-P2/BUG-14).
- If a backbone-RMSD reward is added later: use the backbone mask for *both* alignment and loss,
  and divide by `√N` for a true RMSD (BUG-04).
- Any validity/clash term uses a *bounded* penalty, not `exp(relu(...))` (BUG-08).
