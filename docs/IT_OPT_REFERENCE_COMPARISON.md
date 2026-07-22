# IT-Opt: My Port vs. the Reference `run_it_optimization`

How [`core/scalers/latent_optimization.py`](../src/sampleworks/core/scalers/latent_optimization.py)
maps to the reference driver
`it_opt/protenix/it_optimization_manager.py::run_it_optimization` (lines 288–399).
The goal was to **keep the algorithm** and **fix the reference's bugs** — not to
redesign it.

> Reference paths/lines below are in the external `it_opt/` tree, not this repo.

---

## 1. The three pieces you flagged

### (1) Extracting the embeddings `s` and `z`

| Reference | My port |
|---|---|
| `get_msa_features()` (`:102–110`) reads `{s_inputs, s_trunk, z_trunk}` from `self.model_manager.msa`, produced once by the trunk (`non_diffusion_model_manager.py:282–292`, `pairformer_cycle` `:305–354`). | `LatentOptimization._leaf_latents()` reads `s`/`z` off `features.conditioning` via `AttrLatentIO`, after a single `model.featurize(structure)` run under `torch.no_grad()`. The wrapper's `featurize` *is* the trunk pass; the cached `s`/`z` live on the conditioning dataclass. |
| `init_s_trunk = ....clone().detach()`, `init_z_pair = ....clone().detach()` (`:296–298`) as the anchor baseline; `optimized_s/z = init.expand(batch).clone().detach()` then `.requires_grad = True` (`:303–313`). | `baseline = read(conditioning).detach()`; `leaf = baseline.clone().requires_grad_(True)`. Baselines feed the anchor; leaves are the optimized variables. |
| `s_inputs` is **not** optimized (never gets `requires_grad` — reference BUG-03, effectively frozen). | Not touched — Sampleworks conditioning exposes only `s`/`z` as the optimization levers; `s_inputs` (Protenix-only) stays frozen, matching intent. |

**Difference:** the reference **expands the latents to the batch** (each ensemble
member gets its own `s`/`z`). My port keeps whatever shape the wrapper caches
(shared across the ensemble, broadcast by the model). Per-member latents (for
ensemble diversity) are a follow-up that first needs each wrapper's diffusion
module verified to accept a batched conditioning.

### (2) The loss

| Reference | My port |
|---|---|
| `get_loss_values()` (`:254–286`): `total_loss = main_loss + adversarial_loss_s + adversarial_loss_z`. | `loss = data_loss + anchor(latents, baselines)` in `_latent_adam_step`. |
| `main_loss = BackboneRMSDLossFunction(x0_hat)` — Kabsch-align to a target PDB, per-structure Frobenius norm of backbone deviation, mean over batch (`backbone_rmsd_loss_function.py:79–92`). | `data_loss = reward(coordinates=denoised, …)` — the pluggable `RewardFunctionProtocol`. v1 uses `RealSpaceRewardFunction` (density fit), your chosen objective. Alignment is handled by the sampler (reconciler), so the reward sees a frame-aligned `x0_hat`. |
| `AnchorLossFunction` (`anchor_loss_function.py`): `λ_s·‖s−s₀‖ + λ_z·‖z−z₀‖`, per-sample **Frobenius norm**, mean over batch. | `LatentAnchor`: `Σ w_i·mean((latentᵢ−baselineᵢ)²)` — **mean-squared** rather than norm. |

**Difference (deliberate):** the anchor uses mean-squared deviation, not the
reference's Frobenius norm. Mean-squared is shape-agnostic and normalizes for the
very different element counts of `s` vs `z`, so `w_s`/`w_z` are comparable — part
of the harmonization. Retune weights per objective (the reference's `λ` values
don't transfer, since the main objective differs anyway).

### (3) Back-propagating the gradient to `z` (and `s`)

| Reference (`:335–367`) | My port (`_optimize_one_round` + `_latent_adam_step`) |
|---|---|
| `optimizer = Adam([optimized_s, optimized_z], lr)` **inside the diffusion-step loop** (`:337`). | Built **once per optimization round**, outside the step loop. |
| `noisy = structures.clone().detach()`; `x0_hat = denoise_net_batched(noisy, t_hat, s_inputs, optimized_s, optimized_z)` (`:345–354`). | `sampler.step(coords, model, context, scaler=_GradEnablingScaler(), features=…)` runs the denoiser under autograd and returns the aligned `step_output.denoised` (= differentiable `x0_hat`). |
| `total_loss.backward()` (`:357`). | `loss.backward()`. |
| `clip_grad_norm_([optimized_s]); clip_grad_norm_([optimized_z])` — **separate** (`:360–361`). | `clip_grad_norm_(latent)` per latent — **separate**, matching the reference (an earlier joint `[s,z]` clip starved `s`; reverted 2026-07-20). |
| `optimizer.step()` (`:367`). | `optimizer.step()`. |
| Advance under `no_grad` with a **second** forward `get_x_0_hat_from_x_noisy_batched(...)` then `get_x_t_from_x_0_hat` + `get_x_noisy` (`:373–395`). | Advance with `coords = step_output.state.detach()` from the **same** sampler step — one forward, not two. |

---

## 2. Loop structure

```
REFERENCE run_it_optimization                    MY LatentOptimization.sample
─────────────────────────────                    ────────────────────────────
extract s,z; leaves; requires_grad               extract s,z; leaves (requires_grad)
for outer in range(outer_diffusion_steps):       for outer in range(outer_steps):
    structures = get_initial_latents()               coords = initialize_from_prior()
    (optimizer rebuilt PER STEP — bug)               optimizer = Adam([s,z])  # per round
    for step in range(diffusion_N):                  for i in range(num_steps):
        x0 = denoise(noisy, s, z)   # grad             x0 = sampler.step(grad_enabler).denoised
        loss = main + anchor_s + anchor_z              loss = reward(x0) + anchor(s,z)
        backward; clip s; clip z; step                 backward; clip s; clip z; step
        advance (2nd forward, no_grad)                 coords = step_output.state.detach()
return optimized_dict                             # then: final clean sampling pass
(separately) run_diffusion_process_it_optimized   _sample_with_frozen_latents() -> ensemble
```

The reference's final sampling (`run_diffusion_process_it_optimized`, `:222–240`)
is folded into `sample()` as `_sample_with_frozen_latents`, so one call returns the
saved ensemble.

---

## 3. Kept vs. changed — summary

**Kept (the algorithm):**
- `s_trunk` and `z_trunk` as the optimized variables; `s_inputs` frozen.
- One Adam step per diffusion step, reward on the denoised `x0_hat`, latents
  persisting across steps and rounds.
- Outer resample loop (fresh noise per round); anchor prior to the trunk baseline.
- A final clean sampling pass with the frozen optimized latents produces outputs.

**Fixed (the reference's bugs — from the migration notes):**
- **BUG-02 / Tier-1 #1:** optimizer built once per round, not rebuilt every step
  (the reference's "not actually running Adam" headline).
- **Tier-1 #3 (tried, then REVERTED 2026-07-20):** a single joint `[s, z]` clip was
  tried as a "harmonization," but since `z`'s gradient is ~1e4× `s`'s, the shared
  coefficient is set by `z` and scales `s`'s step down — starving `s`. Reverted to the
  reference's **separate** per-latent clips, which decouple `s`'s step from `z`'s scale.
  (Adam's per-parameter normalization is what actually commensurates the two.)
- **Tier-2 #4:** the trajectory advances using the same differentiable forward — no
  redundant second denoiser pass per step.
- **BUG-16:** no hardcoded `== 160` save step.

**Deliberate deviations (documented):**
- Objective is `RealSpaceRewardFunction` (density), not backbone-RMSD — your choice;
  the loss is pluggable via `RewardFunctionProtocol`.
- Anchor is mean-squared, not Frobenius norm (shape-agnostic; harmonizes s/z scale).
- Shared (not per-member) latents in v1.
- Advance carries a one-step embedding lag (from fusing the two forwards); over a
  full schedule + a final frozen pass this is immaterial. The reference advanced with
  the just-updated embeddings.

**Not ported (out of scope for pure latent optimization):**
- Coordinate guidance / `guidance_direction` (dormant in the reference too).
- NMR / bond-length / violation losses; wandb; checkpoint resume; the within-chain
  clash penalty (reference BUG-08, `exp(relu(...))` gradient explosion).
