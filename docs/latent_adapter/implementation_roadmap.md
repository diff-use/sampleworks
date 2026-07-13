# Implementation Roadmap — Legibility Refactor → Latent-Opt Injection → Validation

> Companion to **`latent_space_optimization.md`** (the verified *what/where* of the
> latent pre-pass: targets, `DeltaInjector`, per-model cache handling) and
> `latent_adapter.py`.
>
> This doc is the **order of operations and the scientific guardrails** around that
> design:
> 1. the legibility refactor that creates a clean insertion seam,
> 2. the latent-opt injection (deferring mechanism details to the companion doc),
> 3. the per-model migration,
> 4. the validation falsifier and the scientific-validity contract that say *when a
>    result may be believed*.
>
> Derived from a design review of `guidance_script_utils.py`, `edm.py`,
> `pure_guidance.py`, and `boltz/wrapper.py`. Confidence levels are stated inline.

---

## 0. The frame: intervention depth

Every way experimental data can change what the model generates sits on one axis —
**where the intervention acts** — running from "after the prior" to "inside the
prior." Capability to *relocate support* increases with depth; so does machinery and
risk.

| Depth | Mechanism | Relocates support? | Cost / machinery | Status in repo |
|---|---|---|---|---|
| 1 | Output selection (best-of-N / importance-weighted) | No — support-limited | Huge compute; unbiased; any reward | **Not implemented** (should be the baseline) |
| 2 | Coordinate steering (DPS / FK) | No — support-limited | Biased; cheap; needs differentiable reward | **What the code does today** |
| 3 | Latent remapping (`s_trunk` / `z_trunk`) | **Yes** | Per-model; needs manifold regularizer; inference-opt *or* learned | **This work** (see companion doc) |
| 4 | Weight fine-tuning (reward alignment) | Yes, globally | Most expensive; risks forgetting the physics prior | Future |

Two established conclusions that motivate moving to depth 3 (≈85% confidence,
pending held-out evidence):

- Depths 1–2 are **support-limited**: they steer/select within what the frozen
  trunk already generates. Coordinate guidance (depth 2) is, within support, a
  *biased variance-reduction trick* over best-of-N selection (depth 1).
- Only depth ≥3 changes **capability**, because only it edits the *information* in
  `s`/`z`. Coordinate guidance cannot inject information absent from `s_trunk`.

⇒ For the scientifically interesting regime (under-sampled / alternate states), the
work is at depth 3. This roadmap gets there safely.

---

## 1. Phase 0 — Legibility refactor (PREREQUISITE, no behavior change)

**Why first:** `guidance_script_utils.py` is ~710 lines, of which ~38% is pure
save/serialize/metadata and the scientific flow (`_run_guidance`) does not begin
until **line 427**. Injecting latent-opt into this file means surgery in the middle
of plumbing. Extracting the plumbing first turns the insertion into a one-line
change in an obvious place. This is also a standing code-style preference: keep the
scientific narrative front-and-center, mechanism in its own module.

### Steps

1. **Create `runs/outputs.py`** and move (verbatim, no logic change):
   - `save_everything` (currently L272)
   - `save_trajectory`, `_save_trajectory`, `_save_fk_steering_trajectory` (L66–150)
   - `save_losses` (L151), `_write_coords_into_array` (L83)
   - `_write_job_metadata` (L604), `get_job_result` (L639), `epoch_seconds` (L634)

2. **Create `runs/factory.py`** (or, better, fold into the wrappers later) and move:
   - `get_model_and_device` (L165)
   - the model-specific structure-prep dispatch currently inside `_run_guidance`
     (`process_structure_for_boltz` / `annotate_structure_for_*`, the
     `"Boltz" in wrapper_class_name` branches). This is the string-dispatch smell
     flagged by repo issue #192; relocating it is the first step toward killing it.

3. **Rewrite `_run_guidance` as a linear top-down narrative** in the surviving file
   (rename the file to `runs/guidance.py` if desired). Target shape (~5 readable
   steps, science visible on the first screen):

   ```python
   def _run_guidance(args, guidance_type, model_wrapper, device):
       reward, structure = load_inputs(args, device)          # inputs
       structure         = prepare_for_model(model_wrapper, structure, args)  # factory
       sampler           = build_sampler(args, device, model_wrapper)
       scaler            = build_step_scaler(args)
       guidance          = build_guidance(guidance_type, args)
       result            = guidance.sample(structure, model_wrapper, sampler, scaler, reward)
       save_everything(args, result, guidance_type)           # outputs
   ```

4. **Order rule:** entry/orchestration functions at the **top** of each file,
   helpers below or imported. (`_run_guidance` at L427-after-100-lines-of-savers is
   the anti-pattern.)

5. **Add an optional `features=None` param** to `PureGuidance.sample` and
   `FKSteering.sample`: if provided, skip the internal `model.featurize(structure)`
   (currently `pure_guidance.py` L76). Backward-compatible; this is the hook Phase 1
   uses to pass an optimized latent in.

### Acceptance criteria
- Behavior identical: the only diff is code **relocated**, not changed. Verify by
  re-running the 1VME quick-start and diffing outputs.
- `_run_guidance` reads in one screen; "what does this do scientifically" is
  answerable from the top of the file.
- No new imports cross the science↔plumbing boundary except the explicit
  `from .outputs import save_everything`.

---

## 2. Phase 1 — Latent-opt pre-pass (mechanism: see companion doc §4)

The latent design itself (targets, `DeltaInjector`, objective choices, per-model
cache handling) is finalized in **`latent_space_optimization.md` §3–4**. This
section only records *how it lands in the refactored code* and *which slice to build
first*.

### Seam (now obvious after Phase 0)
```python
features = model.featurize(structure)
if args.latent_opt:
    features = run_latent_optimization(model, features, reward, reward_inputs, cfg)  # NEW pre-pass
result = guidance.sample(structure, model=model, ..., features=features)            # unchanged sampler
```
The pre-pass is **decoupled**: trunk runs once, `delta` is optimized against a
differentiable objective, the optimized latent is detached and handed to the
**unchanged** sampler. Depth-3 (relocate) then composes with optional depth-2
(local steer) on top.

### Start slice (corrects an earlier suggestion)
- **Begin with RF3 or Boltz1, `s_trunk`, `DeltaInjector`, objective (a) aux-head**
  — per companion doc §3/§4.4, these cache nothing extra, so the optimized latent
  flows straight through `step()` with **zero model surgery**.
- **Boltz2 is the hardest case, not the first:** `s_trunk` is baked into the cached
  `diffusion_conditioning` (`boltz/wrapper.py` ~L805–826), so a boundary injection
  reaches only the `s_trunk` arg, not `q/c/biases`. It requires **recomputing
  `diffusion_conditioning`** from the perturbed `s,z`. Do it after the easy models
  prove signal.
- **Protenix is medium:** `step()` detaches conditioning latents; bypass that for
  the leaf and recompute `pair_z`/`p_lm`/`c_l` if perturbing `z`.

### Decisions to lock before coding
1. **Target:** `s_trunk` first; add `z_trunk` only when the MSA-information lever is
   explicitly needed.
2. **Objective:** (a) aux-head (sampler never called during opt) preferred; (b)
   single `model.step` `x̂₀` call as a cheap differentiable decoder if a
   coordinate-space density reward is wanted. Neither unrolls the sampler.
3. **Regularizer:** `‖delta‖` penalty + a hard clamp on `‖delta‖` (manifold guard;
   see §5).

---

## 3. Phase 2 — Per-model capability (later; the "per-model idea")

Promote the pre-pass to an **optional capability Protocol** rather than a free
function with model branches:

```python
@runtime_checkable
class LatentSteerable(Protocol):
    def optimize_latent(self, features, reward, reward_inputs, cfg) -> GenerativeModelInput: ...
```

- Each wrapper implements its own `s/z → conditioning` rule (Boltz2 recomputes
  `diffusion_conditioning`; RF3/Boltz1 pass through; Protenix recomputes its caches).
  This is the model-specific part that *cannot* be shared — and the read of
  `boltz/wrapper.py` proves it: the relationship between `(s,z)` and what the
  denoiser consumes differs per model.
- The **sampler/decoder stays shared and unchanged** (the only thing that should be
  shared — it is comparability infrastructure, not the intervention).
- A driver checks `isinstance(model, LatentSteerable)` and uses it if present; models
  without the capability simply don't offer latent-opt. No lowest-common-denominator
  abstraction, no string dispatch.

---

## 4. Phase 3 — Learnable / amortized remapping (future)

Resolves "static steering → learnable process." Requires three additions the current
architecture lacks (today the graph is severed every step at `edm.py:420`, and there
is no trainable parameter, optimizer, or training entry point):

1. **A graph-preserving / truncated-BPTT sampler mode** — if the single-step `x̂₀`
   decode proves too biased for the objective gradient.
2. **A `fit`/trainer** (optimizer + dataset loop) distinct from `sample`, calling the
   same wrapper capability and backprop-ing into its latent parameters.
3. **An amortized adapter** trained across many `(structure, data)` pairs that
   *predicts* the remap, vs. per-target inference-time optimization.

Defer until per-target latent-opt (Phase 1) shows it beats the baselines in §5.

---

## 5. Validation — the falsifier (MUST precede any claim)

No architecture matters until you can distinguish a **real recovered ensemble** from
**forward-model overfitting**. Build the judge before the thing it judges.

- **Held-out signal:** held-out reflections (R-free), and/or blind recovery of a
  *known* alternate conformation withheld from the reward.
- **Matched-compute baselines, all scored on held-out data:**
  1. best-of-N **importance-weighted** selection (depth 1),
  2. coordinate guidance (depth 2, current),
  3. latent-opt (depth 3, this work).
- **The decisive test:** *does latent-opt beat matched-compute best-of-N on held-out
  fit?* If depth-2 guidance does **not** beat depth-1 selection at equal
  forward-pass budget, the guidance layer is adding nothing but search. If depth-3
  does not beat both, the representation remap is not earning its complexity.

Report compute as forward-pass count, not wall-clock, so the comparison is fair.

---

## 6. Scientific-validity contract (caveats as enforced requirements)

These are not warnings to remember; encode them so they cannot be silently skipped.

1. **Forward-model overfitting (the real hazard).** Unregularized optimization
   against a differentiable density term will produce non-physical structures that
   fit the map. *Defense:* `‖delta‖` regularizer **and** held-out R-free (train-fit
   improves while held-out degrades = overfitting detected). The goal is to leave the
   prior's *default mode*, **not** its *valid-structure manifold*.
2. **Optimization ≠ sampling.** `argmin_delta L(decode)` is a point estimate; it
   collapses to a mode and destroys Boltzmann/population weighting. If relative
   populations matter, *sample* the latent (retain noise) rather than optimize to a
   point — or report the result as a single state, not an ensemble.
3. **Cross-model comparison is outcome-based, not operation-based.** Latent-opt is a
   *different operation* per model (different `s/z` shapes and `→conditioning`
   rules). Comparability comes only from the held-out outcome metric (§5), never from
   claiming the operations are equivalent. Do **not** assert operational equivalence
   across models.
4. **Manifold guard.** Clamp `‖delta‖` to avoid OOD / NaN `x̂₀`. Carry the guards
   from the downstream-impact diagram (companion doc §4.4).

---

## 7. Sequencing summary

```
Phase 0  Legibility refactor (no behavior change)         ← do first; unblocks the rest
Phase 1  Latent-opt pre-pass, RF3/Boltz1, s_trunk, aux-head ← companion doc has the mechanism
   ▸ build §5 baselines + held-out harness IN PARALLEL with Phase 1
Phase 2  Promote to LatentSteerable capability (per-model)
Phase 3  Differentiable-sampler mode + trainer (amortized)  ← only if Phase 1 beats baselines
```

The legibility refactor is the cheapest, highest-leverage step and a hard
prerequisite. The validation harness (§5) gates everything downstream: it is the
falsifier that tells you whether depth-3 is worth building at all.

---

## 8. Newcomer execution order (feature-first, refactor-earned)

> **This inverts §1–§7's ordering on purpose.** §0–§7 are the *architecturally*
> correct sequence (refactor → feature). As a **new team member**, the *socially*
> correct sequence is the opposite: ship the additive feature first, earn standing
> and context, and treat the shared-code refactor (Phase 0) as something you earn the
> right to propose — not your opening move (Chesterton's fence: don't move a fence
> before you know why it's there). Each step below has a concrete code anchor and its
> blast radius.

### Step 1 — Land latent-opt as a self-contained additive module *(blast radius: ~zero)*
- **Anchor (new code):** `src/sampleworks/models/latent_adapter.py` (already present,
  untracked). Mechanism is finalized in `latent_space_optimization.md` §4.1
  (`DeltaInjector`, lines 126–133) and §4.3 (decoupled pre-pass, lines 151–161).
- **Anchor (start model — RF3, easiest):** `RF3Wrapper`
  (`src/sampleworks/models/rf3/wrapper.py:211`). `step` at `:549` reads
  `S_trunk_I=cond.s_trunk` directly (`:598`) — clean passthrough, **no cache to
  recompute**. `s_trunk` is a first-class field (`:51`). Add an additive
  `optimize_latent(...)` method here; guard so other models are untouched.
- **Alt start (Boltz1, also easy):** `Boltz1Wrapper`
  (`src/sampleworks/models/boltz/wrapper.py:967`); `step` at `:1202` reads
  `s_trunk=cond.s` (`:1260`), no extra cache.
- **Do NOT start with Boltz2** — `step` at `:846` consumes a *cached*
  `diffusion_conditioning` baked at `:820`; the perturbation is partially ignored
  unless recomputed. Hardest case, last.

### Step 1 (extended) — Protenix slice: two blockers, one additive fix
Protenix is the only model with **both** failure modes (verified against
`src/sampleworks/models/protenix/wrapper.py`). When you extend past RF3/Boltz1:

- **Blocker 1 — `step` detaches the latent under grad** (`protenix/wrapper.py:672–681`):
  a deliberate DPS optimization (grad flows only to coordinates, not back through the
  no-grad pairformer). Fatal here — a `requires_grad` `s_trunk` leaf is detached at
  `:677` on the first op. *RF3/Boltz don't do this* (RF3 passes the leaf straight
  through at `rf3/wrapper.py:598`).
- **Blocker 2 — `z` lives in derived caches** (`:600–624`): `pair_z`/`p_lm`/`c_l` are
  precomputed from `z` via `prepare_cache`; `z_trunk` reaches the diffusion module
  *mostly through them*, not the `z_trunk` arg. Direct analog of Boltz2's
  `diffusion_conditioning` baking.
- **Mechanical detail:** `ProtenixConditioning` is
  `@dataclass(frozen=True, slots=True)` (`:43`) — write with
  `dataclasses.replace(cond, s_trunk=...)`, not in place (Boltz's is mutable).

**`s_trunk`-only is still the easy first slice on Protenix:** `s_trunk` goes
*directly* to the diffusion module (`:688`) and is **not** in the `pair_z` cache, so
only Blocker 1 applies — **no cache recompute needed**.

**Additive fix (newcomer-safe — no edit to the shared `step`):** in your decoupled
pre-pass, call the diffusion module directly, mirroring `step` (`:683–693`) *minus*
the detach:
```python
x0 = self.model.diffusion_module.forward(
    x_noisy=x_t, t_hat_noise_level=t,
    input_feature_dict=cond.features,
    s_inputs=cond.s_inputs.detach(),                    # keep frozen
    s_trunk=s0 + delta,                                  # ← un-detached leaf (the variable)
    z_trunk=cond.z_trunk.detach(),
    pair_z=cond.pair_z, p_lm=cond.p_lm, c_l=cond.c_l,    # s-only: caches stay valid
)
```
This bypasses the detach for exactly the tensor you optimize, touches no shared code,
and is Protenix's equivalent of RF3's "just pass the leaf."

**`z_trunk` later (two options):**
1. Recompute caches in your decode:
   `pair_z = self.model.diffusion_module.diffusion_conditioning.prepare_cache(relp, z0+dz, ...)`,
   then `p_lm/c_l = ...atom_attention_encoder.prepare_cache(..., pair_z, ...)` so grad
   flows `z → pair_z → diffusion` (Boltz2-style recompute).
2. **Hypothesis to test (~70%):** set `enable_diffusion_shared_vars_cache=False`
   (`ProtenixConfig` at `:84` → drives `:600`); caches become `None` (`:626`) and — *if*
   `diffusion_module.forward` recomputes conditioning from `z_trunk` when they're
   `None` — the `z_trunk` gradient flows automatically (per-call recompute cost).
   Verify the `None` path actually recomputes before relying on it.

**Per-model start order:** RF3/Boltz1 (pass the leaf) → Boltz2 (recompute
`diffusion_conditioning`) → Protenix (bypass detach via direct `forward`; recompute
caches for `z`).

### Step 2 — The one acceptable edit to shared code *(blast radius: one optional param × 2 methods)*
- **Anchor:** `PureGuidance.sample` at `src/sampleworks/core/scalers/pure_guidance.py:46`;
  it calls `features = model.featurize(structure)` at `:74`. Add `features=None`; if
  provided, skip the re-featurize. Mirror in `FKSteering.sample`
  (`src/sampleworks/core/scalers/fk_steering.py:67`, featurize at `:99`).
- **Framing for review:** "I need to hand the sampler a pre-optimized latent" —
  backward-compatible, obviously feature-justified. If you want *zero* shared edits in
  PR #1, wrap/subclass instead and propose this param afterward.

### Step 3 — Build the validation harness *(blast radius: new files only)*
- **Anchor (reward to reuse):** `src/sampleworks/core/rewards/real_space_density.py`.
- **Anchor (baseline + held-out):** add a new script under `scripts/eval/` mirroring
  the sweep pattern in `run_grid_search.py`; compare best-of-N IW selection vs.
  coordinate guidance vs. latent-opt on held-out R-free. Align with existing
  `scripts/eval/EVALUATION.md` and `src/sampleworks/eval/`.
- **Why now:** additive, touches no one's architecture, earns *scientific*
  credibility — the §5 falsifier that justifies everything later.

### Step 4 — Boy-scout only your own path *(blast radius: a local helper in a file you already touch)*
- **Anchor:** the `diffusion_conditioning` call inside `_pairformer_pass`
  (`src/sampleworks/models/boltz/wrapper.py:820`, within the method at `:747`). When
  you reach the Boltz2 slice, factor out a `_compute_diffusion_conditioning(s, z, ...)`
  helper that **both** `featurize` and your `optimize_latent` call. Scoped to exactly
  the code your feature forces you to touch — no crusade.

### Step 5 — Keep a private friction log *(blast radius: none — observe, don't edit)*
Record each pain point with its concrete cost; this converts intuition into evidence.
- **String dispatch:** `src/sampleworks/utils/guidance_script_utils.py:450–480`
  (the `if "Protenix"/"RF3"/"Boltz" in wrapper_class_name` chain).
- **Buried entry point:** `_run_guidance` at `guidance_script_utils.py:427`.
- **Scattered config contract:** the `getattr(args, ...)` defaults throughout
  `guidance_script_utils.py` and `src/sampleworks/utils/guidance_script_arguments.py`.

### Step 6 — Raise improvements as questions, aligned to debt the team already owns
- **Anchor:** the existing TODO at `guidance_script_utils.py:443`
  (`# See https://github.com/diff-use/sampleworks/issues/192 ...`). The team *already
  agrees* the dispatch is debt — align with #192 rather than your own opinion.
- **Anchor (the welcomed extension):** propose your per-model latent work as an
  optional capability on the Protocol at `src/sampleworks/models/protocol.py:99`
  (`FlowModelWrapper`). The README explicitly invites "new ModelWrappers" and
  "differentiable modules for new data modalities" — your Phase 2 `LatentSteerable`
  is exactly that, so it lands as *welcomed addition*, not *unsolicited refactor*.
- **Social anchor:** the author (`pyproject.toml` → Karson). Ask before touching
  shared code: "I'm adding latent-opt; is there history behind the saving/dispatch
  before I'd consider factoring any out?"

### Step 7 — The Phase 0 legibility refactor: earned, deferred
- **Anchor:** the ~270 plumbing lines in `guidance_script_utils.py` (savers at
  `:66–163`, `save_everything` `:272`, `_write_job_metadata` `:604`, `get_job_result`
  `:639`) → `runs/outputs.py`.
- **Precondition:** standing earned (Steps 1–3 shipped) + a populated friction log
  (Step 5) + author buy-in (Step 6). Then land it in small, behavior-preserving
  slices — never as a week-one rewrite.

**One-line strategy:** *additive feature + validation first (Steps 1–3, your
mandate, ~zero blast radius) → observe and align (Steps 4–6) → propose the shared
refactor only once earned (Step 7).*
