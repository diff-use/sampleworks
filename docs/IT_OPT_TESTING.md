# IT-Opt — Testing, Verification, and Wiring

How to run [`LatentOptimization`](../src/sampleworks/core/scalers/latent_optimization.py), debug it
when it misbehaves, what it produces on real targets, and where it is wired into the pipeline. For
the architecture, read [IT_OPT_DESIGN.md](IT_OPT_DESIGN.md) first.

Protenix is the primary test target (it is the model the reference algorithm was written for), so the
recipes below use it; Boltz1/RF3 need neither precondition in §1.

---

## 1. Protenix preconditions (read first)

**(a) The optimizable latent must reach the trunk.** Protenix's `step()` detaches the cached trunk
outputs when gradients are on (to avoid double-backward across the denoising steps) — correct for
coordinate guidance, but it would make a *latent* gradient **exactly zero**, silently. The
`detach_unless_leaf` helper in [models/protenix/wrapper.py](../src/sampleworks/models/protenix/wrapper.py)
handles both paths automatically: a cached latent detaches, but a latent IT-opt injected as an
optimizable leaf (`requires_grad=True`) is kept attached. No manual edit is needed — if the §2 Level-0
check fails, confirm the leaf actually has `requires_grad=True` and that `single_attr`/`pair_attr` are
`s_trunk`/`z_trunk`.

**(b) For `z` optimization, disable the diffusion shared-vars cache.** `pair_z`, `p_lm`, `c_l` are
cached from `z_trunk` at featurize time. Optimizing `z_trunk` with the cache on makes the diffusion
module read the **stale** `pair_z` — the `z` gradient is only partial and the forward is wrong. Build
the Protenix wrapper with `enable_diffusion_shared_vars_cache=False` (which `_run_guidance` does for
`LATENT_OPT`), or start single-only (`optimize_pair=False`), which the cache doesn't affect.

> Recommended first Protenix run: `optimize_single=True, optimize_pair=False` with the cache at its
> default. It exercises the whole loop without the cache subtlety; then turn on `z` with the cache off.

## 2. Debug ladder

The runtime lives on the Astera pod; edit locally, sync, run in the Protenix env. Work up the levels
and stop at the first that fails.

```bash
pixi run -e protenix-dev python your_it_opt_test.py            # needs the checkpoint + a map/structure
pixi run -e protenix-dev python -m pytest tests/models/test_latent_adapter.py -q   # CPU, no checkpoint
```

### Level 0 — Does the gradient reach the latent? (the #1 failure mode)

Needs no reward; confirms precondition (a). Inject a `requires_grad` leaf as `z_trunk` (or
`s_trunk`), run one differentiable denoiser forward, and check the leaf received a gradient:

```python
import torch
from sampleworks.utils.guidance_script_utils import get_model_and_device, get_reward_function_and_structure
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import GenerativeModelInput

device, model = get_model_and_device("cuda:0", "<protenix.ckpt>", "protenix")
reward, structure = get_reward_function_and_structure(
    density="<map.ccp4>", device=device, em=False, loss_order=2,
    resolution=1.8, structure_path="<structure.cif>",
)

feats = model.featurize(structure)                       # trunk pass
io = AttrLatentIO(single_attr="s_trunk", pair_attr="z_trunk")

z0 = io.read_pair(feats.conditioning).detach()
z_leaf = z0.clone().requires_grad_(True)
cond = io.write_pair(feats.conditioning, z_leaf)
feats2 = GenerativeModelInput(conditioning=cond)         # conditioning-only (x_init removed in #330)

sampler = AF3EDMSampler(EDMSamplerConfig(device=str(device), augmentation=False))
schedule = sampler.compute_schedule(num_steps=200)
t_hat = schedule.t_hat[100]
x = torch.as_tensor(model.initialize_from_prior(batch_size=1, features=feats2))

with torch.enable_grad():
    x0 = model.step(x, t_hat, features=feats2)
    x0.sum().backward()

print("z_leaf.grad is None:", z_leaf.grad is None)
print("z_leaf.grad abs-sum:", None if z_leaf.grad is None else z_leaf.grad.abs().sum().item())
```

**Expect** `grad is None → False`, `abs-sum → > 0`. Repeat with `read_single`/`write_single` and
`s_trunk` to check `s`.

> With the `z` cache **on**, this still shows a non-zero grad (through the direct `z_trunk` arg) but
> the forward used the stale `pair_z`. Level 0 confirms the leaf reaches the trunk, not cache
> correctness — that is why §1(b) matters for real `z` runs.

### Level 1 — A tiny end-to-end run

```python
from sampleworks.core.scalers.latent_optimization import LatentOptimization

itopt = LatentOptimization(
    ensemble_size=1, num_steps=20, outer_steps=1,   # tiny
    learning_rate=0.05, max_grad_norm=1.0,
    optimize_single=True, optimize_pair=False,      # single-only is cache-safe
    anchor_weight_single=1.0, single_attr="s_trunk", pair_attr="z_trunk",
)
out = itopt.sample(structure, model, sampler, step_scaler=None, reward=reward)
opt = out.metadata["optimization_losses"][0]        # per-step data losses, round 0
print("first→last opt loss:", opt[0], "→", opt[-1])
```

**Expect** the optimization loss trends **down** across the round and `out.final_state` has shape
`[ensemble_size, n_atoms, 3]`.

### Level 2 — Scale up

Raise `num_steps` (→200), `outer_steps` (→2–4), `ensemble_size`; turn on `z` (`optimize_pair=True`
**with the cache disabled**). Compare the final ensemble's density fit (RSCC) to an unguided baseline.

## 3. Failure → diagnosis

| Symptom | Likely cause | Fix |
|---|---|---|
| `latent.grad is None` / zero (Level 0) | Injected latent isn't a leaf reaching the trunk (§1a); or you optimized a latent the model doesn't route to. | Confirm `requires_grad=True`; use `single_attr="s_trunk"`, `pair_attr="z_trunk"` for Protenix. |
| `z` grad non-zero but structures don't respond | Stale `pair_z` cache (§1b). | `enable_diffusion_shared_vars_cache=False`, or optimize single-only first. |
| Optimization loss is flat | LR too small; anchor too strong (latents pinned); or gradient fully clipped. | Raise `learning_rate`; lower `anchor_weight_*`; raise `max_grad_norm`. Log the pre-clip grad norm. |
| Loss drops but structures degrade | Latent drifted off-manifold. | Raise `anchor_weight_*`; fewer steps/rounds; raise `bond_length_weight`. |
| NaNs | Augmentation/churn stochasticity, or bf16 latents. | `EDMSamplerConfig(augmentation=False)`; keep latents float32; lower LR. |
| `ValueError: no optimizable latent on the conditioning` | Wrong attr names (Boltz `"s"`/`"z"` on Protenix). | Use `s_trunk`/`z_trunk`. |
| `ValueError: State atom count != reward_inputs atom count` | Reward built from a different atom set than the model's. | Use the same structure/density the CLI takes. |

Cheap instrumentation (one line each in `_latent_adam_step`): the pre-clip grad norm (is the clip
biting?) and the anchor's `mean((Δs)²)`, `mean((Δz)²)` (how far the latents drifted).

## 4. CLI and wiring

`GuidanceType.LATENT_OPT` routes to `LatentOptimization` in `_run_guidance`
([utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py)):

```bash
sampleworks-guidance --model protenix --guidance-type latent_opt …
```

The same seam resolves the model name to `single_attr`/`pair_attr` (raising a clear `ValueError` for
an unsupported model) and, for Protenix, disables the diffusion shared-vars cache so the `z` gradient
survives. The direct-script route in §2 is the better *debugging* surface — it isolates the scaler
from the grid-search / save machinery.

**Production footprint** (each tagged with a greppable `IT-opt wiring` comment):
- [utils/guidance_constants.py](../src/sampleworks/utils/guidance_constants.py) — `GuidanceType.LATENT_OPT`.
- [utils/guidance_script_arguments.py](../src/sampleworks/utils/guidance_script_arguments.py) —
  `add_latent_opt_args` (`--which-latent`, `--learning-rate`, `--outer-steps`, `--anchor-weight`,
  `--max-grad-norm`, `--bond-length-weight`); the six names are also in `_DYNAMIC_ATTRS` so
  `GuidanceConfig.from_cli` copies them onto the config.
- [utils/guidance_script_utils.py](../src/sampleworks/utils/guidance_script_utils.py) — the
  `LATENT_OPT` dispatch, the Protenix cache-off decision, and the `save_trajectory` case.
- [core/scalers/latent_optimization.py](../src/sampleworks/core/scalers/latent_optimization.py) —
  per-latent grad clips and the `latent_drift` diagnostic.

## 5. Verification results

Verified against the sampleworks-release paper's altloc metrics on the already-generated 40-protein
ensembles (native occupancy — no regeneration). Conditions: `baseline` (unguided), `coord_guidance`
(the paper's shipped guided method), and `z` (IT-opt) at bond-geometry weights 0 / 5e-5 / 1e-4 / 1e-3.

| dimension (metric) | baseline | coord_guid | z (w=0) | z 5e-5 |
|---|---|---|---|---|
| density fit (RSCC ≥ 0.8) | 42% | inert | 92% | 92% |
| accuracy (nearer-altloc RMSD, med.) | 2.14 | inert | 1.02 | ~1.0 |
| reach both (RMSD max(A,B) med.; ≤2Å) | 2.18; 46% | inert | 1.3; 54% | ~1.3; 54% |
| diversity (ensembles that split A/B) | 7% | 8% | 18% | 22% |
| clean bimodal (clustering ≥ 0.5) | 10.6% | 10.6% | 8% | 9% |
| clashes (mean) | 0.38 | — | 0.47 | 0.38 |

Bond-geometry weight sweep (mean / median clash / RSCC ≥ 0.8; unguided baseline = 0.38 / 0.00 / 42%):

| weight | mean clash | median clash | RSCC ≥ 0.8 | diversity |
|---|---|---|---|---|
| 0 | 0.47 | 0.25 | 92% | 18% |
| **5e-5** | **0.38** | **0.25** | **92%** | **22%** (default) |
| 1e-4 | 0.39 | 0.25 | 92% | 17% |
| 1e-3 | 0.35 | 0.00 | 98% | 11% |

**Interpretation.** IT-opt-`z` strongly improves density fit and accuracy (RSCC ≥ 0.8 goes 42% → 92%,
mirroring the paper's *guided* 45.4% → 96.0%; nearer-altloc RMSD halves). It modestly improves
diversity (~3× more ensembles reach both altlocs, 7% → 22%) but does **not** achieve clean bimodal
capture — the clustering silhouette stays flat and ~80% of ensembles still collapse to one
conformation, consistent with the paper's thesis. `coord_guidance` is inert on every metric here. The
default bond weight `5e-5` is the smallest that restores mean clash to baseline while keeping the full
density gain and the diversity; `1e-3` over-constrains and erodes diversity. Default
`bond_length_weight` is `5e-5` (`LatentOptimization.__init__` + `--bond-length-weight`); `0` disables
the penalty.

## 6. Open problems

1. **`coord_guidance` is inert vs. the paper.** The paper reports coordinate guidance taking RSCC ≥
   0.8 from 45.4% to 96.0%; here it is indistinguishable from baseline on every metric, and it is the
   *latent* optimization that reproduces the density jump. Likely a step-size/config difference (paper
   optimal 0.1 for Protenix). Resolve with a `coord_guidance` guidance-strength sweep.
2. **RSCC uses a local scorer, not the repo-exact pipeline.** The numbers come from a homemade scorer,
   not `scripts/eval/rscc_grid_search_script.py` (`process_group` → density transformer → Kabsch →
   `extract_tight` at 2.0 Å → `rscc`). The homemade baseline (42%) matches the paper's (45.4%), which
   calibrates it, but a repo-exact run is the final confirmation. Needs a depth-4 trial-dir tree
   (`{PROTEIN}_native_occ/{model}_MD/{scaler}/ens{N}_gw{W}/refined.cif`); the generated ensembles are
   flat, so symlinks suffice.
3. **Absolute fractions sit above the paper's.** Baseline is 42% (RSCC ≥ 0.8) and 10.6% (clustering ≥
   0.5) vs the paper's 45.4% and 1.6% — RSCC matches, clustering is ~7× higher. Likely row population:
   we score native occupancy only (~85 rows), the paper aggregates the full 791-segment sweep.
   Within-our-runs comparisons are sound; cross-to-paper *absolute* fractions are not until the
   population is matched.
4. **Five proteins need patching.** 6RP1, 7Z0E, 4OLE, 8Z76, 2I6H raise "No common atoms found" (chain/
   residue-naming mismatch), so they drop from every aggregate. Resolve with
   `scripts/patch_output_cif_files.py` (needs network for `rcsb.fetch`; the `~/.sampleworks/rcsb`
   cache is empty) or sequence-based atom matching.
