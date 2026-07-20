# Testing & Debugging IT-Opt with Protenix

How to exercise [`LatentOptimization`](../src/sampleworks/core/scalers/latent_optimization.py)
on Protenix and diagnose it when it misbehaves. Protenix is a good test target
because it is the model the reference algorithm was written for — but it needs two
preconditions that Boltz1/RF3 don't.

---

## 0. Two Protenix-specific preconditions (read first)

**(a) The detach bypass must be active.** Protenix's `step()` normally detaches the
cached trunk outputs when gradients are on, which would make the latent gradient
**exactly zero** (silently — no error). The reversible test edit in
[models/protenix/wrapper.py](../src/sampleworks/models/protenix/wrapper.py) (search
`IT-OPT TEST EDIT`) keeps `s_trunk`/`z_trunk` attached. It's a commented-out/relabeled
block so you can revert to coordinate-guidance behavior later. **If you reverted it,
the gradient check in §2 will fail.**

**(b) For `z` optimization, disable the diffusion shared-vars cache.** `pair_z`,
`p_lm`, `c_l` are cached from `z_trunk` at featurize time. If you optimize `z_trunk`
but leave the cache on, the diffusion module reads the **stale** cached `pair_z` and
the `z` gradient is only partial (and the forward is wrong). Build the Protenix
wrapper with `enable_diffusion_shared_vars_cache=False`, or **start with single-only**
optimization (`optimize_pair=False`), which the cache doesn't affect.

> Recommended first Protenix run: **`optimize_single=True, optimize_pair=False`** with
> the cache at its default. It exercises the whole loop and the detach bypass without
> the cache subtlety. Then turn on `z` with the cache disabled.

---

## 1. Environment

The runtime lives on the Astera pod; source is edited locally and synced. Run in the
Protenix env:

```bash
# on the pod
pixi run -e protenix-dev python your_it_opt_test.py
# fast unit tests that don't need a checkpoint (CPU):
pixi run -e protenix-dev python -m pytest tests/models/test_latent_adapter.py -q
```

You need the Protenix checkpoint and a target density map + structure (the same
inputs `sampleworks-guidance --model protenix …` takes; see AGENTS.md for a full CLI
example).

---

## 2. Debug ladder

Work up these levels; stop at the first one that fails and fix that.

### Level 0 — Does the gradient reach the latent? (the #1 failure mode)

This is the single most important check and needs no reward. It confirms precondition
(a). Save as `it_opt_gradcheck.py`:

```python
import torch
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device, get_reward_function_and_structure,
)
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import GenerativeModelInput

device, model = get_model_and_device("cuda:0", "<protenix.ckpt>", "protenix")
reward, structure = get_reward_function_and_structure(
    density="<map.ccp4>", device=device, em=False, loss_order=2,
    resolution=1.8, structure_path="<structure.cif>",
)

feats = model.featurize(structure)                      # trunk pass
io = AttrLatentIO(single_attr="s_trunk", pair_attr="z_trunk")

# inject a requires_grad leaf as z_trunk (or s_trunk)
z0 = io.read_pair(feats.conditioning).detach()
z_leaf = z0.clone().requires_grad_(True)
cond = io.write_pair(feats.conditioning, z_leaf)
feats2 = GenerativeModelInput(x_init=feats.x_init, conditioning=cond)

# one differentiable denoiser forward at a mid-schedule noise level
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

**Expect:** `grad is None → False`, `abs-sum → > 0`. If the grad is `None`/zero, the
detach bypass is not active (precondition a) — re-apply the `IT-OPT TEST EDIT`. Repeat
with `read_single`/`write_single` and `s_trunk` to check `s`.

> Note: with the `z` cache **on**, this still shows a *non-zero* grad (through the
> direct `z_trunk` arg) but the forward used the stale `pair_z`. Level 0 confirms the
> detach bypass, not cache correctness — that's why §0(b) matters for real `z` runs.

### Level 1 — A tiny end-to-end optimization run

```python
from sampleworks.core.scalers.latent_optimization import LatentOptimization

itopt = LatentOptimization(
    ensemble_size=1,
    num_steps=20,          # tiny
    outer_steps=1,         # tiny
    learning_rate=0.05,
    max_grad_norm=1.0,
    optimize_single=True,
    optimize_pair=False,   # start single-only (cache-safe); flip on later
    anchor_weight_single=1.0,
    single_attr="s_trunk",
    pair_attr="z_trunk",
)
out = itopt.sample(structure, model, sampler, step_scaler=None, reward=reward)

opt = out.metadata["optimization_losses"][0]   # per-step data losses, round 0
print("first→last opt loss:", opt[0], "→", opt[-1])
print("final-pass losses:", [round(x, 3) for x in out.losses if x is not None][:5], "…")
```

**Expect:** the optimization loss (`optimization_losses`) trends **down** across the
round, and `out.final_state` has shape `[ensemble_size, n_atoms, 3]`. A flat loss →
see the table below.

### Level 2 — Scale up

Raise `num_steps` (→200), `outer_steps` (→2–4), `ensemble_size`, then turn on `z`
(`optimize_pair=True` **with the cache disabled**, §0b). Compare the final ensemble's
density fit (RSCC) to an unguided baseline.

---

## 3. Failure → diagnosis

| Symptom | Likely cause | Fix |
|---|---|---|
| `latent.grad is None` or zero (Level 0) | Detach bypass not active (precond a); or you optimized a latent the model doesn't route to. | Re-apply the `IT-OPT TEST EDIT`; confirm `single_attr`/`pair_attr` are `"s_trunk"`/`"z_trunk"` for Protenix. |
| `z` grad non-zero but structures don't respond / look wrong | Stale `pair_z` cache (precond b). | Build the wrapper with `enable_diffusion_shared_vars_cache=False`, or optimize single-only first. |
| Optimization loss is flat | LR too small, or anchor too strong (latents pinned to baseline), or gradient fully clipped. | Raise `learning_rate`; lower `anchor_weight_*`; raise `max_grad_norm`. Log the pre-clip grad norm. |
| Loss decreases but structures degrade (clashy/unphysical) | Latent drifted off-manifold. | Increase `anchor_weight_*`; fewer steps/rounds; add a validity term later. |
| NaNs | Augmentation + churn noise stochasticity, or bf16 latents. | Set `EDMSamplerConfig(augmentation=False)`; keep latents in float32; lower LR. |
| `ValueError: no optimizable latent on the conditioning` | Wrong attr names (used Boltz `"s"`/`"z"` on Protenix). | Use `single_attr="s_trunk", pair_attr="z_trunk"`. |
| `ValueError: State atom count != reward_inputs atom count` | Reward built from a different atom set than the model's. | Ensure the structure/density match the model's atom array (same inputs as the CLI). |

---

## 4. What "working" looks like

- **Level 0:** non-zero `s_trunk` and `z_trunk` gradients.
- **Level 1:** `optimization_losses[0]` decreases monotonically-ish across the round;
  the final-pass reward is ≤ the unguided reward.
- **Level 2:** the saved ensemble fits the density better than an unguided Protenix
  sample (higher RSCC / lower real-space loss), without gross geometry violations.

Instrument with the reference's cheap signals: log the **pre-clip** joint grad norm
each step (is the clip biting?) and the anchor's `mean((Δs)²)`, `mean((Δz)²)` (how far
the latents have drifted). Both are one line in `_latent_adam_step`.

---

## 5. Note on the current wiring

`LatentOptimization` is **not yet wired into the CLI** (`GuidanceType.LATENT_OPT` and
the `_run_guidance` branch are Phase 2). Until then, drive it from a script as above —
which is also the better debugging surface, since it isolates the scaler from the
grid-search/save machinery. Once validated, wiring it in makes it reachable as
`sampleworks-guidance --model protenix --guidance-type latent_opt …`.
