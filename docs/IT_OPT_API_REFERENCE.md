# IT-Opt API Reference — signatures & contracts

Companion to [IT_OPT_API_PLAN.txt](IT_OPT_API_PLAN.txt) (the overall architecture).
This document holds the dense material — verbatim signatures, field lists, and
per-method contracts — so the architecture plan can stay readable. It is split into
**the API the feature owns** and **the seams the feature consumes**.

Signatures are quoted from source as of this writing; treat the source as
authoritative if they diverge.

---

## Part 1 — API the feature owns

### 1.1 `LatentOptimization` — the entry point
`src/sampleworks/core/scalers/latent_optimization.py`

A `TrajectoryScalerProtocol`. Peer of `PureGuidance` / `FKSteering`.

```python
def __init__(
    self,
    ensemble_size: int = 1,
    num_steps: int = 200,
    guidance_t_start: float = 0.0,
    *,
    outer_steps: int = 1,
    learning_rate: float = 0.05,
    max_grad_norm: float = 1.0,
    optimize_single: bool = True,
    optimize_pair: bool = True,
    anchor_weight_single: float = 0.0,
    anchor_weight_pair: float = 0.0,
    bond_length_weight: float = 0.0,
    single_attr: str = "s",
    pair_attr: str = "z",
):
```

| Parameter | Meaning / contract |
|---|---|
| `ensemble_size` | Structures sampled in parallel; they **share** the latents. |
| `num_steps` | Diffusion steps per pass. |
| `guidance_t_start` | Fraction in `[0,1]`; stored as `self.guidance_start = int(guidance_t_start * num_steps)`. Steps before it are plain frozen-latent diffusion. |
| `outer_steps` | Resample rounds (fresh prior noise each). **Constructor default is `1`; the CLI default is `2`.** |
| `learning_rate` | Adam LR. One persistent Adam is built **once per round**. |
| `max_grad_norm` | Per-latent grad-clip threshold; `s` and `z` are clipped **independently**. |
| `optimize_single`, `optimize_pair` | Which representations are optimized. |
| `anchor_weight_single`, `anchor_weight_pair` | Per-latent L2-to-baseline anchor weights. |
| `bond_length_weight` | Weight of the coordinate-space bond/clash penalty; `0` disables it. |
| `single_attr`, `pair_attr` | Conditioning attribute names (`"s"`/`"z"` Boltz; `"s_trunk"`/`"z_trunk"` Protenix/RF3). |

All params after `guidance_t_start` are keyword-only. `single_attr`/`pair_attr` default
to Boltz names but the wiring always passes explicit, model-resolved names.

```python
def sample(
    self,
    structure: dict,
    model: FlowModelWrapper,
    sampler: TrajectorySampler,
    step_scaler: StepScalerProtocol,
    reward: RewardFunctionProtocol,
    num_particles=1,
) -> GuidanceOutput:
```

Contract:
- Matches `TrajectoryScalerProtocol.sample`. **`step_scaler` is ignored** (v1 does no
  coordinate guidance) but kept for conformance.
- Returns a `GuidanceOutput` whose `metadata` carries `"optimization_losses"`
  (per-round list of per-step data losses) and `"latent_drift"` (per-round relative
  L2 drift of each latent from baseline). Adds `"model_atom_array"` **only when**
  `reconciler.has_mismatch and processed.model_atom_array is not None`.
- Note: siblings emit `"trajectory_denoised"`; IT-opt does not. Downstream code must
  treat scaler-specific metadata keys as optional.

Private helpers (not public API, listed for orientation): `_leaf_latents` (promote
s/z to leaves), `_optimize_one_round` (one outer round), `_latent_adam_step` (score +
backward + clip + step), `_sample_with_frozen_latents` (final clean pass).

### 1.2 `LatentAnchor`
`core/scalers/latent_optimization.py`

```python
def __init__(self, weights: Sequence[float]): ...
def __call__(self, latents: Sequence[Tensor], baselines: Sequence[Tensor]) -> Tensor:
    # sum_i  w_i * mean((latent_i - baseline_i) ** 2)
```

Mean-squared (not the reference's Frobenius norm), so weights for `s` and `z` land on a
comparable scale. **Not** a `RewardFunctionProtocol` — it regularizes latents, so it is
co-located in the scaler and added to the loss directly.

### 1.3 `_GradEnablingScaler`
`core/scalers/latent_optimization.py`

```python
class _GradEnablingScaler:
    requires_gradients = True
    def scale(self, state, context, *, model=None) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(state), torch.zeros(state.shape[0], device=state.device)
    def guidance_strength(self, context) -> Tensor:
        return torch.zeros_like(context.t_effective)
```

The minimal `StepScalerProtocol` that only turns autograd on. Because
`AF3EDMSampler.step` reads `getattr(scaler, "requires_gradients", False)`, attaching
this makes the denoiser run under grad and the returned `denoised` carry a graph to the
latent leaves — while the zero direction keeps the trajectory advance unguided.

### 1.4 Latent adapter
`src/sampleworks/models/latent_adapter.py`

**`LatentIO` (protocol)** — the minimal cross-model surface:

```python
def read_single(self, conditioning) -> Tensor | None
def write_single(self, conditioning, single: Tensor)      # returns a COPY of conditioning
def read_pair(self, conditioning)   -> Tensor | None
def write_pair(self, conditioning, pair: Tensor)          # returns a COPY of conditioning
```

**`AttrLatentIO(single_attr: str, pair_attr: str | None = None)`** — the implementation.
Reads via `getattr`; writes via `dataclasses.replace` (honors `frozen=True, slots=True`
conditioning and preserves sidecar state). `pair_attr=None` ⇒ pair accessors are no-ops
(single-rep-only mode). This is the object `LatentOptimization` uses (Path 1).

**Archived — the injector / decorator family** (`LatentInjector`, `DeltaInjector`,
`AffineInjector`, `LatentAdaptedWrapper`). An alternative "Path 2" that trained a transform
(`k*latent + b` affine, or `latent + delta`) at the `featurize` boundary via a decorator. Never
wired in; superseded by the direct-leaf `LatentOptimization` scaler (Path 1 above) and removed from
the build. Code + reasoning preserved in
[latent_adapter/archived_injector_family.md](latent_adapter/archived_injector_family.md).

**Per-model attribute maps** (documentation/config only — no model is imported):

```python
DEFAULT_SINGLE_REP_ATTR = {"boltz1": "s", "boltz2": "s", "protenix": "s_trunk", "rf3": "s_trunk"}
DEFAULT_PAIR_REP_ATTR   = {"boltz1": "z", "boltz2": "z", "protenix": "z_trunk", "rf3": "z_trunk"}
```

### 1.5 `BondGeometryReward`
`src/sampleworks/core/rewards/geometry.py`

```python
def __init__(self, atom_array: AtomArray, weight: float, device, *,
             bond_tolerance: float = 0.2, clash_padding: float = 0.4, bond_power: int = 2): ...
def __call__(self, coords: Tensor) -> Tensor      # weight * (bond_length_loss + collision_loss)
```

Contract:
- **Not** a `RewardFunctionProtocol` — `__call__` takes only `coords`
  `[ensemble, n_atoms, 3]`. Added as a **separate** term in the loss, not dispatched as
  the reward.
- Built inside `sample()` **only when** `bond_length_weight > 0`, from
  `processed.model_atom_array or processed.atom_array` so bond indices match the denoised
  coordinate ordering. Topology (bonds, ideal lengths, collision matrix, scored-pair
  mask) is computed once at construction. `collision_loss` is `O(n_atoms^2)` in memory.
- Bounded hinges (clamped positive part), not the reference's exploding `exp(relu(...))`.

---

## Part 2 — Seams the feature consumes

These are contracts IT-opt must honor; it does not own them.

### 2.1 Model — `src/sampleworks/models/protocol.py`

**`GenerativeModelInput`** — plain `@dataclass` (not frozen), generic over `C`:

```python
@dataclass
class GenerativeModelInput(Generic[C]):
    x_init: Float[Array, "*batch atoms 3"]
    conditioning: C | None
```

`conditioning` is the model-specific bundle — in practice a `@dataclass(frozen=True,
slots=True)` carrying `s`/`z` (or `s_trunk`/`z_trunk`) as named attributes. IT-opt
rebuilds the whole `GenerativeModelInput` to swap latents:
`GenerativeModelInput(x_init=features.x_init, conditioning=<rewritten>)`.

**`FlowModelWrapper` (protocol)** — the three methods IT-opt uses:

```python
def featurize(self, structure: dict) -> GenerativeModelInput[C]
def step(self, x_t, t: Float[Array, "*batch"], *, features: GenerativeModelInput[C] | None = None)
def initialize_from_prior(self, batch_size: int, features=None, *, shape=None)
```

IT-opt calls `featurize` once (under `no_grad`) and `initialize_from_prior` per round /
final pass. It does **not** call `model.step` directly — the sampler drives stepping.
(The archived `LatentAdaptedWrapper` was the mirror image — it *implemented* this protocol as a
decorator; see [the archive](latent_adapter/archived_injector_family.md).)

### 2.2 Sampler — `src/sampleworks/core/samplers/protocol.py`, `edm.py`

**`StepParams`** `@dataclass(frozen=True, slots=True)` — per-step context. Fields include
`step_index`, `total_steps`, `t`, `dt`, `noise_scale`, `reward`, `reward_inputs`,
`reconciler`, `alignment_reference`, `metadata`. Builder methods return shallow copies:

```python
def with_reward(self, reward, reward_inputs) -> StepParams
def with_reconciler(self, reconciler, alignment_reference=None) -> StepParams
def with_metadata(self, metadata) -> StepParams
```

`t_effective` returns `t` or raises if `t is None`.

**`TrajectorySampler` (protocol)**:

```python
def compute_schedule(self, num_steps: int) -> SamplerSchedule
def get_context_for_step(self, step_index: int, schedule) -> StepParams   # populates t, dt, noise_scale
def step(self, state, model_wrapper, context, *, scaler=None, features=None) -> SamplerStepOutput
```

**`SamplerStepOutput`** `@dataclass(frozen=True, slots=True)`, generic over `StateT`:

```python
state: StateT                 # coordinates after the step
denoised: StateT | None       # x_hat_0 prediction (what IT-opt scores)
loss: Float[Array, " batch"] | None
log_proposal_correction: Float[Array, " batch"] | None   # FK resampling only
```

**The gradient gate** (`AF3EDMSampler.step`, `edm.py`): the load-bearing detail.

```python
allow_gradients = True if scaler and getattr(scaler, "requires_gradients", False) else False
...
noisy_state = torch.as_tensor(noisy_state).detach().requires_grad_(allow_gradients)
...
with torch.set_grad_enabled(allow_gradients):
    x_hat_0 = model_wrapper.step(noisy_state, t_hat, features=features)
```

`requires_gradients` is duck-typed (read via `getattr`, not declared on
`StepScalerProtocol`). This is the *only* way `step()` runs the denoiser under autograd.
The returned `state`/`denoised` are **not** re-detached inside `step()` — the caller
detaches between iterations.

### 2.3 Reward — `src/sampleworks/core/rewards/protocol.py`

**`RewardFunctionProtocol.__call__`**:

```python
def __call__(
    self,
    coordinates: Float[Tensor, "batch n_atoms 3"],
    elements: Int[Tensor, "batch n_atoms"],
    b_factors: Float[Tensor, "batch n_atoms"],
    occupancies: Float[Tensor, "batch n_atoms"],
    unique_combinations: Tensor | None = None,
    inverse_indices: Tensor | None = None,
) -> Float[Tensor, ""]:      # 0-dim scalar; differentiable w.r.t. coordinates
```

IT-opt calls it with exactly the four keyword args `coordinates=<denoised>, elements=,
b_factors=, occupancies=` and never the optional vmap args. `coordinates` is the freshly
denoised `x_hat_0`, not `reward_inputs.input_coords`.

**`RealSpaceRewardFunction`** (the v1 objective) sums rendered density over the ensemble
(axis 0) into one map, then returns MSE/L1 vs the target. Occupancies are pre-scaled to
`1/ensemble_size`, so the sum is an ensemble average — consistent with shared latents.

**`RewardInputs`** `@dataclass` — `elements`, `b_factors`, `occupancies`, `input_coords`.
IT-opt reads the first three and forwards them to every reward call.

### 2.4 Structure processing — `src/sampleworks/eval/structure_utils.py`

```python
def process_structure_to_trajectory_input(
    structure: dict, coords_from_prior: torch.Tensor,
    features: GenerativeModelInput, ensemble_size: int,
) -> SampleworksProcessedStructure
```

Returns a frozen bundle. Fields IT-opt uses:

| Field / method | Use |
|---|---|
| `reconciler` | `AtomReconciler`; moved to device via `.to(...)`, threaded into `context.with_reconciler`. |
| `input_coords` | Model-space reference coords `[ensemble, n_model, 3]`; passed as `alignment_reference`. |
| `to_reward_inputs(device=...)` | Builds the `RewardInputs`; prefers the model atom array, copies structure B-factors onto common atoms. |
| `model_atom_array` | Geometry topology for `BondGeometryReward` (`or atom_array`); attached to metadata on mismatch. |
| `atom_array` | Fallback geometry/reward atom set. |

**`AtomReconciler`** `@dataclass(frozen=True, slots=True)` — bidirectional index adapter
between model-space and structure-space atoms. Key surface: `.has_mismatch`, `.to(device)`,
`.align(model_coords, model_reference, allow_gradients=False, ...)`. IT-opt does **not**
call `align` directly — it hands the reconciler + `alignment_reference` to the sampler,
which aligns each step's `denoised` to the reference before reward evaluation. Notes:
`process_structure_to_trajectory_input` **mutates `structure['asym_unit']` in place**, and
`coords_from_prior` only donates dtype/device.

---

## Part 3 — Wiring & known gaps

### 3.1 Construction site — `_run_guidance`
`src/sampleworks/utils/guidance_script_utils.py` (the `guidance_type == GuidanceType.LATENT_OPT` branch)

```python
which_latent = getattr(args, "which_latent", "pair")
anchor_weight = getattr(args, "anchor_weight", 0.0)
model_key = str(args.model)
LatentOptimization(
    ensemble_size=args.ensemble_size,
    num_steps=num_steps,                       # = args.num_diffusion_steps
    guidance_t_start=args.guidance_start / num_steps if args.guidance_start > 0 else 0.0,
    outer_steps=getattr(args, "outer_steps", 2),
    learning_rate=getattr(args, "learning_rate", 0.05),
    max_grad_norm=getattr(args, "max_grad_norm", 1.0),
    optimize_single=which_latent in ("single", "both"),
    optimize_pair=which_latent in ("pair", "both"),
    single_attr=DEFAULT_SINGLE_REP_ATTR[model_key],
    pair_attr=DEFAULT_PAIR_REP_ATTR[model_key],
    anchor_weight_single=anchor_weight if which_latent in ("single", "both") else 0.0,
    anchor_weight_pair=anchor_weight if which_latent in ("pair", "both") else 0.0,
    bond_length_weight=bond_length_weight,
)
```

For Protenix, the same seam disables the diffusion-shared-vars cache when
`guidance_type == LATENT_OPT` so `z_trunk` gradients are not blocked by a stale cache.

### 3.2 Known gaps (as-built)

- **CLI override gap.** The six latent-opt attrs are **not** in `_DYNAMIC_ATTRS`
  (`guidance_script_arguments.py`), so `GuidanceConfig.from_cli` does not copy
  CLI-provided values onto the config after `__post_init__`. On the plain CLI path the
  values stay at defaults unless the config is populated another way (e.g. grid search).
  Fix: add the six names to `_DYNAMIC_ATTRS`.
- **Unknown model.** `DEFAULT_*_REP_ATTR[model_key]` raises an uncaught `KeyError` for any
  model string outside the four wired models.
- **Re-detach responsibility.** `SamplerStepOutput.state/.denoised` are not re-detached
  inside `step()`; the loop must (it does — `coords = step_output.state.detach()`).
- **Protenix `s_inputs`** stays frozen (stage-1 output); only `s_trunk`/`z_trunk` are
  optimized, matching the reference.
