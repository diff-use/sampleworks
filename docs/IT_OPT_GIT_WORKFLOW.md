# IT-Opt — Branch State and Remaining Work

**Point-in-time working note (updated 2026-07-20).** This records where the inference-time
latent-optimization (IT-opt) feature stands on the branch, what production files it touches, and
what is left before it can merge, so a future agent does not re-derive it. It is not architecture;
verify every claim against `git`/`gh` before acting, because the branch state drifts.

Companion docs: [IT_OPTIMIZATION_PLAN.md](IT_OPTIMIZATION_PLAN.md),
[IT_OPT_TESTING_PROTENIX.md](IT_OPT_TESTING_PROTENIX.md),
[IT_OPT_REFERENCE_COMPARISON.md](IT_OPT_REFERENCE_COMPARISON.md).

---

## Current state

- The work lives on the experiment branch **`fy/it-opt-z`**, cut off `fy/it-optimization` (which is
  the umbrella feature branch behind draft **PR #313 → `main`**). The IT-opt changes on `fy/it-opt-z`
  are still an uncommitted working tree at the time of writing.
- Both latents now run: the single representation `s` was validated first, and the pair
  representation `z` was enabled by featurizing Protenix with the diffusion shared-variables cache
  off (with the cache on, the `z_trunk` gradient is silently zero — see
  [IT_OPT_TESTING_PROTENIX.md](IT_OPT_TESTING_PROTENIX.md)).
- IT-opt is now a first-class guidance type in the pipeline (see the footprint below), so it is
  reachable both from the CLI (`sampleworks-guidance --guidance-type latent_opt …`) and
  programmatically through `run_guidance`.
- Findings so far (see [IT_OPTIMIZATION_PLAN.md](IT_OPTIMIZATION_PLAN.md) for detail): under the
  density objective `s` is effectively inert while `z` is the lever; `z` raises the paper's altloc
  RSCC but drifts off-manifold and adds a few clashes when unregularized; coordinate guidance is the
  safe, modest baseline it must beat. None of the modes crosses the paper's 0.8 RSCC bar yet.

## Production footprint (what a reviewer should look at)

All of these are on `fy/it-opt-z`; the IT-opt additions are tagged with a greppable `IT-opt wiring`
comment where they sit inside otherwise-unrelated functions.

- `src/sampleworks/utils/guidance_constants.py` — added `GuidanceType.LATENT_OPT`.
- `src/sampleworks/utils/guidance_script_arguments.py` — added `add_latent_opt_args` and its registry
  entry (`--which-latent`, `--learning-rate`, `--outer-steps`, `--anchor-weight`, `--max-grad-norm`).
- `src/sampleworks/utils/guidance_script_utils.py` — added the `LATENT_OPT` dispatch branch in
  `_run_guidance`, the Protenix cache-off decision, and the `LATENT_OPT` case in `save_trajectory`.
- `src/sampleworks/core/scalers/latent_optimization.py` — reverted the joint `[s, z]` grad-clip to
  the reference's per-latent clips, and added the per-round `latent_drift` diagnostic.
- `docs/IT_OPTIMIZATION_PLAN.md`, `docs/IT_OPT_REFERENCE_COMPARISON.md` — corrected the clip
  description to match the code.

## Remaining before this can merge

1. **Productionize the wrapper `IT-OPT TEST EDIT`.** `src/sampleworks/models/protenix/wrapper.py`
   still keeps `s_trunk`/`z_trunk` attached via a commented "TEST EDIT" scaffold. It is behaviorally
   inert for the other guidance types (featurize runs under `no_grad`, so those latents are non-grad
   constants and not detaching them is a no-op), but a scaffold should not land on `main`. Turn it
   into a proper opt-in flag gated on the latent-optimization path.
2. **Optionally add a gradient-presence guard** in `LatentOptimization`: after the first backward,
   assert each optimized leaf received a non-`None` gradient, so a future cache/config mistake fails
   loudly instead of silently optimizing nothing.
3. **Get a defensible efficacy result** before marking PR #313 ready: control the sampling RNG (a
   common fixed seed across modes) and try the anchor on `z` so its RSCC gains land on-manifold
   rather than as clashes.

## Do NOT commit (keep out of the PR)

- `uv.lock` — this project pins dependencies with `pixi.lock`; there is no `[tool.uv]` in
  `pyproject.toml`, so `uv.lock` is a stray resolution and is gitignored.
- The `it_opt_scratch/` directory — throwaway drivers, run outputs, and logs; gitignored.
- The `1vme_final_carved_edited_0.5occa_0.5occb/` scratch inputs and the `AGENTS.md` deletion — these
  are unrelated working-tree changes; keep the IT-opt commit scoped.

## Scratch tooling (in the gitignored `it_opt_scratch/`)

- `run_targets.py` — batch generation through the `run_guidance` interface, over a list of targets
  (CSV columns `name,structure,density,resolution`, or a JSON list). Loads the model once and writes
  `<out_dir>/<mode>/refined.cif` per target and mode.
- `score.py` — scores a target's `run_targets.py` output on the paper's metrics: RSCC over the altloc
  region and a non-bonded heavy-atom clash count, per mode.
