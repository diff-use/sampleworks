# Design: CIF output correctness

**Status:** Draft / RFC
**Issues:** [#68](https://github.com/diff-use/sampleworks/issues/68) (valid CIF: ids / entities / header), [#238](https://github.com/diff-use/sampleworks/issues/238) (Protenix chain-ID mismatch)
**Goal:** Write natively valid mmCIF from every model so the post-processing patch
(`scripts/patch_output_cif_files.py`) is no longer required for correctness, and so
downstream evaluation is not silently corrupted for multi-chain inputs.

This doc consolidates the "various things we have to do to fix our CIF outputs" into a
single catalog + phased plan, per the request on #238.

---

## 1. Background

Guidance produces `refined.cif` (and trajectory CIFs) directly from a model's output
`AtomArray`. Today those files are missing header/entity information and, for Protenix,
carry the wrong chain IDs. A separate script, `scripts/patch_output_cif_files.py`,
post-processes each output against the original input/RCSB entry to make it usable by the
eval pipeline (tortoize, phenix, bond-geometry, RSCC). The patch is a workaround; #68 and
#238 both ask us to fix the generation side so no post-processing is needed.

## 2. How each model handles structure processing today

Chain-ID handling is the crux and it differs per model:

| Model    | Path to output AtomArray | Chain IDs |
|----------|--------------------------|-----------|
| **RF3**    | `InferenceInput.from_atom_array(atom_array, chain_info=…)`; `model_aa = pipeline_output["atom_array"].copy()` (`models/rf3/wrapper.py:340,402`) | **Preserved** from the input AtomArray — no lossy round-trip |
| **Boltz**  | YAML carries per-chain `id: {chain_id}` (`models/boltz/wrapper.py:479,493`); echoed back and decoded (`models/boltz/wrapper.py:207`) | **Preserved** |
| **Protenix** | Structure → chain-less JSON → `SampleDictToFeatures(json).get_feature_dict()` rebuilds the AtomArray (`models/protenix/wrapper.py:393`) | **Reinvented** (A, B, C… by entity/copy order) |

Protenix is the outlier: `structure_to_protenix_json()` (`models/protenix/structure_processing.py:515`)
builds an AlphaFold3-style JSON whose entity dicts have **no per-chain `id` slot** — chains
are collapsed into entities with a copy `count` (`structure_processing.py:660`). The real
chain IDs are only set on the *input-side* array (`structure_processing.py:566`,
`set_annotation("label_asym_id", …)`), which is discarded; the array that reaches output
(`models/protenix/wrapper.py:478` `model_aa = cast(AtomArray, atom_array_protenix)`) has
Protenix's invented letters. That array is what gets written to `refined.cif`
(`utils/guidance_script_utils.py:325-342`).

## 3. The CIF write path and what it emits

All native output goes through biotite `CIFFile()` + `set_structure()` + `.write()`:

- `save_everything` → `refined.cif`: `utils/guidance_script_utils.py:339-342` (ensemble becomes a
  multi-model `AtomArrayStack` at `:331-337`).
- `save_structure_to_cif`: `utils/atom_array_utils.py:183-185` (generic; used by synthetic-density
  and altloc-fixed temp CIFs).
- Trajectory CIFs: `utils/guidance_script_utils.py:123,148` via biotite `save_structure` (a
  different, higher-level API).

The input structure is read with atomworks `parse(...)` (`guidance_script_utils.py:236`), which
returns only the `AtomArray` — **the input CIF's header categories are dropped before writing.**

### Requirement-by-requirement status (#68)

| Requirement | Status on the native writers | Evidence |
|---|---|---|
| **(a) unique `atom_site.id` for every row** | **Already satisfied.** Verified on the pinned **biotite 1.4.0**: a 3-model × 5-atom stack writes rows `id = 1..15`, all unique. biotite assigns globally-unique ids across models, including the ensemble case. | pinned `biotite==1.4.0` (`pixi.lock`); reproduced directly |
| **(b) entity / `entity_id`** | **Missing.** No `entity`/`entity_poly` category is ever constructed; `set_structure` only emits `label_entity_id` if the array carries that annotation. | `guidance_script_utils.py:339-342`; patch injects it at `patch_output_cif_files.py:220-231` |
| **(c) header (cell, symmetry, struct, entity_poly, pdbx_poly_seq_scheme, …)** | **Missing.** Only the custom `sampleworks` metadata block is written; everything else was dropped at parse time. | `guidance_script_utils.py:341`; patch recovers header from the template at `patch_output_cif_files.py:233` |

> **Scope correction for #68:** the "every atom_site row needs a unique id" item is *already*
> met by the current biotite writer — the patch overwrites `id` (`patch_output_cif_files.py:242`)
> only because it manipulates a *template* CIF (res-id remap, NaN drop, coord re-insert) whose
> original id column no longer matches. The remaining native gaps are **(b) entities** and
> **(c) header**.

## 4. What `patch_output_cif_files.py` does (the behavior a native fix must absorb)

Per output CIF (`patch_output_cif_files.py:145-262`): fetch original RCSB/template entry →
remap `res_id` to original numbering (chain-aware) → copy the `sampleworks` metadata → drop
NaN-coord atoms → set `label_entity_id` (single-entity hack, `:220-231`) → `set_structure`
into the template so its **header is preserved** (`:233`) → reconcile `pdbx_poly_seq_scheme`
→ overwrite `atom_site.id` with a unique `arange` (`:245`) → ensure `occupancy`/`B_iso`
columns → write `<stem>-patched.cif`.

### Silent-corruption hazard (multi-chain)

The chain-mismatch guard is currently disabled (`patch_output_cif_files.py:159-168`, from #237):

```python
if cif_key[0] != ref_key[0]:
    logger.error(f"Chain mismatch while remapping residues for {cif_path} vs {reference_path}")
    # return msg          <-- commented out in #237 ("breaks multi-chain stuff for now")
mapping[cif_key] = ref_key[1]
```

With the early-return removed, the loop zips output vs reference chain keys **by position**
(`patch_output_cif_files.py:157`). When Protenix's chain ordering doesn't line up with the
reference's, residues are remapped onto the **wrong chain with no failure** — the concrete
downstream-eval hazard behind the P0. Re-enabling the guard requires the output chain IDs to
actually match the reference (§2), otherwise multi-chain runs simply hard-fail.

## 5. Root causes

1. **Protenix drops chain IDs** because its JSON schema has no per-chain `id` (§2).
2. **Header is discarded at parse time** — atomworks `parse` returns only the `AtomArray`;
   nothing carries cell/symmetry/entity/struct categories to the writer.
3. **No cross-model entity bookkeeping** — only Protenix carries `label_entity_id`
   (`models/protenix/structure_processing.py:241,281,340`); Boltz/RF3 do not, so there is no
   uniform place to build `entity`/`entity_poly`.

## 6. Proposed phased plan

**Phase 0 — verified, no action.** `atom_site.id` uniqueness (#68a) is already handled by
biotite 1.4.0. Remove it from scope; keep a regression test so a biotite bump can't silently
reintroduce non-uniqueness.

**Phase 1 — native header carry-over (#68c), low/medium risk.** At write time in
`save_everything`, use the input CIF (available as `GuidanceConfig.structure`) as a template and
write coordinates into it, mirroring the patch but at generation time. Decisions to settle:
which header categories are trustworthy to copy; how to handle atom-count/ordering differences
between the model array and the input (the model may add/drop atoms); interaction with
`resolve_mixed_hetatm_atom_altlocs`. Prefer a **whitelist copy** (cell, symmetry, `struct`,
`exptl`, `entity`, `entity_poly`) over a full template overwrite to avoid coordinate/row
misalignment.

**Phase 2 — Protenix chain IDs (#238).**
- *2a (scoped, no protenix bump):* for **single-chain / order-aligned multi-chain**, relabel
  `model_atom_array.chain_id` from the reference via the existing `AtomReconciler`
  correspondence (`eval/structure_utils.py:191`) before writing, then re-enable the
  `patch_output_cif_files.py` guard for those cases. Caveat: the reconciler keys on **sorted
  chain position**, not label (`utils/atom_array_utils.py:604-635`,
  `utils/atom_reconciler.py:64-107`), so this is only correct when Protenix's sorted chain
  order matches the reference's.
- *2b (general multi-chain):* needs either an upstream Protenix input format that accepts
  explicit chain IDs (the protenix 1.0/2.0 bump #238 mentions) **or** a sequence/geometry-based
  chain-matching layer in sampleworks (because position-based matching is exactly what breaks
  when copy/entity order is arbitrary — homo-multimers, L/H vs A/B, ligand chains).

**Phase 3 — entity / `entity_poly` generation (#68b).** Build entity bookkeeping that survives
the model round-trip, uniformly across RF3/Boltz/Protenix. Depends on Phase 2 (entities and
chains are linked).

**Phase 4 — retire the patch.** Once Phases 1–3 land, `patch_output_cif_files.py` becomes
optional and can be deleted or reduced to res-id renumbering. **Interim safety:** decide whether
the chain-mismatch guard (§4) should hard-fail rather than silently positional-zip — this trades
silent corruption for a loud error on multi-chain until Phase 2 lands.

## 7. Open questions

- Do we take the Protenix 1.0/2.0 bump as the vehicle for 2b, or invest in a sampleworks-side
  chain matcher we control? (#238 leans toward the bump.)
- For Phase 1, is copying the input header acceptable when the model changed the atom set, or do
  we need to regenerate `entity_poly`/`pdbx_poly_seq_scheme` from the model array?
- Interim: flip the `patch_output_cif_files.py` guard to hard-fail on chain mismatch now (loud
  but blocks current multi-chain patching), or leave as-is until Phase 2?

## 8. Key references

- Protenix chain loss: `models/protenix/structure_processing.py:515,566,660`; `models/protenix/wrapper.py:393,478,505`
- Chain-ID preservation (RF3/Boltz): `models/rf3/wrapper.py:340,402`; `models/boltz/wrapper.py:207,479,493`
- Writers: `utils/guidance_script_utils.py:236,325-342,123,148`; `utils/atom_array_utils.py:183-185`
- Reconciler / position-based matching: `eval/structure_utils.py:191`; `utils/atom_reconciler.py:64-107`; `utils/atom_array_utils.py:604-635`
- Patch + disabled guard: `scripts/patch_output_cif_files.py:145-262`, esp. `:157-168,220-231,233,242`
