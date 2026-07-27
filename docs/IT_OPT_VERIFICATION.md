# IT-opt method verification — results and open problems

Verification of inference-time latent optimization (IT-opt) against the metrics from the
sampleworks-release paper, run on the already-generated 40-protein ensembles (native occupancy) —
no regeneration, no changes to the generated CIFs.

## Method

For each of `baseline` (unguided), `coord_guidance` (the paper's shipped guided method), and `z`
(IT-opt latent optimization, at bond-geometry weights 0 / 5e-5 / 1e-4 / 1e-3), the paper's altloc
metrics were computed on the existing ensembles:

- **Clustering / bimodal capture** — the repo's own `nn_lddt_clustering`
  (`scripts/eval/lddt_evaluation_script.py`), giving `avg_silhouette_to_ref` (0 poor … 1 perfect
  capture of both altlocs) and per-altloc `occupancies`. Reference altlocs and per-segment selections
  come from `updated-init40-analysis-config.csv`; no CIF patching was needed (numbering aligns — the
  1VME spot-check reproduces the paper's own 0.0034).
- **RMSD** — altloc-aware, min over the ensemble to each deposited altloc (`AllAtomRMSD`,
  Kabsch-superimposed); reported both as the *nearer* altloc and as `max(A, B)` (reaches *both*).
- **RSCC + clash** — a local scorer (see "open problems" #2 — this is **not** the repo's exact
  `extract_tight` pipeline).

## Results

Key conditions (default penalized condition = `z 5e-5`):

| dimension (metric)                  | baseline | coord_guid | z (w=0) | z 5e-5 |
|-------------------------------------|----------|------------|---------|--------|
| density fit (RSCC ≥ 0.8)            | 42%      | inert      | 92%     | 92%    |
| accuracy (nearer-altloc RMSD, med.) | 2.14     | inert      | 1.02    | ~1.0   |
| reach both (RMSD max(A,B) med.; ≤2Å)| 2.18; 46%| inert      | 1.3; 54%| ~1.3; 54% |
| diversity (ensembles that split A/B)| 7%       | 8%         | 18%     | 22%    |
| clean bimodal (clustering ≥ 0.5)    | 10.6%    | 10.6%      | 8%      | 9%     |
| clashes (mean)                      | 0.38     | —          | 0.47    | 0.38   |

Bond-geometry weight sweep (mean clash / median clash / RSCC ≥ 0.8; unguided baseline = 0.38 / 0.00 / 42%):

| weight | mean clash | median clash | RSCC ≥ 0.8 | diversity (split) |
|--------|-----------|--------------|------------|-------------------|
| 0      | 0.47      | 0.25         | 92%        | 18%               |
| 5e-5   | 0.38      | 0.25         | 92%        | 22%   **(default)** |
| 1e-4   | 0.39      | 0.25         | 92%        | 17%               |
| 1e-3   | 0.35      | 0.00         | 98%        | 11%               |

**Interpretation.** IT-opt-`z` strongly improves **density fit and accuracy** (RSCC ≥ 0.8 goes
42% → 92%, reproducing the paper's *guided* jump of 45.4% → 96.0%; nearer-altloc RMSD halves). It
**modestly improves diversity** (ensembles reach both altlocs ~3× more often, 7% → 22%) but does
**not** achieve **clean bimodal capture** — the clustering silhouette stays flat and ~80% of
ensembles still collapse to a single conformation. This is consistent with the paper's central
thesis: methods improve density fit without capturing both deposited states. **`coord_guidance` is
inert on every metric** (identical to baseline). The default bond weight `5e-5` is the smallest that
restores mean clash to baseline while keeping the full density gain and the diversity; `1e-3`
over-constrains and erodes diversity (22% → 11%).

## Open problems

1. **`coord_guidance` is inert — discrepancy with the paper.** The paper reports its coordinate
   guidance taking RSCC ≥ 0.8 from 45.4% to 96.0%, but in these runs `coord_guidance` is
   indistinguishable from `baseline` on *every* metric (RSCC, RMSD, clustering, occupancy). In our
   runs it is the *latent* optimization (`z`), not the coordinate guidance, that reproduces the
   paper's density improvement. **Likely cause:** the coordinate-guidance step size / configuration
   differs from the paper's (paper optimal is 0.1 for Protenix). **To resolve:** a `coord_guidance`
   guidance-strength sweep, confirming whether it is a config artifact or a genuine no-op here.

2. **RSCC is a local scorer, not the repo's exact pipeline.** The RSCC/clash numbers above come from
   a homemade scorer, not `scripts/eval/rscc_grid_search_script.py` (`process_group` → differentiable
   density transformer → Kabsch align → `extract_tight` at 2.0 Å → `rscc`). Reassuringly the homemade
   **baseline (42%) matches the paper's baseline (45.4%)**, which calibrates it, but a repo-exact run
   is the last confirmation. **To resolve:** run the repo RSCC script. This needs input restructuring —
   a depth-4 trial-dir tree (`{PROTEIN}_native_occ/{model}_MD/{scaler}/ens{N}_gw{W}/refined.cif`), maps
   and structures co-located under `base_map_dir`, and occupancy-name parsing that accepts
   `native_occ`. The generated ensembles are flat (`targets_out_40/{protein}/{scaler}/refined.cif`),
   so symlinks (not copies) suffice.

3. **Absolute fractions sit above the paper's on the sweep metrics.** Our baseline is 42% (RSCC ≥ 0.8)
   and 10.6% (clustering ≥ 0.5) vs the paper's 45.4% and 1.6%. The RSCC baseline matches; the
   clustering baseline is ~7× higher. **Likely cause:** row population — we score native occupancy
   only (~85 rows over 35 proteins), while the paper aggregates the full 791-segment occupancy sweep,
   a different easy/hard mix. Within-our-runs comparisons are sound; cross-to-paper *absolute*
   fractions are not directly comparable until the population is matched. **To resolve:** score across
   the paper's full occupancy sweep, or restrict to the paper's exact segment set.

4. **Five proteins are unmatchable without patching.** 6RP1, 7Z0E, 4OLE, 8Z76, and 2I6H raise
   "No common atoms found" in both the RMSD and clustering passes (chain/residue-naming mismatch
   between the predicted CIF and the deposited reference), so they drop out of every aggregate (n=35
   of 40 for RMSD, ~85 rows for clustering). **To resolve:** run `scripts/patch_output_cif_files.py`
   first (renumber to the deposited PDB — needs network for `rcsb.fetch`, and the `~/.sampleworks/rcsb`
   cache is currently empty), or add sequence-based atom matching.

## Config recorded in this branch

- Default `bond_length_weight` is now **`5e-5`** (`LatentOptimization.__init__`, the
  `--bond-length-weight` CLI arg, and the `guidance_script_utils` fallback), with the sweep table in
  the `LatentOptimization` docstring. Set `--bond-length-weight 1e-3` for fuller clash cleanup at the
  cost of diversity, or `0` to disable the penalty.
