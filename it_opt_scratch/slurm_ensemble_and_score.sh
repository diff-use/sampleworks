#!/usr/bin/env bash
# Generate an ensemble for one target and score it, as a SLURM array task.
#
#   IN    a CSV of targets: name,structure,density,resolution   (one row per target)
#   OUT   <OUTPUT_ROOT>/<name>/refined-patched.cif   the ensemble
#         <OUTPUT_ROOT>/<name>/rscc.csv              per-window RSCC
#         <OUTPUT_ROOT>/<name>/rmsd.csv              per-window min-altloc-RMSD
#         <OUTPUT_ROOT>/rscc_all.csv                 all targets, after `aggregate`
#         <OUTPUT_ROOT>/rmsd_all.csv                 all targets, after `aggregate`
#
# Each array task handles exactly one row, so a failed target fails only its own task and can be
# requeued on its own. Nothing is appended to a shared file until `aggregate`, so tasks never
# race each other.
#
# Stages (argument 1, default `all`):
#   generate    GPU. sample the ensemble, then add the crystallographic header
#   score       CPU (GPU optional). RSCC + min-altloc-RMSD into two per-target CSVs
#   all         both, in one task
#   aggregate   concatenate every per-target CSV into the two final ones (run once, at the end)
#
# Submit as two dependent arrays so the CPU stage does not sit on a GPU:
#
#   N=$(( $(wc -l < targets.csv) - 1 ))
#   gen=$(sbatch --parsable --array=1-$N --gres=gpu:1 --cpus-per-task=6 --mem=64G \
#                --time=2:00:00 it_opt_scratch/slurm_ensemble_and_score.sh generate)
#   scr=$(sbatch --parsable --array=1-$N --dependency=aftercorr:$gen --cpus-per-task=4 --mem=32G \
#                --time=1:00:00 it_opt_scratch/slurm_ensemble_and_score.sh score)
#   sbatch --dependency=afterany:$scr --cpus-per-task=1 --mem=4G \
#          it_opt_scratch/slurm_ensemble_and_score.sh aggregate
#
# Or in one array (simpler, wastes the GPU during scoring):
#   sbatch --array=1-$N --gres=gpu:1 it_opt_scratch/slurm_ensemble_and_score.sh all
#
# Without SLURM it runs row 1 unless you set TASK_ID:
#   TASK_ID=3 bash it_opt_scratch/slurm_ensemble_and_score.sh all
#
# BEFORE SUBMITTING: the header-patching step downloads the PDB entry from RCSB. Compute nodes
# on many clusters have no outbound network. Warm the cache on a login node first by running the
# `generate` stage for every target there, or pre-populate ~/.sampleworks/rcsb -- otherwise every
# task fails at step 2. See the note in check_prerequisites below.

set -euo pipefail

# ------------------------------- settings -------------------------------
# Override any of these by exporting them before sbatch, e.g. `MODE=z_only sbatch ...`.
REPO="${REPO:-/home/dev/workspace}"
TARGETS="${TARGETS:-$REPO/it_opt_scratch/targets.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/it_opt_scratch/slurm_out}"
SELECTIONS="${SELECTIONS:-$REPO/it_opt_scratch/paper_maxrmsd_selections.csv}"
PROCESSED_DIR="${PROCESSED_DIR:-/home/dev/test_data/processed}"

MODE="${MODE:-s_plus_z}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-8}"
BOND_LENGTH_WEIGHT="${BOND_LENGTH_WEIGHT:-5e-5}"
NUM_STEPS="${NUM_STEPS:-200}"

GEN_ENV="${GEN_ENV:-protenix-dev}"        # has protenix + torch
ANALYSIS_ENV="${ANALYSIS_ENV:-analysis}"  # has gemmi + the density tooling
# ------------------------------------------------------------------------

STAGE="${1:-all}"

# aggregate only reads OUTPUT_ROOT, so it runs anywhere -- no repo, no pixi, no GPU.
if [[ "$STAGE" == "aggregate" ]]; then
    # Keep the header from the first file, skip it in the rest.
    for metric in rscc rmsd; do
        out="$OUTPUT_ROOT/${metric}_all.csv"
        first=1
        : > "$out"
        for f in "$OUTPUT_ROOT"/*/"${metric}.csv"; do
            [[ -e "$f" ]] || continue
            if [[ $first == 1 ]]; then cat "$f"; first=0; else tail -n +2 "$f"; fi
        done >> "$out"
        echo "[aggregate] $(( $(wc -l < "$out") - 1 )) rows -> $out"
    done
    exit 0
fi

cd "$REPO"

# ---- which target is this task? Row 1 of the CSV is the header, so add one. ----
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-1}}"
row=$(( TASK_ID + 1 ))
line=$(sed -n "${row}p" "$TARGETS")
[[ -n "$line" ]] || { echo "no row $row in $TARGETS" >&2; exit 1; }
IFS=, read -r NAME STRUCTURE DENSITY RESOLUTION <<< "$line"

# The window list and the PDB header lookup are keyed by the bare 4-character PDB id, while the
# target name usually carries a suffix describing the map (e.g. 2YL0_0.5occA_0.5occB).
PDB="${NAME%%_*}"
OUT_DIR="$OUTPUT_ROOT/$NAME"
mkdir -p "$OUT_DIR"

echo "[task $TASK_ID] $NAME (pdb $PDB) stage=$STAGE mode=$MODE ens=$ENSEMBLE_SIZE"
echo "  structure  $STRUCTURE"
echo "  density    $DENSITY  @ ${RESOLUTION} A"
echo "  out        $OUT_DIR"

if [[ "$STAGE" == "generate" || "$STAGE" == "all" ]]; then
    echo "[1/4] sampling the ensemble"
    pixi run -e "$GEN_ENV" python -u it_opt_scratch/run_targets_simplified.py \
        --structure "$STRUCTURE" \
        --density "$DENSITY" \
        --resolution "$RESOLUTION" \
        --mode "$MODE" \
        --ensemble-size "$ENSEMBLE_SIZE" \
        --num-steps "$NUM_STEPS" \
        --bond-length-weight "$BOND_LENGTH_WEIGHT" \
        --name "$NAME" \
        --skip-existing \
        --output-dir "$OUT_DIR"

    # RSCC needs the unit cell and space group, which sampling does not write. This fetches them
    # from the PDB entry (network!) and writes refined-patched.cif alongside refined.cif.
    echo "[2/4] adding the crystallographic header"
    pixi run -e "$ANALYSIS_ENV" python scripts/patch_output_cif_files.py \
        --input-dir "$OUT_DIR" \
        --depth 1 \
        --cif-pattern refined.cif \
        --rcsb-pattern "($PDB)" \
        --grid-search-input-dir "$PROCESSED_DIR"
fi

if [[ "$STAGE" == "score" || "$STAGE" == "all" ]]; then
    SCORED_CIF="$OUT_DIR/refined-patched.cif"
    [[ -f "$SCORED_CIF" ]] || { echo "missing $SCORED_CIF -- did the generate stage finish?" >&2; exit 1; }
    REFERENCE="$PROCESSED_DIR/$PDB/${PDB}_single_001_density_input.cif"

    echo "[3/4] RSCC"
    pixi run -e "$ANALYSIS_ENV" python it_opt_scratch/score_rscc_simplified.py \
        --prediction "$SCORED_CIF" \
        --reference "$REFERENCE" \
        --map "$DENSITY" \
        --resolution "$RESOLUTION" \
        --selections-csv "$SELECTIONS" \
        --protein "$PDB" \
        --out "$OUT_DIR/rscc.csv"

    echo "[4/4] min-altloc-RMSD"
    pixi run -e "$ANALYSIS_ENV" python it_opt_scratch/score_rmsd_simplified.py \
        --prediction "$SCORED_CIF" \
        --reference "$REFERENCE" \
        --selections-csv "$SELECTIONS" \
        --protein "$PDB" \
        --out "$OUT_DIR/rmsd.csv"
fi

echo "[task $TASK_ID] $NAME done"
