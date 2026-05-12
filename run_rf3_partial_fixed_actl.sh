#!/bin/bash
# Run RF3 partial occ-sweep grid search inside the current container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GRID_SEARCH_ACTL="${RUN_GRID_SEARCH_ACTL:-$SCRIPT_DIR/run_grid_search_actl.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
Usage: run_rf3_partial_fixed_actl.sh

Environment overrides: DATA_DIR, RESULTS_DIR, MSA_CACHE_DIR, PROTEINS_CSV, GPU_DEVICES, RF3_CHECKPOINT
EOF
    exit 0
fi

DATA_DIR="${DATA_DIR:-/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/sampleworks-exp/occ_sweep/grid_search_results}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${HOME}/sampleworks-exp/msa_cache}"
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"
GPU_DEVICES="${GPU_DEVICES:-4}"

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"

echo "[$(date)] Starting RF3 occ-sweep partial run on GPUs $GPU_DEVICES"
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$RUN_GRID_SEARCH_ACTL" \
    --model rf3 \
    --proteins "$PROTEINS_CSV" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.0 0.005 0.01 0.02 0.035 0.05 0.1" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR" \
    --model-checkpoint "${RF3_CHECKPOINT:-/checkpoints/rf3_foundry_01_24_latest.ckpt}" \
    2>&1 | tee "$RESULTS_DIR/rf3_partial_run_occ_sweep.log"

echo "[$(date)] RF3 job completed."
