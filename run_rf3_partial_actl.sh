#!/bin/bash
# Run RF3 partial-diffusion grid search inside the current container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GRID_SEARCH_ACTL="${RUN_GRID_SEARCH_ACTL:-$SCRIPT_DIR/run_grid_search_actl.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
Usage: run_rf3_partial_actl.sh

Environment overrides:
  DATA_DIR       Input directory containing proteins.csv
  RESULTS_DIR    Output directory
  MSA_CACHE_DIR  MSA cache directory
  PROTEINS_CSV   Explicit proteins CSV path
  GPU_DEVICES    CUDA_VISIBLE_DEVICES value
EOF
    exit 0
fi

DATA_DIR="${DATA_DIR:-/mnt/diffuse-private/raw/sampleworks/initial_dataset_40}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/sampleworks-exp/grid_search_results_rf3_partial}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${HOME}/sampleworks-exp/msa_cache}"
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"
GPU_DEVICES="${GPU_DEVICES:-4,5}"

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"

echo "[$(date)] Starting RF3 partial run on GPUs $GPU_DEVICES"
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$RUN_GRID_SEARCH_ACTL" \
    --model rf3 \
    --proteins "$PROTEINS_CSV" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.01 0.02 0.05" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR" \
    --model-checkpoint "${RF3_CHECKPOINT:-/checkpoints/rf3_foundry_01_24_latest.ckpt}" \
    2>&1 | tee "$RESULTS_DIR/rf3_partial_run.log"

echo "[$(date)] RF3 job completed."
