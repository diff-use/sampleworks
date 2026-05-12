#!/bin/bash
# Run RF3 and Protenix grid searches inside the current container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GRID_SEARCH_ACTL="${RUN_GRID_SEARCH_ACTL:-$SCRIPT_DIR/run_grid_search_actl.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
Usage: run_rf3_protenix_mdc_actl.sh

Environment overrides: DATA_DIR, RESULTS_DIR, MSA_CACHE_DIR, PROTEINS_CSV, RF3_GPU_DEVICES, PROTENIX_GPU_DEVICES
EOF
    exit 0
fi

DATA_DIR="${DATA_DIR:-/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/sampleworks-exp/occ_sweep/grid_search_results}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${HOME}/sampleworks-exp/msa_cache}"
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"
RF3_GPU_DEVICES="${RF3_GPU_DEVICES:-0,1,2,3}"
PROTENIX_GPU_DEVICES="${PROTENIX_GPU_DEVICES:-4,5,6,7}"

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"

RF3_RESULTS_DIR="$RESULTS_DIR/rf3"
PROTENIX_RESULTS_DIR="$RESULTS_DIR/protenix"

PIDS=()
NAMES=()

echo "[$(date)] Starting RF3 on GPUs $RF3_GPU_DEVICES"
(
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$RF3_GPU_DEVICES" "$RUN_GRID_SEARCH_ACTL" \
        --model rf3 \
        --proteins "$PROTEINS_CSV" \
        --scalers pure_guidance \
        --ensemble-sizes "8" \
        --gradient-weights "0.0 0.01 0.02 0.05 0.1" \
        --gradient-normalization --augmentation --align-to-input \
        --output-dir "$RF3_RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/rf3_run.log"
) &
PIDS+=("$!")
NAMES+=("RF3")

echo "[$(date)] Starting Protenix on GPUs $PROTENIX_GPU_DEVICES"
(
    set -o pipefail
    CUDA_VISIBLE_DEVICES="$PROTENIX_GPU_DEVICES" "$RUN_GRID_SEARCH_ACTL" \
        --model protenix \
        --proteins "$PROTEINS_CSV" \
        --scalers pure_guidance \
        --partial-diffusion-step 120 \
        --ensemble-sizes "8" \
        --gradient-weights "0.0 0.1 0.2 0.5" \
        --gradient-normalization --augmentation --align-to-input \
        --output-dir "$PROTENIX_RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/protenix_run.log"
) &
PIDS+=("$!")
NAMES+=("Protenix")

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[$(date)] ${NAMES[$index]} completed successfully"
    else
        status=$?
        echo "[$(date)] ${NAMES[$index]} failed with exit code $status" >&2
        failed=1
    fi
done

exit "$failed"
