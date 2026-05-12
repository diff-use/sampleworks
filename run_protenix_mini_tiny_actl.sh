#!/bin/bash
# Run Protenix tiny and mini grid searches inside the current container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GRID_SEARCH_ACTL="${RUN_GRID_SEARCH_ACTL:-$SCRIPT_DIR/run_grid_search_actl.sh}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
Usage: run_protenix_mini_tiny_actl.sh

Environment overrides:
  DATA_DIR, MSA_CACHE_DIR, TINY_RESULTS_DIR, MINI_RESULTS_DIR, PROTEINS_CSV
  TINY_GPU_DEVICES, MINI_GPU_DEVICES, PROTENIX_TINY_CHECKPOINT, PROTENIX_MINI_CHECKPOINT
EOF
    exit 0
fi

DATA_DIR="${DATA_DIR:-/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-/data/sampleworks-exp/msa_cache}"
if [[ -n "${RESULTS_DIR:-}" ]]; then
    TINY_RESULTS_DIR="${TINY_RESULTS_DIR:-$RESULTS_DIR/protenix_tiny}"
    MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-$RESULTS_DIR/protenix_mini}"
    TINY_LOG_FILE="$RESULTS_DIR/protenix_tiny_run.log"
    MINI_LOG_FILE="$RESULTS_DIR/protenix_mini_run.log"
else
    TINY_RESULTS_DIR="${TINY_RESULTS_DIR:-/data/sampleworks-exp/occ_sweep/grid_search_results_protenix_tiny}"
    MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-/data/sampleworks-exp/occ_sweep/grid_search_results_protenix_mini}"
    TINY_LOG_FILE="$TINY_RESULTS_DIR/protenix_tiny_run.log"
    MINI_LOG_FILE="$MINI_RESULTS_DIR/protenix_mini_run.log"
fi
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"
TINY_GPU_DEVICES="${TINY_GPU_DEVICES:-2,3}"
MINI_GPU_DEVICES="${MINI_GPU_DEVICES:-6,7}"
PROTENIX_TINY_CHECKPOINT="${PROTENIX_TINY_CHECKPOINT:-/extra_checkpoints/protenix_tiny_default_v0.5.0.pt}"
PROTENIX_MINI_CHECKPOINT="${PROTENIX_MINI_CHECKPOINT:-/extra_checkpoints/protenix_mini_default_v0.5.0.pt}"

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    exit 1
fi
if [[ ! -f "$PROTENIX_TINY_CHECKPOINT" ]]; then
    echo "Error: Protenix tiny checkpoint not found: $PROTENIX_TINY_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$PROTENIX_MINI_CHECKPOINT" ]]; then
    echo "Error: Protenix mini checkpoint not found: $PROTENIX_MINI_CHECKPOINT" >&2
    exit 1
fi
if [[ "$TINY_RESULTS_DIR" == "$MINI_RESULTS_DIR" ]]; then
    echo "Error: TINY_RESULTS_DIR and MINI_RESULTS_DIR must be different for parallel runs." >&2
    exit 1
fi

mkdir -p "$TINY_RESULTS_DIR" "$MINI_RESULTS_DIR" "$(dirname "$TINY_LOG_FILE")" "$(dirname "$MINI_LOG_FILE")" "$MSA_CACHE_DIR"

PIDS=()
NAMES=()

start_protenix() {
    local name="$1"
    local gpu_devices="$2"
    local checkpoint="$3"
    local output_dir="$4"
    local log_file="$5"

    echo "[$(date)] Starting $name on GPUs $gpu_devices"
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="$gpu_devices" "$RUN_GRID_SEARCH_ACTL" \
            --model protenix \
            --proteins "$PROTEINS_CSV" \
            --model-checkpoint "$checkpoint" \
            --scalers pure_guidance \
            --partial-diffusion-step 120 \
            --ensemble-sizes "8" \
            --gradient-weights "0.0 0.05 0.1 0.2 0.35 0.5" \
            --gradient-normalization --augmentation --align-to-input \
            --output-dir "$output_dir" \
            2>&1 | tee "$log_file"
    ) &
    PIDS+=("$!")
    NAMES+=("$name")
    echo "[$(date)] $name job started (PID: ${PIDS[-1]})"
}

start_protenix "Protenix tiny" "$TINY_GPU_DEVICES" "$PROTENIX_TINY_CHECKPOINT" "$TINY_RESULTS_DIR" "$TINY_LOG_FILE"
start_protenix "Protenix mini" "$MINI_GPU_DEVICES" "$PROTENIX_MINI_CHECKPOINT" "$MINI_RESULTS_DIR" "$MINI_LOG_FILE"

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
