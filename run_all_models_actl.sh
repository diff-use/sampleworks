#!/bin/bash
# Run all 4 Sampleworks model grid searches in parallel inside the current container.
#
# This is the ACTL/Kubernetes-native version of run_all_models.sh. It intentionally
# does not run Docker; launch/connect to the sampleworks image first, then run this
# script in that container.
#
# Usage:
#   run_all_models_actl.sh
#
# Environment overrides:
#   DATA_DIR       Input directory containing proteins.csv (default: /data/input)
#   RESULTS_DIR    Output directory (default: /data/results)
#   MSA_CACHE_DIR  MSA cache directory (default: /root/.sampleworks/msa)

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat << 'EOF'
Run all 4 Sampleworks model grid searches in parallel inside the current container.

Usage:
  run_all_models_actl.sh

Environment overrides:
  DATA_DIR       Input directory containing proteins.csv (default: /data/input)
  RESULTS_DIR    Output directory (default: /data/results)
  MSA_CACHE_DIR  MSA cache directory (default: /root/.sampleworks/msa)
  PROTEINS_CSV   Explicit proteins CSV path (default: $DATA_DIR/proteins.csv)
EOF
    exit 0
fi

DATA_DIR="${DATA_DIR:-/data/input}"
RESULTS_DIR="${RESULTS_DIR:-/data/results}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-/root/.sampleworks/msa}"
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    echo "Set DATA_DIR or PROTEINS_CSV to the mounted Sampleworks input path." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"

echo "=========================================="
echo "Starting all model grid searches natively (4 jobs x 2 GPUs)"
echo "Data: $DATA_DIR"
echo "Proteins CSV: $PROTEINS_CSV"
echo "Results: $RESULTS_DIR"
echo "MSA Cache: $MSA_CACHE_DIR"
echo "Checkpoints: BAKED INTO IMAGE (with mount fallback)"
echo ""
echo "Models:"
echo "  - Boltz2 X-ray (GPUs 0,1)"
echo "  - Boltz2 MD    (GPUs 2,3)"
echo "  - RF3          (GPUs 4,5)"
echo "  - Protenix     (GPUs 6,7)"
echo "=========================================="

PIDS=()
NAMES=()

start_grid_search() {
    local name="$1"
    local gpu_devices="$2"
    local pixi_env="$3"
    local log_file="$4"
    shift 4

    echo "[$(date)] Starting $name on GPUs $gpu_devices"
    CUDA_VISIBLE_DEVICES="$gpu_devices" \
        pixi run -e "$pixi_env" python /app/run_grid_search.py "$@" \
        2>&1 | tee "$log_file" &
    PIDS+=("$!")
    NAMES+=("$name")
    echo "[$(date)] $name job started (PID: ${PIDS[-1]})"
}

start_grid_search "Boltz2 X-ray" "0,1" "boltz" "$RESULTS_DIR/boltz2_xrd_run.log" \
    --proteins "$PROTEINS_CSV" \
    --model boltz2 \
    --method "X-RAY DIFFRACTION" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.0 0.05 0.1 0.2 0.35 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR"

start_grid_search "Boltz2 MD" "2,3" "boltz" "$RESULTS_DIR/boltz2_md_run.log" \
    --proteins "$PROTEINS_CSV" \
    --model boltz2 \
    --method "MD" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.0 0.05 0.1 0.2 0.35 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR"

start_grid_search "RosettaFold3" "4,5" "rf3" "$RESULTS_DIR/rf3_run.log" \
    --proteins "$PROTEINS_CSV" \
    --model rf3 \
    --partial-diffusion-step 120 \
    --scalers pure_guidance \
    --ensemble-sizes "8" \
    --gradient-weights "0.0 0.005 0.01 0.02 0.035 0.05 0.1" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR"

start_grid_search "Protenix" "6,7" "protenix" "$RESULTS_DIR/protenix_run.log" \
    --proteins "$PROTEINS_CSV" \
    --model protenix \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.0 0.05 0.1 0.2 0.35 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "All 4 jobs launched! PIDs: ${PIDS[*]}"
echo "Logs:"
echo "  - $RESULTS_DIR/boltz2_xrd_run.log"
echo "  - $RESULTS_DIR/boltz2_md_run.log"
echo "  - $RESULTS_DIR/rf3_run.log"
echo "  - $RESULTS_DIR/protenix_run.log"
echo ""
echo "Monitor GPU usage: nvidia-smi -l 1"
echo "Waiting for all jobs to complete..."
echo "=========================================="

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

echo ""
echo "=========================================="
if [[ "$failed" -eq 0 ]]; then
    echo "[$(date)] All jobs completed successfully!"
else
    echo "[$(date)] One or more jobs failed. Check logs above."
fi
echo "=========================================="

exit "$failed"
