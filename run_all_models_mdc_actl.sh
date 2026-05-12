#!/bin/bash
# Run selected Sampleworks model grid searches inside the current container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_GRID_SEARCH_ACTL="${RUN_GRID_SEARCH_ACTL:-$SCRIPT_DIR/run_grid_search_actl.sh}"

show_help() {
    cat << 'EOF'
Usage: run_all_models_mdc_actl.sh [OPTIONS]

Options:
  --boltz2-xrd    Run Boltz2 with X-ray diffraction (GPUs 0,1)
  --boltz2-md     Run Boltz2 with MD (GPUs 2,3)
  --rf3           Run RosettaFold3 (GPUs 4,5)
  --protenix      Run Protenix (GPUs 6,7)
  --all           Run all models (default if no args)
  --help          Show this help message

Environment overrides:
  DATA_DIR       Input directory containing proteins.csv
  RESULTS_DIR    Output directory
  MSA_CACHE_DIR  MSA cache directory
  PROTEINS_CSV   Explicit proteins CSV path
EOF
}

DATA_DIR="${DATA_DIR:-/mnt/diffuse-private/raw/sampleworks/initial_dataset_40}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/sampleworks-exp/grid_search_results}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-${HOME}/sampleworks-exp/msa_cache}"
PROTEINS_CSV="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"

RUN_BOLTZ2_XRD=false
RUN_BOLTZ2_MD=false
RUN_RF3=false
RUN_PROTENIX=false

if [[ $# -eq 0 ]]; then
    RUN_BOLTZ2_XRD=true
    RUN_BOLTZ2_MD=true
    RUN_RF3=true
    RUN_PROTENIX=true
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --boltz2-xrd|--boltz-xrd|--boltz2_xrd) RUN_BOLTZ2_XRD=true ;;
            --boltz2-md|--boltz-md|--boltz2_md) RUN_BOLTZ2_MD=true ;;
            --rf3|--rosettafold3|--rosettafold) RUN_RF3=true ;;
            --protenix) RUN_PROTENIX=true ;;
            --all) RUN_BOLTZ2_XRD=true; RUN_BOLTZ2_MD=true; RUN_RF3=true; RUN_PROTENIX=true ;;
            --help|-h) show_help; exit 0 ;;
            *) echo "Unknown option: $1" >&2; echo "Use --help for usage." >&2; exit 1 ;;
        esac
        shift
    done
fi

if [[ ! -f "$PROTEINS_CSV" ]]; then
    echo "Error: proteins CSV not found: $PROTEINS_CSV" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"

echo "=========================================="
echo "Starting selected model grid searches natively"
echo "Data: $DATA_DIR"
echo "Proteins CSV: $PROTEINS_CSV"
echo "Results: $RESULTS_DIR"
echo "MSA Cache: $MSA_CACHE_DIR"
echo "Models to run:"
[[ "$RUN_BOLTZ2_XRD" == true ]] && echo "  - Boltz2 X-ray (GPUs 0,1)"
[[ "$RUN_BOLTZ2_MD" == true ]] && echo "  - Boltz2 MD (GPUs 2,3)"
[[ "$RUN_RF3" == true ]] && echo "  - RosettaFold3 (GPUs 4,5)"
[[ "$RUN_PROTENIX" == true ]] && echo "  - Protenix (GPUs 6,7)"
echo "=========================================="

PIDS=()
NAMES=()

start_grid_search() {
    local name="$1"
    local gpu_devices="$2"
    local log_file="$3"
    shift 3

    echo "[$(date)] Starting $name on GPUs $gpu_devices"
    CUDA_VISIBLE_DEVICES="$gpu_devices" "$RUN_GRID_SEARCH_ACTL" "$@" 2>&1 | tee "$log_file" &
    PIDS+=("$!")
    NAMES+=("$name")
    echo "[$(date)] $name job started (PID: ${PIDS[-1]})"
}

if [[ "$RUN_BOLTZ2_XRD" == true ]]; then
    start_grid_search "Boltz2 X-ray" "0,1" "$RESULTS_DIR/boltz2_xrd_run.log" \
        --model boltz2 --method "X-RAY DIFFRACTION" --proteins "$PROTEINS_CSV" \
        --scalers pure_guidance --partial-diffusion-step 120 --ensemble-sizes "8" \
        --gradient-weights "0.1 0.2 0.5" --gradient-normalization --augmentation \
        --align-to-input --output-dir "$RESULTS_DIR"
fi

if [[ "$RUN_BOLTZ2_MD" == true ]]; then
    start_grid_search "Boltz2 MD" "2,3" "$RESULTS_DIR/boltz2_md_run.log" \
        --model boltz2 --method "MD" --proteins "$PROTEINS_CSV" \
        --scalers pure_guidance --partial-diffusion-step 120 --ensemble-sizes "8" \
        --gradient-weights "0.1 0.2 0.5" --gradient-normalization --augmentation \
        --align-to-input --output-dir "$RESULTS_DIR"
fi

if [[ "$RUN_RF3" == true ]]; then
    start_grid_search "RosettaFold3" "4,5" "$RESULTS_DIR/rf3_run.log" \
        --model rf3 --proteins "$PROTEINS_CSV" --scalers pure_guidance \
        --ensemble-sizes "8" --gradient-weights "0.01 0.02 0.05" \
        --gradient-normalization --augmentation --align-to-input --output-dir "$RESULTS_DIR"
fi

if [[ "$RUN_PROTENIX" == true ]]; then
    start_grid_search "Protenix" "6,7" "$RESULTS_DIR/protenix_run.log" \
        --model protenix --proteins "$PROTEINS_CSV" --scalers pure_guidance \
        --partial-diffusion-step 120 --ensemble-sizes "8" --gradient-weights "0.1 0.2 0.5" \
        --gradient-normalization --augmentation --align-to-input --output-dir "$RESULTS_DIR"
fi

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "No models selected to run." >&2
    exit 1
fi

echo "Waiting for ${#PIDS[@]} jobs: ${PIDS[*]}"
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
