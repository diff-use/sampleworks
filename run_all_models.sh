#!/bin/bash
# Run Boltz1 and Boltz2 grid searches in parallel, 4 GPUs each
# Total: 8 GPUs used (2 jobs x 4 GPUs each)
#
# Checkpoints are BAKED INTO the Docker image - no need to mount them!
#
# Usage:
#   ./run_all_models.sh

set -e

# Configuration - uses absolute path to data
DATA_DIR="/mnt/diffuse-private/raw/sampleworks/initial_dataset_40"
RESULTS_DIR="${RESULTS_DIR:-$HOME/sampleworks-exp/grid_search_results}"

# Create output directory
mkdir -p "$RESULTS_DIR"

# Common docker options
DOCKER_OPTS="--rm --shm-size=16g"

echo "=========================================="
echo "Starting Boltz model grid searches"
echo "Data: $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "Checkpoints: BAKED INTO IMAGE"
echo "=========================================="

# Track background job PIDs
declare -a PIDS=()
declare -a PID_NAMES=()

# Function to run a model with specific GPUs
# run_model launches a Docker container that executes the grid search for the given model and environment on the specified GPUs, streams combined stdout/stderr to "$RESULTS_DIR/<model>_run.log", and records the background PID in PIDS and PID_NAMES.
run_model() {
    local model=$1
    local env=$2
    local gpus=$3
    shift 3
    local extra_args=("$@")

    echo "[$(date)] Starting $model on GPUs $gpus"

    docker run $DOCKER_OPTS \
        --gpus "\"device=$gpus\"" \
        -v /mnt/diffuse-private:/mnt/diffuse-private:ro \
        -v "$RESULTS_DIR:/data/results" \
        sampleworks:latest \
        -e "$env" run_grid_search.py \
        --proteins "$DATA_DIR/proteins.csv" \
        --models "$model" \
        --scalers "pure_guidance" \
        --ensemble-sizes "1 4" \
        --gradient-weights "0.1 0.2" \
        --gradient-normalization --augmentation --align-to-input \
        --use-tweedie \
        --output-dir /data/results \
        "${extra_args[@]}" \
        2>&1 | tee "$RESULTS_DIR/${model}_run.log" &

    PIDS+=($!)
    PID_NAMES+=("$model")
    echo "[$(date)] $model job started (PID: $!)"
}

# Run Boltz1 and Boltz2 in parallel with 4 GPUs each:
# - boltz1: GPUs 0,1,2,3
# - boltz2: GPUs 4,5,6,7

# Boltz1 (GPUs 0-3) - checkpoints baked in, uses defaults
run_model "boltz1" "boltz" "0,1,2,3"

# Boltz2 (GPUs 4-7) - needs --methods flag
run_model "boltz2" "boltz" "4,5,6,7" --methods "X-RAY DIFFRACTION"

echo ""
echo "=========================================="
echo "Both Boltz model jobs launched!"
echo "Logs:"
echo "  - $RESULTS_DIR/boltz1_run.log"
echo "  - $RESULTS_DIR/boltz2_run.log"
echo ""
echo "Monitor GPU usage: nvidia-smi -l 1"
echo "Waiting for all jobs to complete..."
echo "=========================================="

# Wait for all background jobs and check exit codes
overall_exit=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[$(date)] ${PID_NAMES[$i]} completed successfully"
    else
        echo "[$(date)] ${PID_NAMES[$i]} FAILED (exit code: $?)"
        overall_exit=1
    fi
done

echo ""
echo "=========================================="
if [ $overall_exit -eq 0 ]; then
    echo "[$(date)] All jobs completed successfully!"
else
    echo "[$(date)] Some jobs FAILED — check logs above"
fi
echo "=========================================="
exit $overall_exit