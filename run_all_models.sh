#!/bin/bash
# Run all model grid searches in parallel: Boltz1, Boltz2, Protenix, and RF3
# Total: 16 GPUs used (4 jobs x 4 GPUs each)
#
# Checkpoints are BAKED INTO the Docker image - no need to mount them!
#
# Usage:
#   ./run_all_models.sh

set -e

# Configuration - uses absolute path to data
DATA_DIR="/mnt/diffuse-private/raw/sampleworks/initial_dataset_40"
RESULTS_DIR="${RESULTS_DIR:-$HOME/sampleworks-exp/grid_search_results}"
# Docker image to use (override with IMAGE env var)
IMAGE="${IMAGE:-diffuseproject/sampleworks:latest}"

# Create output directory
mkdir -p "$RESULTS_DIR"

# Common docker options
DOCKER_OPTS="--rm --shm-size=16g"

echo "=========================================="
echo "Starting all model grid searches"
echo "Models: boltz1, boltz2, protenix, rf3"
echo "Data: $DATA_DIR"
echo "Results: $RESULTS_DIR"
echo "Image: $IMAGE"
echo "Checkpoints: BAKED INTO IMAGE"
echo "=========================================="

# Track background job PIDs
declare -a PIDS=()
declare -a PID_NAMES=()

# Function to run a model with specific GPUs
# Usage: run_model <model> <env> <gpus> [extra_args...]
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
        "$IMAGE" \
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

# Run all four models in parallel with 4 GPUs each:
# - boltz1:   GPUs 0,1,2,3
# - boltz2:   GPUs 4,5,6,7
# - protenix: GPUs 8,9,10,11
# - rf3:      GPUs 12,13,14,15

# Boltz1 (GPUs 0-3) - checkpoints baked in, uses defaults
run_model "boltz1" "boltz" "0,1,2,3"

# Boltz2 (GPUs 4-7) - needs --methods flag
run_model "boltz2" "boltz" "4,5,6,7" --methods "X-RAY DIFFRACTION"

# Protenix (GPUs 8-11)
run_model "protenix" "protenix" "8,9,10,11"

# RF3 (GPUs 12-15)
run_model "rf3" "rf3" "12,13,14,15"

echo ""
echo "=========================================="
echo "All model jobs launched!"
echo "Logs:"
echo "  - $RESULTS_DIR/boltz1_run.log"
echo "  - $RESULTS_DIR/boltz2_run.log"
echo "  - $RESULTS_DIR/protenix_run.log"
echo "  - $RESULTS_DIR/rf3_run.log"
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
