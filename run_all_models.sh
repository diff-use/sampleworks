#!/bin/bash
# Run all 4 model grid searches in parallel, 2 GPUs each
# Total: 8 GPUs used (4 jobs x 2 GPUs each)
#
# Models:
#   - Boltz2 X-ray diffraction (GPUs 0,1)
#   - Boltz2 MD               (GPUs 2,3)
#   - RosettaFold3             (GPUs 4,5)
#   - Protenix                 (GPUs 6,7)
#
# Checkpoints are BAKED INTO the Docker image at /checkpoints/.
# If missing, the code auto-falls back to mounted paths.
#
# Usage:
#   ./run_all_models.sh

set -e

# Configuration
DATA_DIR="/mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps"
RESULTS_DIR="${RESULTS_DIR:-/data/sampleworks-exp/occ_sweep/grid_search_results}"
MSA_CACHE_DIR="${MSA_CACHE_DIR:-/data/sampleworks-exp/msa_cache}"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$MSA_CACHE_DIR"

# Common docker options
DOCKER_OPTS="--rm --shm-size=16g"

echo "=========================================="
echo "Starting all model grid searches (4 jobs x 2 GPUs)"
echo "Data: $DATA_DIR"
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

# --- Boltz2 X-ray Diffraction (GPUs 0,1) ---
echo "[$(date)] Starting Boltz2 X-ray on GPUs 0,1"
docker run $DOCKER_OPTS \
    --gpus '"device=0,1"' \
    -v "$DATA_DIR:/data/inputs:ro" \
    -v "$RESULTS_DIR:/data/results" \
    -v "$MSA_CACHE_DIR:/root/.sampleworks/msa" \
    diffuseproject/sampleworks:latest \
    -e boltz run_grid_search.py \
    --proteins "/data/inputs/proteins.csv" \
    --models boltz2 \
    --methods "X-RAY DIFFRACTION" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.1 0.2 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir /data/results \
    2>&1 | tee "$RESULTS_DIR/boltz2_xrd_run.log" &
PIDS+=($!)
echo "[$(date)] Boltz2 X-ray job started (PID: ${PIDS[-1]})"

# --- Boltz2 MD (GPUs 2,3) ---
echo "[$(date)] Starting Boltz2 MD on GPUs 2,3"
docker run $DOCKER_OPTS \
    --gpus '"device=2,3"' \
    -v "$DATA_DIR:/data/inputs:ro" \
    -v "$RESULTS_DIR:/data/results" \
    -v "$MSA_CACHE_DIR:/root/.sampleworks/msa" \
    diffuseproject/sampleworks:latest \
    -e boltz run_grid_search.py \
    --proteins "/data/inputs/proteins.csv" \
    --models boltz2 \
    --methods "MD" \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.1 0.2 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir /data/results \
    2>&1 | tee "$RESULTS_DIR/boltz2_md_run.log" &
PIDS+=($!)
echo "[$(date)] Boltz2 MD job started (PID: ${PIDS[-1]})"

# --- RosettaFold3 (GPUs 4,5) ---
echo "[$(date)] Starting RosettaFold3 on GPUs 4,5"
docker run $DOCKER_OPTS \
    --gpus '"device=4,5"' \
    -v "$DATA_DIR:/data/inputs:ro" \
    -v "$RESULTS_DIR:/data/results" \
    -v "$MSA_CACHE_DIR:/root/.sampleworks/msa" \
    diffuseproject/sampleworks:latest \
    -e rf3 run_grid_search.py \
    --proteins "/data/inputs/proteins.csv" \
    --models rf3 \
    --partial-diffusion-step 120 \
    --scalers pure_guidance \
    --ensemble-sizes "8" \
    --gradient-weights "0.01 0.02 0.05" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir /data/results \
    2>&1 | tee "$RESULTS_DIR/rf3_run.log" &
PIDS+=($!)
echo "[$(date)] RosettaFold3 job started (PID: ${PIDS[-1]})"

# --- Protenix (GPUs 6,7) ---
echo "[$(date)] Starting Protenix on GPUs 6,7"
docker run $DOCKER_OPTS \
    --gpus '"device=6,7"' \
    -v "$DATA_DIR:/data/inputs:ro" \
    -v "$RESULTS_DIR:/data/results" \
    -v "$MSA_CACHE_DIR:/root/.sampleworks/msa" \
    diffuseproject/sampleworks:latest \
    -e protenix run_grid_search.py \
    --proteins "/data/inputs/proteins.csv" \
    --models protenix \
    --scalers pure_guidance \
    --partial-diffusion-step 120 \
    --ensemble-sizes "8" \
    --gradient-weights "0.1 0.2 0.5" \
    --gradient-normalization --augmentation --align-to-input \
    --output-dir /data/results \
    2>&1 | tee "$RESULTS_DIR/protenix_run.log" &
PIDS+=($!)
echo "[$(date)] Protenix job started (PID: ${PIDS[-1]})"

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

# Wait for all background jobs
wait

echo ""
echo "=========================================="
echo "[$(date)] All jobs completed!"
echo "=========================================="
