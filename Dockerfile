# syntax=docker/dockerfile:1
# Sampleworks - Protein structure prediction with diffusion model guidance
#
# This container includes all three model environments: boltz, protenix, rf3
# Checkpoints are baked into the image at /checkpoints/ via a pre-built base image.
#
# Build:
#   docker build -t pixi-with-checkpoints .
#
# CI builds pull checkpoints automatically from Harbor via:
#   COPY --from=harbor.astera.sh/library/sampleworks-checkpoints:latest
# No checkpoint files are needed in the build context or on the CI runner.
#
# To rebuild the checkpoints base image (only needed when checkpoints change):
#   See /data/users/diffuse/checkpoint-build/ on the GPU server
#
# Run examples:
#   # Show help
#   docker run pixi-with-checkpoints --help
#
#   # Run grid search with Boltz1 (checkpoint baked in)
#   docker run --gpus all -v /data:/data pixi-with-checkpoints \
#     -e boltz run_grid_search.py \
#     --proteins /data/proteins.csv \
#     --models boltz1 \
#     --scalers pure_guidance \
#     --ensemble-sizes "1 4" \
#     --gradient-weights "0.1 0.2" \
#     --output-dir /data/results \
#     --use-tweedie \
#     --gradient-normalization \
#     --augmentation \
#     --align-to-input
#
#   # Run grid search with Boltz2 (checkpoint baked in)
#   docker run --gpus all -v /data:/data pixi-with-checkpoints \
#     -e boltz run_grid_search.py \
#     --proteins /data/proteins.csv \
#     --models boltz2 \
#     --scalers pure_guidance \
#     --methods "X-RAY DIFFRACTION" \
#     --ensemble-sizes "1 4" \
#     --gradient-weights "0.1 0.2" \
#     --output-dir /data/results \
#     --use-tweedie
#
#   # Interactive shell
#   docker run --gpus all -it pixi-with-checkpoints bash
#
# Baked-in checkpoints (from diffuseproject/sampleworks-checkpoints:latest):
#   /checkpoints/boltz1_conf.ckpt                   - Boltz1 model (~3.5GB)
#   /checkpoints/boltz2_conf.ckpt                   - Boltz2 model (~2.3GB)
#   /checkpoints/ccd.pkl                             - Chemical Component Dictionary (~345MB)
#   /checkpoints/mols/                               - Boltz2 molecule data (~2GB)
#   /checkpoints/rf3_foundry_01_24_latest.ckpt       - RF3 model (~2.9GB)
#   /checkpoints/protenix_base_default_v0.5.0.pt     - Protenix model (~1.4GB)
#
# Checkpoints base image:
#   All checkpoints live in harbor.astera.sh/library/sampleworks-checkpoints:latest.
#   To rebuild that image, see /data/users/diffuse/checkpoint-build/ on the GPU server.

# ============================================================================
# Base stage: CUDA + Pixi + common system dependencies
# ============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    # Pixi configuration
    PIXI_HOME=/root/.pixi \
    PATH="/root/.pixi/bin:${PATH}" \
    # Python configuration
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Disable user site packages (isolation)
    PYTHONNOUSERSITE=1 \
    # Optimize CUDA compilation for H100
    TORCH_CUDA_ARCH_LIST="9.0"

# Install system dependencies required for building scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    # Required for some scientific packages
    libffi-dev \
    libssl-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Pixi package manager
RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

# Copy all project files - needed because sampleworks is installed as editable package
# The pypi-dependencies section has: sampleworks = {editable = true, path = "."}
COPY pyproject.toml pixi.lock ./
COPY experiments/ ./experiments/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_grid_search.py ./
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ============================================================================
# Bake in model checkpoints from pre-built base image on Docker Hub
# ============================================================================
# Checkpoints (~10 GB) rarely change, so this layer is placed before pixi
# installs to stay cached even when dependencies update.
COPY --from=harbor.astera.sh/library/sampleworks-checkpoints:latest /checkpoints/ /checkpoints/

# ============================================================================
# Install all three environments: boltz, protenix, rf3
# ============================================================================
# IMPORTANT: Must be a single RUN — separate RUNs create overlay layers that
# duplicate shared conda packages (numpy, CUDA libs, etc.), using ~37 GB vs
# ~14 GB in a single layer. This difference causes disk-full failures on
# smaller CI runners (ubuntu-latest can be 72 GB or 145 GB).
RUN pixi install -e boltz --frozen && \
    pixi install -e protenix --frozen && \
    pixi install -e rf3 --frozen

# ============================================================================
# Pre-compile CUDA extensions to avoid JIT compilation at runtime
# This triggers the dilate_points_cuda extension build
# ============================================================================
RUN pixi run -e boltz python -c "\
from sampleworks.core.forward_models.xray.real_space_density_deps.ops import dilate_atom_centric; \
print('CUDA extensions compiled successfully')" || echo "CUDA extension pre-compilation skipped (no GPU during build)"

# This image carries pixi environments and checkpoints. Runtime source should
# come from ACTL's synced checkout at /home/dev/workspace, not from stale code
# baked into /app during image construction.
RUN rm -rf /app/src /app/scripts /app/experiments /app/run_grid_search.py \
    && mkdir -p /home/dev/workspace

COPY --chmod=755 run_experiments run_experiments.sh run_all_models.sh /usr/local/bin/
RUN printf '\n# ACTL scientist workflow: land in the synced Sampleworks checkout.\nif [[ $- == *i* ]] && [ -z "${SAMPLEWORKS_NO_AUTO_CD:-}" ] && [ -d /home/dev/workspace ]; then\n    cd /home/dev/workspace\nfi\n' >> /root/.bashrc

ENV SAMPLEWORKS_PIXI_PROJECT_DIR=/app \
    SAMPLEWORKS_APP_DIR= \
    SAMPLEWORKS_REQUIRE_PREBUILT_PIXI=1

# Set default checkpoint paths via environment variables
ENV BOLTZ1_CHECKPOINT=/checkpoints/boltz1_conf.ckpt \
    BOLTZ2_CHECKPOINT=/checkpoints/boltz2_conf.ckpt \
    CCD_PATH=/checkpoints/ccd.pkl \
    RF3_CHECKPOINT=/checkpoints/rf3_foundry_01_24_latest.ckpt \
    PROTENIX_CHECKPOINT=/checkpoints/protenix_base_default_v0.5.0.pt

ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]
