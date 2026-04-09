# syntax=docker/dockerfile:1
# Sampleworks - Protein structure prediction with diffusion model guidance
#
# This container includes all three model environments: boltz, protenix, rf3
# Checkpoints are baked into the image at /checkpoints/ via a pre-built base image.
#
# Build:
#   docker build -t sampleworks .
#
# CI builds pull checkpoints automatically from Docker Hub via:
#   COPY --from=diffuseproject/sampleworks-checkpoints:latest
# No checkpoint files are needed in the build context or on the CI runner.
#
# To rebuild the checkpoints base image (only needed when checkpoints change):
#   See /data/users/diffuse/checkpoint-build/ on the GPU server
#
# Run examples:
#   # Show help
#   docker run sampleworks --help
#
#   # Run grid search with Boltz1 (checkpoint baked in)
#   docker run --gpus all -v /data:/data sampleworks \
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
#   docker run --gpus all -v /data:/data sampleworks \
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
#   docker run --gpus all -it sampleworks bash
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
#   All checkpoints live in diffuseproject/sampleworks-checkpoints:latest on Docker Hub.
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
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_grid_search.py ./
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ============================================================================
# Bake in model checkpoints from pre-built base image on Docker Hub
# ============================================================================
RUN df -h /
COPY --from=diffuseproject/sampleworks-checkpoints:latest /checkpoints/ /checkpoints/
RUN df -h /

# ============================================================================
# Install all three environments: boltz, protenix, rf3
# ============================================================================
# Split into separate layers so changing one env doesn't invalidate the others
RUN df -h / && pixi install -e boltz --frozen && df -h /
RUN df -h / && pixi install -e protenix --frozen && df -h /
RUN df -h / && pixi install -e rf3 --frozen && df -h /

# ============================================================================
# Pre-compile CUDA extensions to avoid JIT compilation at runtime
# This triggers the dilate_points_cuda extension build
# ============================================================================
RUN pixi run -e boltz python -c "\
from sampleworks.core.forward_models.xray.real_space_density_deps.ops import dilate_atom_centric; \
print('CUDA extensions compiled successfully')" || echo "CUDA extension pre-compilation skipped (no GPU during build)"

# Set default checkpoint paths via environment variables
ENV BOLTZ1_CHECKPOINT=/checkpoints/boltz1_conf.ckpt \
    BOLTZ2_CHECKPOINT=/checkpoints/boltz2_conf.ckpt \
    CCD_PATH=/checkpoints/ccd.pkl \
    RF3_CHECKPOINT=/checkpoints/rf3_foundry_01_24_latest.ckpt \
    PROTENIX_CHECKPOINT=/checkpoints/protenix_base_default_v0.5.0.pt

ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]
