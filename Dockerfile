# syntax=docker/dockerfile:1
# Sampleworks - Protein structure prediction with diffusion model guidance
#
# This container includes all three model environments: boltz, protenix, rf3
# Checkpoints are baked into the image at /checkpoints/
#
# Build:
#   docker build -t sampleworks .
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
# Baked-in checkpoints:
#   /checkpoints/boltz1_conf.ckpt  - Boltz1 model
#   /checkpoints/boltz2_conf.ckpt  - Boltz2 model  
#   /checkpoints/ccd.pkl           - Chemical Component Dictionary (required for Boltz)

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
# Install all three environments: boltz, protenix, rf3
# ============================================================================
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

# ============================================================================
# Download and bake in model checkpoints
# ============================================================================
RUN mkdir -p /checkpoints && \
    echo "Downloading Boltz1 checkpoint..." && \
    curl -L -o /checkpoints/boltz1_conf.ckpt \
        "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt" && \
    echo "Downloading Boltz2 checkpoint..." && \
    curl -L -o /checkpoints/boltz2_conf.ckpt \
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt" && \
    echo "Downloading CCD (Chemical Component Dictionary)..." && \
    curl -L -o /checkpoints/ccd.pkl \
        "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl" && \
    echo "Downloading Boltz2 molecules data..." && \
    curl -L -o /checkpoints/mols.tar \
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar" && \
    cd /checkpoints && tar -xf mols.tar && rm mols.tar && \
    echo "All checkpoints downloaded!" && \
    ls -lh /checkpoints/

# Set default checkpoint paths via environment variables
ENV BOLTZ1_CHECKPOINT=/checkpoints/boltz1_conf.ckpt \
    BOLTZ2_CHECKPOINT=/checkpoints/boltz2_conf.ckpt \
    CCD_PATH=/checkpoints/ccd.pkl

ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]
