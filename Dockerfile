# syntax=docker/dockerfile:1
# Sampleworks - Protein structure prediction with diffusion model guidance
# Multi-stage build for different model environments (Boltz, Protenix, RF3)
#
# Build examples:
#   docker build --target boltz -t sampleworks:boltz .
#   docker build --target protenix -t sampleworks:protenix .
#   docker build --target rf3 -t sampleworks:rf3 .
#
# Run example (with GPU):
#   docker run --gpus all -v /mnt/checkpoints:/checkpoints:ro \
#     sampleworks:boltz scripts/boltz2_pure_guidance.py \
#     --model-checkpoint /checkpoints/boltz2_conf.ckpt \
#     --output-dir /output \
#     --structure /input/structure.cif \
#     --density /input/density.ccp4 \
#     --resolution 1.8

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
    PYTHONNOUSERSITE=1

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

# ============================================================================
# Boltz environment - Primary model for protein structure prediction
# ============================================================================
FROM base AS boltz

# Install boltz environment dependencies
# Using --frozen ensures reproducible builds from lock file
RUN pixi install -e boltz --frozen

# Set up entrypoint to run Python scripts via pixi environment
ENTRYPOINT ["pixi", "run", "-e", "boltz", "python"]
CMD ["--help"]

# ============================================================================
# Protenix environment - Alternative model (requires triton/NVIDIA)
# ============================================================================
FROM base AS protenix

RUN pixi install -e protenix --frozen

ENTRYPOINT ["pixi", "run", "-e", "protenix", "python"]
CMD ["--help"]

# ============================================================================
# RF3 environment - RoseTTAFold3 model
# ============================================================================
FROM base AS rf3

RUN pixi install -e rf3 --frozen

ENTRYPOINT ["pixi", "run", "-e", "rf3", "python"]
CMD ["--help"]
