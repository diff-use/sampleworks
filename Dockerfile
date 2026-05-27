# syntax=docker/dockerfile:1
# Sampleworks - public multi-stage image
#
# The default build produces the public Sampleworks image: dependencies, source,
# and model checkpoints, with no Astera/EXT-specific tooling.
#
# Build public image locally:
#   docker build --platform linux/amd64 -t diffuseproject/pixi-with-checkpoints:local .
#
# Fast context/Dockerfile smoke check without pulling checkpoints or installing
# pixi environments:
#   docker build --target source-check -t sampleworks-source-check .
#
# Private Astera/EXT overlays are built from Dockerfile.astera after this image is
# pushed to the public registry.

# Public default. Astera CI may override this with its Docker Hub cache mirror,
# e.g. harbor.astera.sh/dockerhub-cache/nvidia/cuda:12.4.1-devel-ubuntu22.04.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04

# Public default for the large checkpoint layer. Private builders can override
# this with an internal mirror, but the public Dockerfile itself does not depend
# on Harbor.
ARG CHECKPOINTS_IMAGE=diffuseproject/sampleworks-checkpoints:latest

# ============================================================================
# OS base: CUDA + Pixi + common build/runtime dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS os-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/root/.pixi \
    PATH="/root/.pixi/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST="9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /app

# ============================================================================
# Source/context stage: useful for cheap CI smoke checks
# ============================================================================
FROM os-base AS source

# Copy only what the runtime image needs. The package is installed as an editable
# pixi dependency (`sampleworks = { editable = true, path = "." }`), so source
# must remain in the public image.
COPY pyproject.toml pixi.lock ./
COPY experiments/ ./experiments/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_grid_search.py ./
COPY run_experiments run_experiments.sh run_all_models.sh ./
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 0755 \
        /usr/local/bin/entrypoint.sh \
        ./run_experiments \
        ./run_experiments.sh \
        ./run_all_models.sh \
    && cp ./run_experiments ./run_experiments.sh ./run_all_models.sh /usr/local/bin/

FROM source AS source-check
ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]

# ============================================================================
# Checkpoints: copied from a registry image so the build context stays small
# ============================================================================
FROM ${CHECKPOINTS_IMAGE} AS checkpoints

# ============================================================================
# Pixi environments: install all supported model environments in one layer
# ============================================================================
FROM source AS pixi-envs

# Checkpoints (~10 GB) rarely change, so this layer stays cacheable across most
# source edits and dependency-only rebuilds.
COPY --from=checkpoints /checkpoints/ /checkpoints/

# IMPORTANT: keep these installs in a single RUN. Splitting them into separate
# Docker layers duplicates shared conda packages (numpy, CUDA libs, etc.) and can
# add tens of GB to the image.
RUN --mount=type=cache,target=/root/.cache/pixi \
    --mount=type=cache,target=/root/.cache/rattler \
    --mount=type=cache,target=/root/.cache/uv \
    pixi install -e boltz --frozen && \
    pixi install -e protenix --frozen && \
    pixi install -e rf3 --frozen

# Pre-compile CUDA extensions when the builder supports it. CPU-only builders are
# still allowed to produce the image; the extension can JIT at runtime if needed.
RUN pixi run -e boltz python -c "\
from sampleworks.core.forward_models.xray.real_space_density_deps.ops import dilate_atom_centric; \
print('CUDA extensions compiled successfully')" || echo "CUDA extension pre-compilation skipped (no GPU during build)"

# ============================================================================
# Public runtime: regular Sampleworks image for the public registry
# ============================================================================
FROM pixi-envs AS public

ENV BOLTZ1_CHECKPOINT=/checkpoints/boltz1_conf.ckpt \
    BOLTZ2_CHECKPOINT=/checkpoints/boltz2_conf.ckpt \
    CCD_PATH=/checkpoints/ccd.pkl \
    RF3_CHECKPOINT=/checkpoints/rf3_foundry_01_24_latest.ckpt \
    PROTENIX_CHECKPOINT=/checkpoints/protenix_base_default_v0.5.0.pt

ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]
