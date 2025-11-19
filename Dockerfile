# Multi-stage build for optimized Pixi deployment
# Build stage: Install dependencies with Pixi
FROM ghcr.io/prefix-dev/pixi:noble-cuda-12.9.1 AS build

WORKDIR /app

# Copy application code
COPY . .

# Install both boltz and protenix environments
RUN pixi install -e boltz --locked && \
    pixi install -e protenix --locked && \
    rm -rf ~/.cache/rattler

# Generate shell-hook activation scripts for both environments
RUN pixi shell-hook -e boltz -s bash > /boltz-hook.sh && \
    echo 'exec "$@"' >> /boltz-hook.sh && \
    pixi shell-hook -e protenix -s bash > /protenix-hook.sh && \
    echo 'exec "$@"' >> /protenix-hook.sh

# Environment variable to select which environment to use (default: boltz)
ENV PIXI_ENV=boltz

# Entrypoint that activates the selected environment
ENTRYPOINT ["/bin/bash", "-c", "if [ \"$PIXI_ENV\" = \"protenix\" ]; then exec /bin/bash /protenix-hook.sh \"$@\"; else exec /bin/bash /boltz-hook.sh \"$@\"; fi", "--"]

CMD ["bash"]
