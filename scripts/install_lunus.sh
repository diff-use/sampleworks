#!/usr/bin/env bash
# Install lunus.sf -- differentiable structure factors and diffuse scattering --
# into the active environment, from its own repository.
#
# lunus is developed at github.com/lanl/lunus. It is deliberately not vendored
# into this repo and not synced to pods (see .actlignore), so each environment
# that needs the diffuse guidance target installs it once with this script.
#
# This is a stopgap. The durable form is a pypi-dependency in pyproject.toml,
# alongside rc-foundry. That cannot be locked from macOS: lunus is a git source
# dependency with no published wheel, so pixi must build it on the target
# platform, and a linux-64 environment cannot be built on osx-arm64. Once the
# lock is regenerated on linux-64, the declaration replaces this script.
# See DIFFUSE_SCATTERING_PLAN.md.
#
# Usage (from the repo root, inside the environment you want it in):
#   pixi run -e boltz bash scripts/install_lunus.sh
#   LUNUS_REF=main pixi run -e boltz bash scripts/install_lunus.sh
set -euo pipefail

LUNUS_REPO="${LUNUS_REPO:-https://github.com/lanl/lunus.git}"
# The differentiable engine lives on the `sf` branch; switch this to main once
# that branch merges.
LUNUS_REF="${LUNUS_REF:-sf}"

echo "Installing lunus[sf] from ${LUNUS_REPO}@${LUNUS_REF}"

# The `sf` extra requires torch and gemmi. Both are already supplied by the pixi
# environment, and lunus leaves them unpinned, so pip treats them as satisfied
# and will not pull a second copy of torch over the conda-provided one.
# --upgrade forces a re-fetch, so re-running picks up new commits on a branch ref.
python -m pip install --upgrade "lunus[sf] @ git+${LUNUS_REPO}@${LUNUS_REF}"

# Verify rather than trust the exit status. lunus.sf resolves its public API
# lazily (PEP 562), so `import lunus.sf` alone succeeds even when the torch-side
# modules are broken or absent -- name the symbols the diffuse reward needs.
python - <<'PY'
from importlib.metadata import version

from lunus.sf import mean_and_diffuse, structure_factors_batch

print(f"lunus {version('lunus')} OK")
print(f"  structure_factors_batch <- {structure_factors_batch.__module__}")
print(f"  mean_and_diffuse        <- {mean_and_diffuse.__module__}")
PY
