#!/bin/bash
# Alias for the fixed occ-sweep RF3 ACTL run, matching the remote wrapper name.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR="${RESULTS_DIR:-/data/sampleworks-exp/occ_sweep/grid_search_results}" \
MSA_CACHE_DIR="${MSA_CACHE_DIR:-/data/sampleworks-exp/msa_cache}" \
exec "$SCRIPT_DIR/run_rf3_partial_fixed_actl.sh" "$@"
