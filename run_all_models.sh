#!/usr/bin/env bash
# ACTL-native entry point for Sampleworks preset runs.
#
# The TOML preset is the source of truth. This wrapper only supplies smooth
# pod defaults: persistent /mnt paths, the synced PR source tree on PYTHONPATH,
# and direct use of the prebuilt pixi environments from the image at /app.

set -euo pipefail

script_path="${BASH_SOURCE[0]}"
while [[ -L "$script_path" ]]; do
    script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
    script_target="$(readlink "$script_path")"
    if [[ "$script_target" == /* ]]; then
        script_path="$script_target"
    else
        script_path="$script_dir/$script_target"
    fi
done
script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd)"
repo_root="${SAMPLEWORKS_APP_DIR:-$script_dir}"

preset="${SAMPLEWORKS_PRESET:-all_models}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    preset="$1"
    shift
fi

if [[ "$preset" == *.toml || "$preset" == */* ]]; then
    if [[ "$preset" != /* ]]; then
        preset="$repo_root/$preset"
    fi
fi
preset_label="${preset##*/}"
preset_label="${preset_label%.toml}"

run_name="${SAMPLEWORKS_ACTL_RUN_NAME:-$(hostname -s 2>/dev/null || printf 'sampleworks')}"
default_data_dir="/mnt/diffuse-shared/raw/sampleworks/initial_dataset_40_occ_sweeps"
default_results_dir="/mnt/diffuse-shared/results/sampleworks/${run_name}/${preset_label}"
default_msa_cache_dir="/mnt/diffuse-shared/cache/sampleworks/msa"

export DATA_DIR="${DATA_DIR:-${SAMPLEWORKS_DATA_DIR:-$default_data_dir}}"
export RESULTS_DIR="${RESULTS_DIR:-${SAMPLEWORKS_RESULTS_DIR:-$default_results_dir}}"
export MSA_CACHE_DIR="${MSA_CACHE_DIR:-${SAMPLEWORKS_MSA_CACHE_DIR:-$default_msa_cache_dir}}"
export SAMPLEWORKS_GRID_SEARCH_SCRIPT="${SAMPLEWORKS_GRID_SEARCH_SCRIPT:-$repo_root/run_grid_search.py}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/tmp/pixi-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

shared_checkpoint_dir="/mnt/diffuse-shared/raw/checkpoints"
for checkpoint_var_and_file in \
    "BOLTZ1_CHECKPOINT boltz1_conf.ckpt" \
    "BOLTZ2_CHECKPOINT boltz2_conf.ckpt" \
    "RF3_CHECKPOINT rf3_foundry_01_24_latest.ckpt" \
    "PROTENIX_CHECKPOINT protenix_base_default_v0.5.0.pt"; do
    read -r checkpoint_var checkpoint_file <<<"$checkpoint_var_and_file"
    checkpoint_path="$shared_checkpoint_dir/$checkpoint_file"
    if [[ -z "${!checkpoint_var:-}" && -f "$checkpoint_path" ]]; then
        export "$checkpoint_var=$checkpoint_path"
    fi
done

source_proteins_csv="${PROTEINS_CSV:-$DATA_DIR/proteins.csv}"
if [[ -f "$source_proteins_csv" ]]; then
    # The shared proteins.csv currently contains absolute /data/inputs paths,
    # while ACTL mounts the dataset at /mnt/diffuse-shared. Rewrite a per-run
    # manifest instead of requiring non-root scientists to create /data symlinks.
    manifest_dir="$RESULTS_DIR/_input_manifest"
    manifest_proteins_csv="$manifest_dir/proteins.csv"
    mkdir -p "$manifest_dir"
    legacy_data_dir="/data/inputs"
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "${line//$legacy_data_dir/$DATA_DIR}"
    done <"$source_proteins_csv" >"$manifest_proteins_csv"
    export PROTEINS_CSV="$manifest_proteins_csv"
fi

runner_env="${SAMPLEWORKS_RUNNER_ENV:-rf3}"
pixi_project_dir="${SAMPLEWORKS_PIXI_PROJECT_DIR:-}"
if [[ -z "$pixi_project_dir" ]]; then
    if [[ -f /app/pyproject.toml && -d /app/.pixi ]]; then
        pixi_project_dir="/app"
    else
        pixi_project_dir="$repo_root"
    fi
fi
runner_python="${SAMPLEWORKS_RUNNER_PYTHON:-$pixi_project_dir/.pixi/envs/$runner_env/bin/python}"

needs_runtime_paths=1
for arg in "$@"; do
    case "$arg" in
        --dry-run|--show|--list|-h|--help)
            needs_runtime_paths=0
            ;;
    esac
done

if [[ "$needs_runtime_paths" -eq 1 ]]; then
    if [[ ! -f "${PROTEINS_CSV:-$source_proteins_csv}" ]]; then
        cat >&2 <<EOF
Sampleworks input dataset was not found.

Expected: $source_proteins_csv

On an ACTL sampleworks pod, make sure the diffuse-shared PVC is mounted at
/mnt/diffuse-shared, or override the dataset path, for example:

  DATA_DIR=/mnt/diffuse-shared/raw/sampleworks/<dataset> ./run_all_models.sh

EOF
        exit 2
    fi
    mkdir -p "$RESULTS_DIR" "$MSA_CACHE_DIR"
fi

cat >&2 <<EOF
Sampleworks preset run
  preset:        $preset
  data:          $DATA_DIR
  results:       $RESULTS_DIR
  msa cache:     $MSA_CACHE_DIR
  source:        $repo_root
  pixi project:  $pixi_project_dir
  runner env:    $runner_env
  runner python: $runner_python

EOF

cd "$pixi_project_dir"
if [[ -x "$runner_python" ]]; then
    runner_env_dir="$(cd -- "$(dirname -- "$runner_python")/.." && pwd)"
    export PATH="$runner_env_dir/bin${PATH:+:$PATH}"
    export CONDA_PREFIX="$runner_env_dir"
    export CUDA_HOME="${CUDA_HOME:-$runner_env_dir}"
    export PYTHONNOUSERSITE=1
    exec "$runner_python" -m sampleworks.runs.cli \
        "$preset" \
        --results-dir "$RESULTS_DIR" \
        "$@"
fi

exec pixi run -e "$runner_env" python -m sampleworks.runs.cli \
    "$preset" \
    --results-dir "$RESULTS_DIR" \
    "$@"
