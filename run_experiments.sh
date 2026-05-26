#!/usr/bin/env bash
# Backward-compatible alias for the extensionless run_experiments command.
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

exec "$script_dir/run_experiments" "$@"
