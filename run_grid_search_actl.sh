#!/bin/bash
# Run Sampleworks grid search natively inside the current ACTL/Kubernetes container.

set -euo pipefail

show_help() {
    cat << 'EOF'
Run Sampleworks run_grid_search.py inside the already-running container.

Usage:
  run_grid_search_actl.sh [-e ENV] [run_grid_search.py args...]
  run_grid_search_actl.sh --model rf3 --proteins /data/input/proteins.csv --output-dir /data/results

Options:
  -e, --env ENV   Pixi environment to use: boltz, protenix, or rf3.
                  If omitted, inferred from --model/--models.
  -h, --help      Show this help message.

This script must be run from inside the Sampleworks image. It never starts Docker.
EOF
}

infer_env_from_model() {
    case "$1" in
        boltz1|boltz2) echo "boltz" ;;
        protenix) echo "protenix" ;;
        rf3) echo "rf3" ;;
        *) echo "Error: unknown model '$1'. Expected boltz1, boltz2, protenix, or rf3." >&2; exit 1 ;;
    esac
}

ENV_NAME=""
ARGS=()
MODEL="boltz2"
PROTEINS_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--env)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: $1 requires an environment name: boltz, protenix, or rf3" >&2
                exit 1
            fi
            ENV_NAME="$2"
            shift 2
            ;;
        --model|--models)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: $1 requires a model value" >&2
                exit 1
            fi
            MODEL="$2"
            ARGS+=("--model" "$2")
            shift 2
            ;;
        --method|--methods)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: $1 requires a method value" >&2
                exit 1
            fi
            ARGS+=("--method" "$2")
            shift 2
            ;;
        --proteins)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: --proteins requires a CSV file path" >&2
                exit 1
            fi
            PROTEINS_FILE="$2"
            ARGS+=("$1" "$2")
            shift 2
            ;;
        run_grid_search.py|/app/run_grid_search.py)
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    show_help >&2
    exit 1
fi

if [[ -n "$PROTEINS_FILE" ]]; then
    if [[ ! -f "$PROTEINS_FILE" ]]; then
        echo "Error: proteins CSV not found: $PROTEINS_FILE" >&2
        exit 1
    fi
else
    echo "Error: pass --proteins <csv>" >&2
    exit 1
fi

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME="$(infer_env_from_model "$MODEL")"
fi

case "$ENV_NAME" in
    boltz|protenix|rf3) ;;
    *) echo "Error: invalid pixi environment '$ENV_NAME'. Expected boltz, protenix, or rf3." >&2; exit 1 ;;
esac

exec pixi run -e "$ENV_NAME" python /app/run_grid_search.py "${ARGS[@]}"
