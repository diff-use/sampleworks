#!/bin/bash
# Run Sampleworks grid search natively inside the current ACTL/Kubernetes container.

set -euo pipefail

show_help() {
    cat << 'EOF'
Run Sampleworks run_grid_search.py inside the already-running container.

Usage:
  run_grid_search_actl.sh [-e ENV] [run_grid_search.py args...]
  run_grid_search_actl.sh --params /data/input/params.json --output-dir /data/results
  run_grid_search_actl.sh --model rf3 --proteins /data/input/proteins.csv --output-dir /data/results

Options:
  -e, --env ENV   Pixi environment to use: boltz, protenix, or rf3.
                  If omitted, inferred from --model/--models or --params.
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

infer_env_from_params() {
    pixi run -e boltz python - "$1" << 'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    params = json.load(handle)

if isinstance(params, dict) and isinstance(params.get("params_json"), dict):
    params = params["params_json"]

def model_value(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return model_value(value.get("name") or value.get("type") or value.get("model"))
    if isinstance(value, list):
        if len(value) != 1:
            raise SystemExit("Sampleworks params mode supports exactly one model")
        return str(value[0])
    return str(value)

model = model_value(params.get("model"))
models = params.get("models")
if models is not None:
    if isinstance(models, str):
        models = models.split()
    if not isinstance(models, list) or len(models) != 1:
        raise SystemExit("Sampleworks params mode supports exactly one model")
    if model is not None and str(models[0]) != model:
        raise SystemExit("Sampleworks params JSON defines conflicting model and models values")
    model = str(models[0])

model_section = params.get("model_config") or params.get("model_settings")
if isinstance(model_section, dict):
    nested_model = model_value(
        model_section.get("name") or model_section.get("type") or model_section.get("model")
    )
    if nested_model is not None:
        if model is not None and nested_model != model:
            raise SystemExit("Sampleworks params JSON defines conflicting nested model value")
        model = nested_model

if model in ("boltz1", "boltz2"):
    print("boltz")
elif model == "protenix":
    print("protenix")
elif model == "rf3":
    print("rf3")
else:
    raise SystemExit("params JSON must include model: boltz1, boltz2, protenix, or rf3")
PY
}

ENV_NAME=""
ARGS=()
MODEL=""
PARAMS_FILE=""
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
            ARGS+=("$1" "$2")
            shift 2
            ;;
        --params)
            if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
                echo "Error: --params requires a JSON file path" >&2
                exit 1
            fi
            PARAMS_FILE="$2"
            ARGS+=("$1" "$2")
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

if [[ -n "$PARAMS_FILE" ]]; then
    if [[ ! -f "$PARAMS_FILE" ]]; then
        echo "Error: params JSON not found: $PARAMS_FILE" >&2
        exit 1
    fi
elif [[ -n "$PROTEINS_FILE" ]]; then
    if [[ ! -f "$PROTEINS_FILE" ]]; then
        echo "Error: proteins CSV not found: $PROTEINS_FILE" >&2
        exit 1
    fi
else
    echo "Error: pass --proteins <csv> or --params <json>" >&2
    exit 1
fi

if [[ -z "$ENV_NAME" ]]; then
    if [[ -n "$MODEL" ]]; then
        ENV_NAME="$(infer_env_from_model "$MODEL")"
    elif [[ -n "$PARAMS_FILE" ]]; then
        ENV_NAME="$(infer_env_from_params "$PARAMS_FILE")"
    else
        echo "Error: unable to infer pixi env. Pass -e boltz, -e protenix, or -e rf3." >&2
        exit 1
    fi
fi

case "$ENV_NAME" in
    boltz|protenix|rf3) ;;
    *) echo "Error: invalid pixi environment '$ENV_NAME'. Expected boltz, protenix, or rf3." >&2; exit 1 ;;
esac

exec pixi run -e "$ENV_NAME" python /app/run_grid_search.py "${ARGS[@]}"
