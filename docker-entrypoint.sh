#!/bin/bash
# Sampleworks Docker Entrypoint
#
# Usage:
#   docker run sampleworks -e <pixi_env> <script> [args...]
#   docker run sampleworks -e boltz run_grid_search.py --proteins /data/proteins.csv ...
#   docker run sampleworks --params /data/input/params.json --output-dir /data/results
#   docker run sampleworks bash  # interactive shell
#
# Available pixi environments: boltz, protenix, rf3
#
# Examples:
#   # Run grid search with RF3
#   docker run --gpus all -v /data:/data sampleworks \
#     -e rf3 run_grid_search.py \
#     --proteins /data/proteins.csv \
#     --models rf3 \
#     --scalers pure_guidance \
#     --ensemble-sizes "1 4" \
#     --gradient-weights "0.1 0.2" \
#     --output-dir /data/results \
#     --rf3-checkpoint /data/checkpoints/rf3.ckpt

set -e

show_help() {
    cat << 'EOF'
Sampleworks - Protein structure prediction with diffusion model guidance

USAGE:
    docker run --gpus all --shm-size=16g sampleworks -e <environment> <script> [arguments...]
    docker run --gpus all --shm-size=16g sampleworks --params <params.json> --output-dir <dir>
    docker run sampleworks bash
    docker run sampleworks --help

IMPORTANT:
    Always use --shm-size=16g (or larger) to avoid shared memory errors with DataLoaders.

OPTIONS:
    -e, --env <env>     Pixi environment to use (boltz, protenix, rf3)
    --params FILE       Run grid search from a flexible params.json file
    -h, --help          Show this help message
    bash                Start an interactive shell

ENVIRONMENTS:
    boltz       For boltz1 and boltz2 models
    protenix    For protenix model  
    rf3         For RF3 model

EXAMPLES:
    # Run generic Diffuse/Sampleworks params mode. The JSON chooses the model/env.
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      --params /data/input/params.json \
      --output-dir /data/results/run-001

    # Run grid search with RF3 model
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      -e rf3 run_grid_search.py \
      --proteins /data/proteins.csv \
      --models rf3 \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --gradient-normalization \
      --augmentation \
      --align-to-input \
      --rf3-checkpoint /data/checkpoints/rf3_foundry_01_24_latest.ckpt

    # Run grid search with Boltz1 model
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      -e boltz run_grid_search.py \
      --proteins /data/proteins.csv \
      --models boltz1 \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --boltz1-checkpoint /data/checkpoints/boltz1_conf.ckpt

    # Run grid search with Boltz2 model
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      -e boltz run_grid_search.py \
      --proteins /data/proteins.csv \
      --models boltz2 \
      --scalers pure_guidance \
      --methods "X-RAY DIFFRACTION" \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --boltz2-checkpoint /data/checkpoints/boltz2_conf.ckpt

    # Run grid search with Protenix model
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      -e protenix run_grid_search.py \
      --proteins /data/proteins.csv \
      --models protenix \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --protenix-checkpoint /data/checkpoints/protenix_base_default_v0.5.0.pt

    # Interactive shell
    docker run --gpus all --shm-size=16g -it sampleworks bash

    # Run a custom script
    docker run --gpus all --shm-size=16g -v /data:/data sampleworks \
      -e boltz scripts/boltz2_pure_guidance.py \
      --structure /data/structure.cif \
      --density /data/density.ccp4 \
      --resolution 1.8

GRID SEARCH ARGUMENTS (run_grid_search.py):
    Required:
      --proteins FILE             CSV file with columns: structure,density,resolution,name

    Model selection:
      --models MODEL              Model to use (boltz1, boltz2, protenix, rf3)
                                  Note: Only one model per run currently supported

    Guidance configuration:
      --scalers SCALER            Guidance method (pure_guidance, fk_steering)
      --ensemble-sizes "N M..."   Space-separated ensemble sizes (e.g., "1 4")
      --gradient-weights "X Y..." Space-separated gradient weights (e.g., "0.1 0.2")
      --gradient-normalization    Enable gradient normalization
      --augmentation              Enable data augmentation
      --align-to-input            Enable alignment to input structure

    Output:
      --output-dir DIR            Output directory for results
      --dry-run                   Print commands without executing

    Job control:
      --force-all                 Re-run all jobs including successful ones
      --only-failed               Run only failed jobs
      --only-missing              Run only un-run jobs
      --max-parallel N            Max parallel jobs (default: auto = number of GPUs)

    Model-specific options:
      --boltz1-checkpoint PATH    Path to Boltz1 checkpoint (default: /checkpoints/boltz1_conf.ckpt - BAKED IN)
      --boltz2-checkpoint PATH    Path to Boltz2 checkpoint (default: /checkpoints/boltz2_conf.ckpt - BAKED IN)
      --protenix-checkpoint PATH  Path to Protenix checkpoint
      --rf3-checkpoint PATH       Path to RF3 checkpoint
      --methods METHOD            Boltz2 sampling method (default: "X-RAY DIFFRACTION")

BAKED-IN CHECKPOINTS:
    The following checkpoints are pre-installed in the image:
      /checkpoints/boltz1_conf.ckpt                   - Boltz1 model (~3.5GB)
      /checkpoints/boltz2_conf.ckpt                   - Boltz2 model (~2.3GB)
      /checkpoints/ccd.pkl                             - Chemical Component Dictionary (~345MB)
      /checkpoints/rf3_foundry_01_24_latest.ckpt       - RF3 model (~2.9GB)
      /checkpoints/protenix_base_default_v0.5.0.pt     - Protenix model (~1.4GB)

    FK steering options:
      --num-gd-steps "N M..."     Space-separated GD steps (FK steering only)
      --num-particles N           Number of particles for FK steering (default: 3)
      --fk-lambda FLOAT           Weighting factor for resampling (default: 0.5)
      --fk-resampling-interval N  How often to apply resampling (default: 1)

    Advanced:
      --partial-diffusion-step N  Diffusion step to start from (default: 0)
      --loss-order N              L1 (1) or L2 (2) loss (default: 2)

PROTEINS CSV FORMAT:
    The --proteins CSV file must have the following columns:
      name        - Protein identifier
      structure   - Path to input structure file (.cif, .pdb)
      density     - Path to density map file (.ccp4, .mrc, .map)
      resolution  - Map resolution in Angstroms

    Example:
      name,structure,density,resolution
      1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
      2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8

For full argument details, run:
    docker run sampleworks -e boltz run_grid_search.py --help
EOF
}

infer_env_from_params() {
    local value="$1"
    pixi run -e boltz python - "$value" << 'PY'
import json
import sys

value = sys.argv[1]
with open(value) as handle:
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

# Handle special cases first
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Handle interactive shell
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

# Generic params mode. Diffuse uses this path: it materializes params.json in
# the pod, then passes --params plus --output-dir to this entrypoint.
if [ "$1" = "--params" ]; then
    if [ -z "$2" ] || [[ "$2" == -* ]]; then
        echo "Error: $1 requires a value"
        exit 1
    fi
    ENV="$(infer_env_from_params "$2")"
    exec pixi run -e "$ENV" python /app/run_grid_search.py "$@"
fi

# Parse -e/--env argument
ENV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "Error: -e/--env requires an environment name (boltz, protenix, rf3)"
                exit 1
            fi
            ENV="$2"
            shift 2
            break
            ;;
        *)
            echo "Error: First argument must be -e <environment>, --params, bash, or --help"
            echo ""
            echo "Usage: docker run sampleworks -e <env> <script> [args...]"
            echo "       docker run sampleworks --params <params.json> --output-dir <dir>"
            echo "       docker run sampleworks bash"
            echo "       docker run sampleworks --help"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENV" ]]; then
    echo "Error: Environment not specified. Use -e <env> where env is boltz, protenix, or rf3"
    echo ""
    echo "Usage: docker run sampleworks -e <env> <script> [args...]"
    echo ""
    echo "Examples:"
    echo "  docker run sampleworks -e boltz run_grid_search.py --proteins /data/proteins.csv"
    echo "  docker run sampleworks -e rf3 run_grid_search.py --help"
    echo "  docker run sampleworks bash"
    exit 1
fi

case $ENV in
    boltz|protenix|rf3)
        ;;
    *)
        echo "Error: Invalid environment '$ENV'. Must be one of: boltz, protenix, rf3"
        exit 1
        ;;
esac

# Get the script to run
if [[ $# -eq 0 ]]; then
    echo "Error: No script specified"
    echo "Usage: docker run sampleworks -e <env> <script> [args...]"
    exit 1
fi

SCRIPT="$1"
shift

# If script is "python", run python directly
if [[ "$SCRIPT" == "python" ]]; then
    exec pixi run -e "$ENV" python "$@"
fi

# If script ends in .py, run it with python
if [[ "$SCRIPT" == *.py ]]; then
    # Check if it's a bare script name (like run_grid_search.py)
    if [[ ! -f "$SCRIPT" && -f "/app/$SCRIPT" ]]; then
        SCRIPT="/app/$SCRIPT"
    elif [[ ! -f "$SCRIPT" && -f "/app/scripts/$SCRIPT" ]]; then
        SCRIPT="/app/scripts/$SCRIPT"
    fi
    exec pixi run -e "$ENV" python "$SCRIPT" "$@"
fi

# Otherwise, run command directly via pixi
exec pixi run -e "$ENV" "$SCRIPT" "$@"
