from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sampleworks.utils.guidance_constants import Boltz2Method, GuidanceType, StructurePredictor


def has_params_mode(args: argparse.Namespace) -> bool:
    """Return whether the namespace requests params-file mode.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    bool
        ``True`` when ``--params`` was supplied.
    """
    return bool(getattr(args, "params", None))


def load_run_params(params_path: str) -> tuple[dict[str, Any], str]:
    """Load a Sampleworks run parameter object from a JSON file.

    Parameters
    ----------
    params_path
        Path to a JSON file.

    Returns
    -------
    tuple[dict[str, Any], str]
        Loaded parameter object and a human-readable source string.

    Raises
    ------
    ValueError
        If the path is empty, JSON is invalid, or the parsed value is not an
        object.
    """
    if not str(params_path).strip():
        raise ValueError("--params requires a JSON file path")

    path = Path(params_path).expanduser()
    raw = path.read_text()
    source = f"file:{path}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid params JSON from {source}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Sampleworks params JSON must be an object")

    # Diffuse's full run params shape is {params_json: {...}, output_dir: ...}.
    # Accept that file shape while keeping Sampleworks' contract file-only.
    if isinstance(parsed.get("params_json"), dict):
        inner = dict(parsed["params_json"])
        if "output_dir" in parsed and "output_dir" not in inner:
            inner["output_dir"] = parsed["output_dir"]
        return inner, source

    return parsed, source


def infer_model(params: dict[str, Any]) -> str:
    """Infer the requested Sampleworks model from a params object.

    Parameters
    ----------
    params
        Scientist-owned parameter object.

    Returns
    -------
    str
        One of ``boltz1``, ``boltz2``, ``protenix``, or ``rf3``.
    """
    model = _model_value(params.get("model"))
    models = params.get("models")
    if model is not None and models is not None:
        inferred_models = _models_value(models)
        if inferred_models != [model]:
            raise ValueError("Sampleworks params JSON defines conflicting model and models values")
    elif models is not None:
        inferred_models = _models_value(models)
        model = inferred_models[0]

    model_section = params.get("model_config") or params.get("model_settings")
    if isinstance(model_section, dict):
        nested_model = _model_value(
            model_section.get("name")
            or model_section.get("type")
            or model_section.get("model")
        )
        if nested_model is not None:
            if model is not None and nested_model != model:
                raise ValueError("Sampleworks params JSON defines conflicting nested model value")
            model = nested_model

    if not model:
        raise ValueError("Sampleworks params JSON must include a 'model' field")
    model = str(model)
    valid = {m.value for m in StructurePredictor}
    if model not in valid:
        raise ValueError(f"Unknown Sampleworks model: {model}. Valid options: {sorted(valid)}")
    return model


def _model_value(value: Any) -> str | None:
    """Extract one model name from a scalar or model section.

    Parameters
    ----------
    value
        Model field value.

    Returns
    -------
    str | None
        Extracted model name, if present.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return _model_value(value.get("name") or value.get("type") or value.get("model"))
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Sampleworks params mode supports exactly one model per run")
        value = value[0]
    return str(value)


def _models_value(value: Any) -> list[str]:
    """Extract the models list and reject multi-model params.

    Parameters
    ----------
    value
        Models field value.

    Returns
    -------
    list[str]
        Single-item model list.
    """
    if isinstance(value, str):
        models = value.split()
    elif isinstance(value, (list, tuple)):
        models = [str(item) for item in value]
    else:
        raise ValueError("models must be a string or list containing exactly one model")
    if len(models) != 1:
        raise ValueError("Sampleworks params mode supports exactly one model per run")
    return models


def infer_pixi_env(model: str) -> str:
    """Return the Pixi environment for a Sampleworks model name.

    Parameters
    ----------
    model
        Sampleworks model identifier.

    Returns
    -------
    str
        Pixi environment name.
    """
    if model in (StructurePredictor.BOLTZ_1, StructurePredictor.BOLTZ_2, "boltz1", "boltz2"):
        return "boltz"
    if model in (StructurePredictor.PROTENIX, "protenix"):
        return "protenix"
    if model in (StructurePredictor.RF3, "rf3"):
        return "rf3"
    raise ValueError(f"Unknown Sampleworks model: {model}")


def normalize_space_list(value: Any, *, item_type: type = str, param_name: str) -> str:
    """Normalize scalar/list JSON values to Sampleworks' space-separated CLI strings.

    Parameters
    ----------
    value
        JSON value from params.
    item_type
        Type used to normalize list elements.
    param_name
        Human-readable parameter name for validation errors.

    Returns
    -------
    str
        Existing string values unchanged, list values joined by spaces.
    """
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{param_name} must not be empty")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{param_name} must not be empty")
        try:
            return " ".join(str(item_type(item)) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{param_name} values must be {item_type.__name__}") from exc
    if item_type is not str and isinstance(value, (int, float)):
        try:
            return str(item_type(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{param_name} must be {item_type.__name__}, string, or list") from exc
    raise ValueError(f"{param_name} must be a string or list")


def normalize_optional_single(value: Any, *, param_name: str) -> str:
    """Normalize a scalar or single-item list to a string.

    Parameters
    ----------
    value
        Parameter value.
    param_name
        Human-readable parameter name for validation errors.

    Returns
    -------
    str
        Normalized value.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{param_name} must contain exactly one value")
        return str(value[0])
    return str(value)


def normalize_bool(value: Any, *, param_name: str) -> bool:
    """Normalize a JSON boolean or boolean-like string.

    Parameters
    ----------
    value
        Parameter value.
    param_name
        Human-readable parameter name for validation errors.

    Returns
    -------
    bool
        Normalized boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{param_name} must be a boolean")


def normalize_scalar(value: Any, *, item_type: type, param_name: str) -> Any:
    """Normalize a scalar JSON value.

    Parameters
    ----------
    value
        Parameter value.
    item_type
        Target scalar type.
    param_name
        Human-readable parameter name for validation errors.

    Returns
    -------
    Any
        Converted scalar value.
    """
    if isinstance(value, (list, tuple, dict)):
        raise ValueError(f"{param_name} must be a {item_type.__name__} value")
    try:
        return item_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{param_name} must be a {item_type.__name__} value") from exc


def resolve_output_dir(params: dict[str, Any], cli_output_dir: str | None) -> Path:
    """Resolve output directory with CLI value taking precedence.

    Parameters
    ----------
    params
        Scientist-owned parameter object.
    cli_output_dir
        Value supplied via ``--output-dir``.

    Returns
    -------
    pathlib.Path
        Output directory path.
    """
    paths = params.get("paths") if isinstance(params.get("paths"), dict) else {}
    value = cli_output_dir or params.get("output_dir") or paths.get("output_dir")
    return Path(str(value or "./grid_search_results")).expanduser()


def _lookup(params: dict[str, Any], *keys: str, section: str | None = None) -> Any:
    """Look up a value from top-level params or a nested section.

    Parameters
    ----------
    params
        Parameter object.
    keys
        Candidate keys in priority order.
    section
        Optional nested section to consult before top-level keys.

    Returns
    -------
    Any
        First present value, or ``None``.
    """
    if section and isinstance(params.get(section), dict):
        nested = params[section]
        for key in keys:
            if key in nested:
                return nested[key]
    for key in keys:
        if key in params:
            return params[key]
    return None


def write_inline_proteins_csv(proteins: list[dict[str, Any]], output_dir: Path) -> Path:
    """Materialize inline protein input records to a proteins CSV file.

    Parameters
    ----------
    proteins
        List of objects containing ``name``, ``structure``, ``density``, and
        ``resolution``.
    output_dir
        Run output directory.

    Returns
    -------
    pathlib.Path
        Path to the generated CSV file.
    """
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    path = inputs_dir / "proteins.csv"
    required = ("name", "structure", "density", "resolution")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(required))
        writer.writeheader()
        for index, protein in enumerate(proteins):
            if not isinstance(protein, dict):
                raise ValueError(f"Inline protein #{index + 1} must be an object")
            missing = [key for key in required if key not in protein]
            if missing:
                raise ValueError(
                    f"Inline protein #{index + 1} is missing required keys: {missing}"
                )
            writer.writerow({key: protein[key] for key in required})
    return path


def apply_run_params(args: argparse.Namespace) -> argparse.Namespace:
    """Apply greedy JSON params to an existing argparse namespace.

    Parameters
    ----------
    args
        Parsed arguments from ``run_grid_search.py``.

    Returns
    -------
    argparse.Namespace
        The same namespace, mutated with resolved Sampleworks options.
    """
    params, source = load_run_params(getattr(args, "params", None))
    output_dir = resolve_output_dir(params, getattr(args, "output_dir", None))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = infer_model(params)
    args.model = model
    args.output_dir = str(output_dir)

    inputs_value = params.get("inputs")
    if inputs_value is not None and not isinstance(inputs_value, dict):
        raise ValueError("inputs must be an object")
    inputs = inputs_value or {}
    proteins = (
        params.get("proteins")
        or params.get("proteins_path")
        or inputs.get("proteins")
        or inputs.get("proteins_path")
    )
    if isinstance(proteins, list):
        args.proteins = str(write_inline_proteins_csv(proteins, output_dir))
    elif isinstance(proteins, dict):
        raise ValueError("inputs.proteins must be a file path or a list of protein objects")
    elif proteins:
        args.proteins = str(proteins)
    else:
        raise ValueError("Sampleworks params JSON must define inputs.proteins or proteins_path")

    guidance_mappings: tuple[tuple[str, tuple[str, ...], type], ...] = (
        ("scalers", ("scalers", "scaler"), str),
        ("ensemble_sizes", ("ensemble_sizes", "ensemble_size"), int),
        ("gradient_weights", ("gradient_weights", "gradient_weight"), float),
        ("num_gd_steps", ("num_gd_steps", "num_gd_step"), int),
    )
    for attr, keys, item_type in guidance_mappings:
        value = _lookup(params, *keys, section="guidance")
        if value is not None:
            setattr(
                args,
                attr,
                normalize_space_list(value, item_type=item_type, param_name=keys[0]),
            )

    method = _lookup(params, "method", "methods", section="model")
    if method is None:
        method = _lookup(params, "method", "methods", section="model_config")
    if method is None:
        method = _lookup(params, "method", "methods", section="model_settings")
    if method is not None:
        if model != StructurePredictor.BOLTZ_2:
            raise ValueError("method is only valid for model boltz2")
        method = normalize_optional_single(method, param_name="method")
        valid_methods = {item.value for item in Boltz2Method}
        if method not in valid_methods:
            raise ValueError(f"Unknown Boltz2 method: {method}. Valid options: {sorted(valid_methods)}")
        args.method = method
    elif model != StructurePredictor.BOLTZ_2:
        args.method = None

    scalar_mappings: tuple[tuple[str, tuple[str, ...], str | None, type], ...] = (
        ("partial_diffusion_step", ("partial_diffusion_step",), "guidance", int),
        ("guidance_start", ("guidance_start",), "guidance", int),
        ("loss_order", ("loss_order",), "guidance", int),
        ("num_particles", ("num_particles",), "guidance", int),
        ("fk_lambda", ("fk_lambda",), "guidance", float),
        ("fk_resampling_interval", ("fk_resampling_interval",), "guidance", int),
    )
    for attr, keys, section, item_type in scalar_mappings:
        value = _lookup(params, *keys, section=section)
        if value is not None:
            setattr(args, attr, normalize_scalar(value, item_type=item_type, param_name=keys[0]))

    step_scaler_type = _lookup(params, "step_scaler_type", section="guidance")
    if step_scaler_type is not None:
        step_scaler_type = str(step_scaler_type)
        valid_step_scalers = {"dataspace", "noisespace", "none"}
        if step_scaler_type not in valid_step_scalers:
            raise ValueError(
                f"Unknown step_scaler_type: {step_scaler_type}. "
                f"Valid options: {sorted(valid_step_scalers)}"
            )
        args.step_scaler_type = step_scaler_type

    max_parallel = _lookup(params, "max_parallel", section="execution")
    if max_parallel is not None:
        if max_parallel != "auto":
            normalize_scalar(max_parallel, item_type=int, param_name="max_parallel")
        args.max_parallel = str(max_parallel)

    fk = params.get("fk")
    guidance = params.get("guidance") if isinstance(params.get("guidance"), dict) else {}
    if not isinstance(fk, dict):
        fk = guidance.get("fk") if isinstance(guidance.get("fk"), dict) else {}
    for attr, key, item_type in (
        ("num_gd_steps", "num_gd_steps", int),
        ("num_particles", "num_particles", int),
        ("fk_lambda", "fk_lambda", float),
        ("fk_resampling_interval", "fk_resampling_interval", int),
    ):
        if key in fk:
            value = fk[key]
            if attr == "num_gd_steps":
                value = normalize_space_list(value, item_type=item_type, param_name=key)
            else:
                value = normalize_scalar(value, item_type=item_type, param_name=key)
            setattr(args, attr, value)

    paths = params.get("paths") if isinstance(params.get("paths"), dict) else {}
    model_section = params.get("model") if isinstance(params.get("model"), dict) else {}
    model_config = params.get("model_config") if isinstance(params.get("model_config"), dict) else {}
    model_settings = (
        params.get("model_settings") if isinstance(params.get("model_settings"), dict) else {}
    )
    checkpoint = (
        params.get("model_checkpoint")
        or model_section.get("checkpoint")
        or model_section.get("model_checkpoint")
        or model_config.get("checkpoint")
        or model_config.get("model_checkpoint")
        or model_settings.get("checkpoint")
        or model_settings.get("model_checkpoint")
        or paths.get("model_checkpoint")
    )
    if checkpoint is not None:
        args.model_checkpoint = str(checkpoint)

    for attr in (
        "gradient_normalization",
        "augmentation",
        "align_to_input",
        "em",
        "dry_run",
        "force_all",
        "only_failed",
        "only_missing",
        "disable_chiral_features",
        "track_chiral_features",
    ):
        value = _lookup(params, attr, section="guidance")
        if value is None:
            value = _lookup(params, attr, section="execution")
        if value is not None:
            setattr(args, attr, normalize_bool(value, param_name=attr))

    scalers = args.scalers.split()
    valid_scalers = {item.value for item in GuidanceType}
    invalid_scalers = [scaler for scaler in scalers if scaler not in valid_scalers]
    if invalid_scalers:
        raise ValueError(f"Unknown scalers: {invalid_scalers}. Valid options: {sorted(valid_scalers)}")

    args._sampleworks_params = params
    args._sampleworks_params_source = source
    return args


def _jsonable(value: Any) -> Any:
    """Convert common runtime values to JSON-serializable values.

    Parameters
    ----------
    value
        Arbitrary value.

    Returns
    -------
    Any
        JSON-compatible value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_params_artifacts(args: argparse.Namespace, output_dir: str | Path) -> None:
    """Write original and resolved params artifacts for a params-mode run.

    Parameters
    ----------
    args
        Resolved run namespace.
    output_dir
        Run output directory.
    """
    params = getattr(args, "_sampleworks_params", None)
    if params is None:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.original.json").write_text(json.dumps(params, indent=2, sort_keys=True))
    resolved = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    resolved["params_source"] = getattr(args, "_sampleworks_params_source", None)
    (out / "params.resolved.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))


def sampleworks_runtime_metadata() -> dict[str, Any]:
    """Collect runtime provenance from package metadata and environment.

    Returns
    -------
    dict[str, Any]
        Runtime metadata suitable for run summaries.
    """
    try:
        version = metadata.version("sampleworks")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "version": version,
        "git_sha": os.environ.get("SAMPLEWORKS_GIT_SHA", "unknown"),
        "image_tag": os.environ.get("SAMPLEWORKS_IMAGE_TAG", "unknown"),
    }


def diffuse_runtime_metadata() -> dict[str, Any]:
    """Collect Diffuse runtime metadata from injected environment variables.

    Returns
    -------
    dict[str, Any]
        Diffuse run context.
    """
    return {
        "run_id": os.environ.get("DIFFUSE_RUN_ID"),
        "profile_slug": os.environ.get("DIFFUSE_PROFILE_SLUG"),
        "experiment_id": os.environ.get("DIFFUSE_EXPERIMENT_ID"),
        "api_url": os.environ.get("DIFFUSE_API_URL"),
        "params_path": os.environ.get("DIFFUSE_PARAMS_PATH"),
    }


def write_run_summary(
    *,
    args: argparse.Namespace,
    output_dir: str | Path,
    status: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    total_jobs: int = 0,
    successful_jobs: int = 0,
    failed_jobs: int = 0,
    error: str | None = None,
) -> None:
    """Write the top-level Sampleworks run summary consumed by humans/Diffuse.

    Parameters
    ----------
    args
        Resolved run namespace.
    output_dir
        Run output directory.
    status
        Run status string.
    started_at
        UTC start timestamp.
    finished_at
        UTC finish timestamp. Defaults to ``now``.
    total_jobs
        Number of jobs selected for this run.
    successful_jobs
        Number of successful jobs.
    failed_jobs
        Number of failed jobs.
    error
        Optional error message.
    """
    finished = finished_at or datetime.now(UTC)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_json = out / "results.json"
    summary = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished.isoformat(),
        "runtime_seconds": round((finished - started_at).total_seconds(), 3),
        "sampleworks": sampleworks_runtime_metadata(),
        "diffuse": diffuse_runtime_metadata(),
        "params": {
            "source": getattr(args, "_sampleworks_params_source", None),
            "original_path": str(out / "params.original.json"),
            "resolved_path": str(out / "params.resolved.json"),
        },
        "outputs": {
            "output_dir": str(out),
            "results_json": str(results_json) if results_json.exists() else None,
        },
        "summary": {
            "total": total_jobs,
            "successful": successful_jobs,
            "failed": failed_jobs,
        },
    }
    if error:
        summary["error"] = error
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
