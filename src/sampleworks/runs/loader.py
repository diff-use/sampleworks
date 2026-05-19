"""Load presets from TOML and apply runtime overrides.

Resolution order for every string value (defaults block and ``args``):
  1. ``${VAR}`` references are resolved against the process environment,
     with the preset's ``[defaults]`` block filling in any unset keys.
  2. ``--set <dotted-path>=<value>`` CLI overrides are applied last.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any

from .schema import Job, Preset

_BUNDLED_PRESETS_PACKAGE = "sampleworks.runs.presets"
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def list_bundled_presets() -> list[str]:
    """Return the names (sans ``.toml``) of bundled presets, sorted."""
    files = resources.files(_BUNDLED_PRESETS_PACKAGE)
    return sorted(p.name.removesuffix(".toml") for p in files.iterdir() if p.name.endswith(".toml"))


def load_preset(name_or_path: str, *, overrides: Iterable[str] = ()) -> Preset:
    """Load a preset by bundled name or filesystem path, applying ``--set`` overrides."""
    raw = _read_toml(name_or_path)
    overrides_list = list(overrides)
    raw = _apply_overrides(raw, overrides_list)
    raw = _resolve_variables(raw)
    return _build_preset(name=_preset_name(name_or_path), raw=raw)


def _read_toml(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if path.suffix == ".toml" and path.exists():
        return tomllib.loads(path.read_text())
    bundled = resources.files(_BUNDLED_PRESETS_PACKAGE) / f"{name_or_path}.toml"
    if not bundled.is_file():
        raise FileNotFoundError(
            f"No preset {name_or_path!r}. Bundled: {list_bundled_presets()}. "
            f"Or pass a path to a .toml file."
        )
    return tomllib.loads(bundled.read_text())


def _preset_name(name_or_path: str) -> str:
    return Path(name_or_path).stem if name_or_path.endswith(".toml") else name_or_path


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--set expects KEY=VALUE, got {spec!r}")
        key, value = spec.split("=", 1)
        _set_dotted(raw, key.strip(), _coerce(value))
    return raw


def _set_dotted(obj: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``obj`` at ``a.b.c`` to ``value``. Job name lookup is allowed under ``jobs``."""
    parts = dotted.split(".")
    cursor: Any = obj
    for i, part in enumerate(parts[:-1]):
        cursor = _index(cursor, part, where=".".join(parts[: i + 1]))
    leaf_parent = cursor
    leaf_key = parts[-1]
    if isinstance(leaf_parent, list):
        leaf_parent[_find_in_list(leaf_parent, leaf_key, where=dotted)] = value
    else:
        leaf_parent[leaf_key] = value


def _index(cursor: Any, part: str, *, where: str) -> Any:
    if isinstance(cursor, list):
        return cursor[_find_in_list(cursor, part, where=where)]
    if isinstance(cursor, dict):
        if part not in cursor:
            cursor[part] = {}
        return cursor[part]
    raise TypeError(f"Cannot descend into {type(cursor).__name__} at {where!r}")


def _find_in_list(items: list[Any], key: str, *, where: str) -> int:
    if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
        return int(key)
    for i, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == key:
            return i
    raise KeyError(f"No list element named {key!r} at {where!r}")


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_variables(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand ``${VAR}`` in every string. Env wins; defaults block fills gaps.

    Defaults are resolved in TOML order, so later defaults can reference earlier ones
    (e.g. ``PROTEINS_CSV = "${DATA_DIR}/proteins.csv"``).
    """
    defaults: dict[str, str] = dict(raw.get("defaults", {}))
    accumulated: dict[str, str] = dict(os.environ)
    resolved_defaults: dict[str, str] = {}
    for key, default_value in defaults.items():
        if key in os.environ:
            resolved_defaults[key] = os.environ[key]
        else:
            resolved_defaults[key] = _expand(default_value, accumulated)
        accumulated[key] = resolved_defaults[key]
    resolved = _walk(raw, accumulated)
    resolved["defaults"] = resolved_defaults
    return resolved


def _walk(obj: Any, env: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(item, env) for item in obj]
    if isinstance(obj, str):
        return _expand(obj, env)
    return obj


def _expand(text: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in env:
            raise KeyError(f"Undefined variable ${{{var}}} in preset (no env var, no default)")
        return env[var]

    prev = None
    current = text
    while prev != current:
        prev = current
        current = _VAR_PATTERN.sub(repl, current)
    return current


def _build_preset(*, name: str, raw: dict[str, Any]) -> Preset:
    raw_jobs = raw.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise ValueError(f"Preset {name!r}: 'jobs' must be a list")
    jobs = [
        Job(
            name=str(j["name"]),
            env=str(j["env"]),
            gpus=str(j["gpus"]),
            output_subdir=str(j["output_subdir"]),
            args=dict(j.get("args", {})),
        )
        for j in raw_jobs
    ]
    return Preset(
        name=name,
        description=str(raw.get("description", "")),
        defaults=dict(raw.get("defaults", {})),
        shared_args=dict(raw.get("shared_args", {})),
        jobs=jobs,
    )
